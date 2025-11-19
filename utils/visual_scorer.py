import os
import base64
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
import numpy as np
from pathlib import Path
import Levenshtein
import httpx
import easyocr
import tempfile
from .core import call_mllm

load_dotenv()

COMPARE_TEMPLATE = """You are an expert Webpage Evaluator. Your task is to provide a quantitative and qualitative assessment of the similarity between a generated webpage and a reference webpage.
The default score is 0.
Evaluation Format:
---
Comments:
-Layout (10 points): ${comment and subscore}
-Elements (15 points): ${comment and subscore}
-Content and Text (40 points): ${comment and subscore}
-Style (15 points): ${comment and subscore}
-Overall (20 points): ${comment and subscore}

Score: ${final score}/100
---"""


class VisualScorer:
    proxy_url = os.getenv('IWR_PROXY', '')

    def __init__(self):
        http_client = None
        if self.proxy_url:
            http_client = httpx.Client(proxies=self.proxy_url, transport=httpx.HTTPTransport(local_address='0.0.0.0'))

        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_BASE_URL'), timeout=240, http_client=http_client)
        self.model = os.getenv('MODEL_FOR_EVAL')

        # use easyocr for local OCR
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.easyocr_langs = os.getenv('EASYOCR_LANGS', 'en').split(',')
        try:
            self.easyocr_reader = easyocr.Reader(self.easyocr_langs, gpu=(self.device == 'cuda'))
        except Exception as e:
            print(f"Warning: easyocr reader failed to initialize: {e}")
            self.easyocr_reader = None

        model_name = 'facebook/dinov2-base'
        self.dino_processor = AutoImageProcessor.from_pretrained(model_name)
        torch_dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.dino_model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=None, low_cpu_mem_usage=False, trust_remote_code=True)
        if hasattr(self.dino_model, 'device') and str(self.dino_model.device) == 'meta':
            self.dino_model = self.dino_model.to_empty(device=self.device)
        else:
            self.dino_model = self.dino_model.to(self.device)
        self.dino_model.eval()

    def _extract_text_from_image(self, img_path: str) -> str:
        # Use easyocr to extract text locally. If the reader failed to init, return empty string.
        if not self.easyocr_reader:
            print("easyocr reader not available")
            return ""
        try:
            # detail=0 returns only text strings
            results = self.easyocr_reader.readtext(img_path, detail=0)
            if not results:
                return ""
            # join segments into one string
            return " ".join([r.strip() for r in results if r and r.strip()])
        except Exception as e:
            print(f"OCR (easyocr) error: {e}")
            return ""

    # remote OCR endpoint removed; easyocr is used locally via _extract_text_from_image

    def _calculate_edit_distance_similarity(self, text1: str, text2: str) -> float:
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        distance = Levenshtein.distance(text1_lower, text2_lower)
        max_len = max(len(text1_lower), len(text2_lower))
        similarity = 1.0 - (distance / max_len) if max_len > 0 else 1.0
        return similarity

    def score_ocr_similarity(self, img1_path: str, img2_path: str) -> tuple[float, dict]:
        text1 = self._extract_text_from_image(img1_path)
        text2 = self._extract_text_from_image(img2_path)
        similarity = self._calculate_edit_distance_similarity(text1, text2)
        return similarity, {"checkpoint_text": text1, "screenshot_text": text2, "similarity": similarity}

    def score_dino(self, img1_path: str, img2_path: str) -> float:
        def get_embedding(image_path):
            image = Image.open(image_path).convert('RGB')
            inputs = self.dino_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.dino_model(**inputs)
                return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

        emb1 = get_embedding(img1_path)
        emb2 = get_embedding(img2_path)
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return max(0.0, min(1.0, float(similarity)))

    def score_dino_batch(self, img_pairs: list) -> list:
        if not img_pairs:
            return []
        def get_embedding_batch(image_paths):
            images = [Image.open(path).convert('RGB') for path in image_paths]
            inputs = self.dino_processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.dino_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                return embeddings

        all_img1_paths = [pair[0] for pair in img_pairs]
        all_img2_paths = [pair[1] for pair in img_pairs]
        emb1_batch = get_embedding_batch(all_img1_paths)
        emb2_batch = get_embedding_batch(all_img2_paths)
        similarities = []
        for i in range(len(img_pairs)):
            emb1 = emb1_batch[i]
            emb2 = emb2_batch[i]
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similarities.append(max(0.0, min(1.0, float(similarity))))
        return similarities

    def score_similarity(self, img1_path: str, img2_path: str) -> tuple[int, str]:
        content_list = [{"type": "text", "text": COMPARE_TEMPLATE}]
        for p in [img1_path, img2_path]:
            if os.path.exists(p):
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{self._get_media_type(p)};base64,{self._encode_file(p)}"}
                })

        try:
            response_text = call_mllm(self.model, content_list)
            score = self._parse_score(response_text)
            return max(0, min(100, score)), response_text
        except Exception as e:
            raise Exception(f"MLLM scoring via call_mllm failed: {e}")

    def _encode_file(self, file_path: str) -> str:
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _get_media_type(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        types = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}
        return types.get(ext, 'application/octet-stream')

    def _load_image_as_base64(self, img_path: str) -> str:
        if not os.path.exists(img_path):
            return ""
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def _parse_score(self, response_text: str) -> int:
        match = re.search(r"Score:.*?(\d{1,3})/100", response_text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 0

    def calculate_all_scores(self, data_path: Path, screenshot_paths: list) -> tuple[list, list, list, list, list]:
        checkpoint_paths = []
        for i, screenshot_path in enumerate(screenshot_paths):
            print(data_path, screenshot_path)
            checkpoint_path = Path(data_path) / "checkpoints" / f"step_{i}.png"
            if not checkpoint_path.exists():
                raise Exception(f"Checkpoint image missing: {checkpoint_path}")
            checkpoint_paths.append(checkpoint_path)

        def calculate_all_dino_scores():
            img_pairs = [(str(checkpoint_paths[i]), str(screenshot_paths[i])) for i in range(len(screenshot_paths))]
            dino_scores = self.score_dino_batch(img_pairs)
            for i, score in enumerate(dino_scores):
                # DINO score is in [0,1]; present as percentage
                print(f"  Step {i} DINO score: {score * 100:.2f}%")
            return dino_scores

        print("Starting concurrent scoring...")
        with ThreadPoolExecutor(max_workers=15) as executor:
            dino_future = executor.submit(calculate_all_dino_scores)
            mllm_futures = {executor.submit(self.score_similarity, str(checkpoint_paths[i]), str(screenshot_paths[i])): i for i in range(len(screenshot_paths))}
            ocr_futures = {executor.submit(self.score_ocr_similarity, str(checkpoint_paths[i]), str(screenshot_paths[i])): i for i in range(len(screenshot_paths))}

            mllm_results = {}
            for future in as_completed(mllm_futures):
                index = mllm_futures[future]
                mllm_score, mllm_response = future.result()
                mllm_results[index] = (mllm_score, mllm_response)


            ocr_results = {}
            for future in as_completed(ocr_futures):
                index = ocr_futures[future]
                try:
                    ocr_score, ocr_texts = future.result()
                    ocr_results[index] = (ocr_score, ocr_texts)
                except Exception as e:
                    print(f"  Step {index} OCR failed: {e}")
                    ocr_results[index] = (0.0, {"checkpoint_text": "", "screenshot_text": "", "similarity": 0.0})

            dino_scores = dino_future.result()

        mllm_scores, mllm_responses, ocr_scores, ocr_texts_list = [], [], [], []
        for i in range(len(screenshot_paths)):
            mllm_score, mllm_response = mllm_results[i]
            ocr_score, ocr_texts = ocr_results[i]
            mllm_scores.append(mllm_score)
            mllm_responses.append(mllm_response)
            ocr_scores.append(ocr_score)
            ocr_texts_list.append(ocr_texts)
            # DINO and OCR are in [0,1] and should be displayed as percentages; MLLM is 0-100
            print(f"  Step {i} Final: MLLM={mllm_score}, DINO={dino_scores[i] * 100:.2f}%, OCR={ocr_score * 100:.2f}%")

        return mllm_scores, dino_scores, ocr_scores, mllm_responses, ocr_texts_list
