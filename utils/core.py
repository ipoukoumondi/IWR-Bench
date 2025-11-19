import os
import cv2
import base64
import shutil
import json
import re
from typing import List, Tuple
from config import API_CONFIG, PROMPT_TEMPLATE
from openai import OpenAI
import httpx


def load_examples(jsonl_path: str):
	entries = []
	with open(jsonl_path, 'r', encoding='utf-8') as f:
		for line in f:
			if not line.strip():
				continue
			try:
				j = json.loads(line)
				entries.append(j)
			except Exception:
				continue
	return entries

def get_config(model_name: str):
    # Return best-matching config if present, else default
    for key in API_CONFIG:
        if key in model_name.lower():
            return API_CONFIG[key]
    return API_CONFIG["default"]


def encode_file(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def save_base64_image(img_b64: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(base64.b64decode(img_b64))

def get_media_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    types = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
             '.mp4': 'video/mp4', '.mov': 'video/quicktime', '.webp': 'image/webp', '.svg': 'image/svg+xml'}
    return types.get(ext, 'application/octet-stream')


def resize_image(img_path: str, max_size: int):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: cannot read image {img_path}")
        return

    h, w = img.shape[:2]
    if w > h:
        ratio = max_size / w
        new_w = max_size
        new_h = int(h * ratio)
    else:
        ratio = max_size / h
        new_h = max_size
        new_w = int(w * ratio)

    if ratio < 1.0:
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(img_path, resized)


def extract_frames(video_path: str, output_folder: str, config_dict: dict) -> List[str]:
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        return []

    video_duration = total_frames / fps
    video_fps = config_dict.get('video_fps', 1)
    max_video_frames = config_dict.get('max_video_frames', 64)

    if video_duration > max_video_frames:
        frame_interval = total_frames / max_video_frames
        target_frame_count = max_video_frames
    else:
        frame_interval = fps / video_fps
        target_frame_count = int(video_duration * video_fps)

    frame_paths = []
    for i in range(target_frame_count):
        frame_index = int(i * frame_interval)
        if frame_index >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        if success:
            frame_path = os.path.join(output_folder, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)

    cap.release()
    return frame_paths


def find_files(project_path: str) -> Tuple[str, List[str]]:
    video_path = os.path.join(project_path, 'video.mp4')
    image_paths = []
    stitched_dir = os.path.join(project_path, 'stitched_assets')
    if os.path.exists(stitched_dir):
        for fname in os.listdir(stitched_dir):
            if fname.startswith('stitched_image'):
                image_paths.append(os.path.join(stitched_dir, fname))
    return video_path, sorted(image_paths)


def get_resolution(project_path: str) -> str:
    metadata_path = os.path.join(project_path, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            return meta.get('recording_details', {}).get('resolution', '1920x1080')
    return '1920x1080'


def prepare_images_in_temp_folder(image_paths: List[str], video_path: str, temp_folder: str, config_dict: dict) -> List[str]:
    os.makedirs(temp_folder, exist_ok=True)
    all_image_paths = []
    for i, img_path in enumerate(image_paths):
        filename = f"asset_{i}_{os.path.basename(img_path)}"
        temp_img_path = os.path.join(temp_folder, filename)
        shutil.copy2(img_path, temp_img_path)
        all_image_paths.append(temp_img_path)

    images_used = len(all_image_paths)
    max_frames = config_dict.get('max_images', 70) - images_used
    if max_frames > 0 and os.path.exists(video_path):
        temp_config = config_dict.copy()
        temp_config['max_video_frames'] = min(temp_config.get('max_video_frames', 64), max_frames)
        frame_paths = extract_frames(video_path, temp_folder, temp_config)
        all_image_paths.extend(frame_paths)

    if 'max_single_image_size' in config_dict:
        for p in all_image_paths:
            resize_image(p, config_dict['max_single_image_size'])

    return all_image_paths


def post_process(raw_response: str, assets_path: str) -> str:
    if not raw_response:
        return ''
    code = raw_response.strip()
    html_match = re.search(r'```html\s*\n?(.*?)\n?```', code, re.DOTALL | re.IGNORECASE)
    if html_match:
        code = html_match.group(1).strip()
    code = re.sub(r'<iframe[^>]*>.*?</iframe>', '', code, flags=re.DOTALL | re.IGNORECASE)
    code = re.sub(r'<iframe[^>]*/?>', '', code, flags=re.IGNORECASE)
    code = code.replace('```html', '')
    code = code.replace('```', '')
    code = code.replace('__PLACEHOLDER_ASSETS_BASE_DIR__', assets_path)
    return code


def mock_model_call(prompt: str, image_paths: List[str], resolution: str) -> str:
    img_tag = ''
    if image_paths:
        fname = os.path.basename(image_paths[0])
        img_tag = f'<img src="__PLACEHOLDER_ASSETS_BASE_DIR__/{fname}" width="800" height="600" alt="stitched" />'

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Generated Mock</title>
  <style>body{{font-family:Arial,Helvetica,sans-serif;padding:16px}}.preview{{max-width:1000px}}</style>
</head>
<body>
  <h1>Mock Generated Page ({resolution})</h1>
  <div class="preview">{img_tag}</div>
  <p>This is a mock HTML output produced by IWREval utils.mock_model_call. Replace with a real model call for production.</p>
</body>
</html>"""
    return html


def call_mllm(model_name: str, content_list: List[dict], timeout: int = 300, max_tokens: int = 16384, extra_params: dict | None = None, http_client=None) -> str:
    api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('OPENAI_BASE_URL', None)
    if not api_key:
        msg = 'OPENAI_API_KEY is not set. call_mllm requires a configured API key.'
        print(msg)
        raise RuntimeError(msg)

    client = None
    if http_client is None:
        http_client = None
        try:
            proxy = os.getenv('HTTP_PROXY') or os.getenv('http_proxy')
            if proxy:
                http_client = httpx.Client(proxies=proxy, transport=httpx.HTTPTransport(local_address='0.0.0.0'))
        except Exception as e:
            print(f"Warning: failed to configure httpx proxy client: {e}")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, http_client=http_client)

    api_params = {
        "model": model_name,
        "messages": [{"role": "user", "content": content_list}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if extra_params:
        api_params.update(extra_params)

    try:
        print(f"Calling model {model_name}, frames content count: {len(content_list)}")
        response = client.chat.completions.create(**api_params)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Calling model API call failed: {e}")
        raise


def call_mllm_html(model_name: str, image_paths: List[str], video_path: str, temp_folder: str, resolution: str, prompt: str = None) -> str:
    config_dict = get_config(model_name)
    temp_image_paths = prepare_images_in_temp_folder(image_paths, video_path, temp_folder, config_dict)
    # allow callers to override the default prompt template
    if prompt is None:
        prompt = PROMPT_TEMPLATE.format(resolution=resolution)

    content_list = [{"type": "text", "text": prompt}]
    for img_path in temp_image_paths:
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:{get_media_type(img_path)};base64,{encode_file(img_path)}"}
        })

    return call_mllm(model_name, content_list, timeout=config_dict.get('timeout', 300), max_tokens=config_dict.get('max_tokens', 16384), extra_params=config_dict.get('extra_params', None))


__all__ = [
    'get_config', 'encode_file', 'get_media_type', 'resize_image', 'extract_frames',
    'find_files', 'get_resolution', 'prepare_images_in_temp_folder', 'post_process', 'mock_model_call', 'call_mllm', 'call_mllm_html'
]
