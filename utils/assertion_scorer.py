import os
import base64
import httpx
import json
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any, List, Union

load_dotenv()

ASSERTION_TEMPLATE = """Please compare two webpage screenshots (Image 1 is the previous step, Image 2 is the current step) and determine whether the following assertion is true:

Assertion: {assertion}

Return JSON format without any additional information:
{{
    "think": "the simple thinking process within 100 words",
    "result": "Yes|No"
}}
"""


class AssertionScorer:
    def __init__(self):
        proxy = os.getenv('IWR_PROXY', None)
        http_client = None
        if proxy:
            http_client = httpx.Client(proxies=proxy, transport=httpx.HTTPTransport(local_address='0.0.0.0'))

        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url=os.getenv('OPENAI_BASE_URL'), timeout=240, http_client=http_client)
        self.model = os.getenv('MODEL_FOR_EVAL')

    def evaluate_assertion(self, prev_img_path: str, curr_img_path: str, assertion: str) -> Dict[str, Any]:
        prev_img_b64 = self._load_image_as_base64(prev_img_path)
        curr_img_b64 = self._load_image_as_base64(curr_img_path)
        prompt_text = ASSERTION_TEMPLATE.format(assertion=assertion)

        if not prev_img_b64 or not curr_img_b64:
            return {"result": "Error", "prev_img_path": prev_img_path, "curr_img_path": curr_img_path, "response": "Error: Could not load images", "assertion": assertion, "thinking": ""}

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{prev_img_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{curr_img_b64}"}}
            ]
        }]

        try:
            response = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0, max_tokens=4096, timeout=120)
            full_response = response.choices[0].message.content.strip()
            try:
                if "```json" in full_response:
                    json_start = full_response.find("```json") + 7
                    json_end = full_response.find("```", json_start)
                    json_text = full_response[json_start:json_end].strip()
                elif "```" in full_response:
                    json_start = full_response.find("```") + 3
                    json_end = full_response.find("```", json_start)
                    json_text = full_response[json_start:json_end].strip()
                else:
                    json_text = full_response
                parsed_response = json.loads(json_text)
                result = parsed_response.get("result", "Error")
                thinking = parsed_response.get("think", "")
                if result not in ["Yes", "No"]:
                    result = "Error"
            except Exception as e:
                print(f"JSON parse failed, fallback to text parse: {e}")
                if "Yes" in full_response:
                    result = "Yes"
                elif "No" in full_response:
                    result = "No"
                else:
                    result = "Error"
                thinking = ""

            return {"result": result, "prev_img_path": prev_img_path, "curr_img_path": curr_img_path, "response": full_response, "assertion": assertion, "thinking": thinking}

        except Exception as e:
            print(f"Assertion evaluation error: {e}")
            return {"result": "Error", "prev_img_path": prev_img_path, "curr_img_path": curr_img_path, "response": f"Error: {e}", "assertion": assertion, "thinking": ""}

    def _load_image_as_base64(self, img_path: str) -> str:
        if not os.path.exists(img_path):
            return ""
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def evaluate_all_assertions(self, action_sequence: list, screenshot_paths: list) -> List[Union[str, None]]:
        assertion_results = []
        self.assertion_details = []
        for i, action in enumerate(action_sequence):
            step = i+1
            assertion = action.get("logical_assertion")
            if assertion and step > 0 and step < len(screenshot_paths):
                prev_img_path = screenshot_paths[step-1]
                curr_img_path = screenshot_paths[step]
                detail_result = self.evaluate_assertion(str(prev_img_path), str(curr_img_path), assertion)
                result = detail_result
                assertion_results.append(result)
                self.assertion_details.append(detail_result)
                print(f"IWR: Assertion step {step}: '{assertion}' -> {result['result']}, thinking: {detail_result['thinking']}")
            else:
                assertion_results.append(None)
                self.assertion_details.append(None)
        return assertion_results

    def get_assertion_details(self) -> List[Union[Dict[str, Any], None]]:
        return getattr(self, 'assertion_details', [])
