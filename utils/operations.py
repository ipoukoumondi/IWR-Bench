import asyncio
import os
from typing import Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
import httpx

from .browser import BrowserUse

load_dotenv()


class IWR:
    def __init__(self, browser: BrowserUse):
        self.browser = browser
        proxy_url = os.getenv('IWR_PROXY', None)
        http_client = None
        if proxy_url:
            http_client = httpx.Client(proxies=proxy_url, transport=httpx.HTTPTransport(local_address='0.0.0.0'))

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            timeout=180,
            http_client=http_client
        )
        self.model = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")

    async def _get_llm_response(self, prompt: str) -> Optional[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content

    async def _get_vllm_response(self, query: str) -> Optional[str]:
        state = await self.browser.browser_session.get_state_summary(cache_clickable_elements_hashes=False)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state.screenshot}"}}
                ]}
            ]
        )
        await self.browser.browser_session.remove_highlights()
        return response.choices[0].message.content

    async def _get_page_info(self) -> Tuple[str, str]:
        axtree = await self.browser.get_axtree()
        state = await self.browser.get_browser_state()
        return axtree, state

    async def _try_llm_click(self, action_desc: str, axtree: str, state: str) -> Optional[int]:
        prompt = f"""You need to determine the click index of the action.
The browser Accessibility Tree: {axtree}
The browser State: {state}
The action description is: {action_desc}
If you cannot find the click index, return -1.
Only return the click index, no other text."""

        response = await self._get_llm_response(prompt)
        print(f"get index from LLM response: {response}")
        if response and response.strip().isdigit():
            idx = int(response.strip())
            return idx if idx >= 0 else None
        return None

    async def _try_vision_click(self, action_desc: str) -> Optional[int]:
        prompt = f"""You need to determine the click index of the action by the screenshot.
The action description is: {action_desc}
If you cannot find the click index, return -1.
Only return the click index, no other text."""

        result = await self._get_vllm_response(prompt)
        if result and result.strip().isdigit():
            idx = int(result.strip())
            return idx if idx >= 0 else None

        return None

    async def _execute_click(self, idx: int) -> bool:
        result = await self.browser.click_element_by_index(idx)
        return not (isinstance(result, str) and ("does not exist" in result or "no container found" in result))

    async def click(self, action_desc: str) -> bool:
        axtree, state = await self._get_page_info()
        idx = await self._try_llm_click(action_desc, axtree, state)
        if idx is not None:
            return await self._execute_click(idx)
        idx = await self._try_vision_click(action_desc)
        if idx is not None:
            return await self._execute_click(idx)
        return False

    async def _try_llm_type(self, description: str, axtree: str, state: str) -> Optional[Tuple[int, str]]:
        prompt = f"""You need to determine the element index and the text to input.
The browser Accessibility Tree: {axtree}
The browser State: {state}
The action description is: {description}
Format your response as: element_index,"text_to_type"
If you cannot find the element index, return -1,""
Only return the index and text in the specified format, no other text."""

        response = await self._get_llm_response(prompt)
        if response and ',"' in response:
            idx = int(response.split(',"')[0])
            text = response.split(',"')[1].rstrip('"')
            return (idx, text) if idx >= 0 else None
        return None

    async def _try_vision_type(self, description: str) -> Optional[Tuple[int, str]]:
        prompt = f"""You need to determine the element index and text to input from the screenshot.
The action description is: {description}
Format your response as: element_index,"text_to_type"
If you cannot find the element index, return -1,""
Only return the index and text in the specified format, no other text."""

        result = await self._get_vllm_response(prompt)
        if result and ',"' in result:
            idx = int(result.split(',"')[0])
            text = result.split(',"')[1].rstrip('"')
            return (idx, text) if idx >= 0 else None
        return None

    async def _execute_type(self, idx: int, text: str) -> bool:
        result = await self.browser.input_text(idx, text)
        return not (isinstance(result, str) and ("does not exist" in result or "no container found" in result))

    async def type(self, params: dict) -> bool:
        text = params.get("text", "")
        element_index = params.get("element_index")
        description = params.get("description", "")

        if element_index is not None and text:
            return await self._execute_type(element_index, text)

        if not description:
            return False

        axtree, state = await self._get_page_info()
        result = await self._try_llm_type(description, axtree, state)
        if result:
            idx, text = result
            return await self._execute_type(idx, text)

        print(f"llm cann't findtry vision type: {description}")
        result = await self._try_vision_type(description)
        if result:
            idx, text = result
            return await self._execute_type(idx, text)

        return False

    async def _parse_key_from_description(self, description: str) -> Optional[str]:
        prompt = f"""You are a helpful assistant.
You are given a description of a press key action.
You need to determine which key should be pressed.
Common keys include: Enter, Escape, Tab, Backspace, ArrowUp, ArrowDown, ArrowLeft, ArrowRight, etc.

The action description is: {description}
Only return the key name, no other text.
If you cannot determine the key, return "UNKNOWN" """

        response = await self._get_llm_response(prompt)
        if response and response.strip() != "UNKNOWN":
            return response.strip()
        return None

    async def _execute_press(self, key: str) -> bool:
        await self.browser.send_keys(key)
        return True

    async def press(self, params: dict) -> bool:
        key = params.get("key", "")
        description = params.get("description", "")

        if key:
            return await self._execute_press(key)

        if not description:
            return False

        key = await self._parse_key_from_description(description)
        if key:
            return await self._execute_press(key)

        return False

    async def scroll(self, action_desc: str) -> bool:
        page = await self.browser.browser_session.get_current_page()
        initial_info = await self.browser.browser_session.get_page_info(page)

        result = await self.browser.scroll(down=True, num_pages=0.5, index=0)
        if not result:
            return False

        await asyncio.sleep(1)

        final_info = await self.browser.browser_session.get_page_info(page)
        scroll_changed = abs(final_info.scroll_y - initial_info.scroll_y) > 5

        print(f"scroll_changed: {scroll_changed}, initial_info.scroll_y: {initial_info.scroll_y}, final_info.scroll_y: {final_info.scroll_y}")
        return scroll_changed
