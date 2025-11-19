import asyncio
import json
import os
import re
import base64
import random
import string
from pathlib import Path
from typing import Any

from browser_use.browser import BrowserProfile, BrowserSession
from browser_use.config import load_browser_use_config
from browser_use.controller.service import Controller
from browser_use.filesystem.file_system import FileSystem

from dotenv import load_dotenv
load_dotenv()


class BrowserUse:
    def __init__(self, window_size: dict[str, int] = {'width': 1920, 'height': 1080}, headless: bool = False):
        self.config = load_browser_use_config()
        self.browser_session: BrowserSession | None = None
        self.controller: Controller | None = None
        self.file_system: FileSystem | None = None
        self.window_size = window_size
        self.headless = headless

    async def _init_browser_session(self, **kwargs):
        if self.browser_session:
            return

        headless = os.getenv("HEADLESS", "true").lower() == "true"
        
        random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=10))

        profile_data = {
            'wait_between_actions': 0.5,
            'keep_alive': True,
            'is_mobile': False,
            'device_scale_factor': 1.0,
            'disable_security': False,
            'headless': headless,
            "default": True,
            "allowed_domains": None,
            "window_size": self.window_size,
            'user_data_dir': '/tmp/test_' + random_str
        }

        for key, value in kwargs.items():
            profile_data[key] = value

        profile = BrowserProfile(**profile_data)

        self.browser_session = BrowserSession(browser_profile=profile)
        await self.browser_session.start()

        self.controller = Controller()

        file_system_path = profile_data.get('file_system_path', '/tmp/extract_' + random_str)
        self.file_system = FileSystem(base_dir=Path(file_system_path).expanduser())

    async def get_axtree(self):
        page = await self.browser_session.get_current_page()
        cdp_session = await page.context.new_cdp_session(page)

        await cdp_session.send('Accessibility.enable')
        await cdp_session.send('DOM.enable')

        ax_tree = await cdp_session.send('Accessibility.getFullAXTree')
        return flatten_axtree_to_str(ax_tree)

    async def get_browser_state(self) -> str:
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        state = await self.browser_session.get_state_summary(cache_clickable_elements_hashes=False)
        result = {
            'url': state.url,
            'title': state.title,
            'tabs': [{'url': tab.url, 'title': tab.title} for tab in state.tabs],
            'interactive_elements': [],
        }

        for index, element in state.selector_map.items():
            raw_str = element.clickable_elements_to_string().replace('\t', '').replace('\n[', '[')
            all_indices = [int(x) for x in re.findall(r'\[(\d+)\]', raw_str)]
            sub_element_indices = all_indices[1:] if len(all_indices) > 1 else []

            main_tag_match = re.search(r'\[\d+\]<(.*?)\/>', raw_str)
            main_tag_text = '<' + main_tag_match.group(1).strip() + '/>' if main_tag_match else ""

            elem_info = {
                'index': index,
                'tag': element.tag_name,
                'text': main_tag_text,
            }
            if element.attributes.get('placeholder'):
                elem_info['placeholder'] = element.attributes['placeholder']
            if element.attributes.get('href'):
                elem_info['href'] = element.attributes['href']
            result['interactive_elements'].append(elem_info)
        return json.dumps(result["interactive_elements"])
    
    
    async def get_clean_screenshot(self, need_highlight: bool = False, save_path: str | None = None) -> str:
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'
        
        if need_highlight:
            await self.browser_session.remove_highlights()
        
        await asyncio.sleep(0.2)
        
        page = await self.browser_session.get_current_page()
        screenshot = await page.screenshot(type='png')
        
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(screenshot)
            return save_path
        else:
            base_64_data = base64.b64encode(screenshot).decode('utf-8')
            return base_64_data

    async def go_to_url(self, url: str, new_tab: bool):
        if not self.browser_session:
            await self._init_browser_session()

        action = self.controller.registry.create_action_model()(go_to_url={"url": url, "new_tab": new_tab})
        action_result = await self.controller.act(
            action=action,
            browser_session=self.browser_session,
            file_system=self.file_system,
        )

        await self.browser_session.remove_highlights()
        if action_result.error is None:
            return action_result.extracted_content
        else:
            return "ERROR: " + action_result.error

    async def wait(self, seconds: int):
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        action = self.controller.registry.create_action_model()(wait={"seconds": seconds})
        action_result = await self.controller.act(
            action=action,
            browser_session=self.browser_session,
            file_system=self.file_system,
        )

        await self.browser_session.remove_highlights()
        if action_result.error is None:
            return action_result.extracted_content
        else:
            return "ERROR: " + action_result.error

    async def click_element_by_index(self, index: int):
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        action = self.controller.registry.create_action_model()(click_element_by_index={"index": index})
        action_result = await self.controller.act(
            action=action,
            browser_session=self.browser_session,
            file_system=self.file_system,
        )

        await self.browser_session.remove_highlights()
        if action_result.error is None:
            return action_result.extracted_content
        else:
            return "ERROR: " + action_result.error

    async def input_text(self, index: int, text: str):
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        action = self.controller.registry.create_action_model()(input_text={"index": index, "text": text})
        action_result = await self.controller.act(
            action=action,
            browser_session=self.browser_session,
            file_system=self.file_system,
        )

        await self.browser_session.remove_highlights()
        if action_result.error is None:
            return action_result.extracted_content
        else:
            return "ERROR: " + action_result.error

    async def send_keys(self, keys: str):
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        action = self.controller.registry.create_action_model()(send_keys={"keys": keys})
        action_result = await self.controller.act(
            action=action,
            browser_session=self.browser_session,
            file_system=self.file_system,
        )

        await self.browser_session.remove_highlights()
        if action_result.error is None:
            return action_result.extracted_content
        else:
            return "ERROR: " + action_result.error

    async def upload_file(self, index: int, path: str):
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        action = self.controller.registry.create_action_model()(upload_file={"index": index, "path": path})
        action_result = await self.controller.act(
            action=action,
            browser_session=self.browser_session,
            file_system=self.file_system,
        )

        await self.browser_session.remove_highlights()
        if action_result.error is None:
            return action_result.extracted_content
        else:
            return "ERROR: " + action_result.error

    async def go_back(self):
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        action = self.controller.registry.create_action_model()(go_back={})
        action_result = await self.controller.act(
            action=action,
            browser_session=self.browser_session,
            file_system=self.file_system,
        )

        await self.browser_session.remove_highlights()
        if action_result.error is None:
            return action_result.extracted_content
        else:
            return "ERROR: " + action_result.error

    async def scroll(self, down: bool, num_pages: float, index: int):
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        action = self.controller.registry.create_action_model()(scroll={"down": down, "num_pages": num_pages, "index": index})
        action_result = await self.controller.act(
            action=action,
            browser_session=self.browser_session,
            file_system=self.file_system,
        )

        await self.browser_session.remove_highlights()
        if action_result.error is None:
            return action_result.extracted_content
        else:
            return "ERROR: " + action_result.error

    async def list_tabs(self):
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        state = await self.browser_session.get_state_summary(cache_clickable_elements_hashes=False)
        tabs = [{'index': tab.page_id, 'url': tab.url, 'title': tab.title} for tab in state.tabs]

        await self.browser_session.remove_highlights()
        return tabs

    async def switch_tab(self, tab_index: int,):
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        action = self.controller.registry.create_action_model()(switch_tab={"page_id": tab_index})
        action_result = await self.controller.act(
            action=action,
            browser_session=self.browser_session,
            file_system=self.file_system,
        )

        await self.browser_session.remove_highlights()
        if action_result.error is None:
            return action_result.extracted_content
        else:
            return "ERROR: " + action_result.error

    async def close_tab(self, tab_index: int):
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        action = self.controller.registry.create_action_model()(close_tab={"page_id": tab_index})
        action_result = await self.controller.act(
            action=action,
            browser_session=self.browser_session,
            file_system=self.file_system,
        )

        await self.browser_session.remove_highlights()
        if action_result.error is None:
            return action_result.extracted_content
        else:
            return "ERROR: " + action_result.error

    async def get_dropdown_options(self, index: int):
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        action = self.controller.registry.create_action_model()(get_dropdown_options={"index": index})
        action_result = await self.controller.act(
            action=action,
            browser_session=self.browser_session,
            file_system=self.file_system,
        )

        await self.browser_session.remove_highlights()
        if action_result.error is None:
            return action_result.extracted_content
        else:
            return "ERROR: " + action_result.error

    async def select_dropdown_option(self, index: int, text: str):
        if not self.browser_session:
            return 'Error: No browser session active, please use go to a url'

        action = self.controller.registry.create_action_model()(select_dropdown_option={"index": index, "text": text})
        action_result = await self.controller.act(
            action=action,
            browser_session=self.browser_session,
            file_system=self.file_system,
        )

        await self.browser_session.remove_highlights()
        if action_result.error is None:
            return action_result.extracted_content
        else:
            return "ERROR: " + action_result.error


def flatten_axtree_to_str(axtree, ignored_roles=None, depth=0, node_idx=0):
    if ignored_roles is None:
        ignored_roles = {"none"}

    nodes = axtree["nodes"]
    node = nodes[node_idx]

    role = node.get("role", {}).get("value", "")
    name = node.get("name", {}).get("value", "")

    if node.get("ignored", False) or role in ignored_roles:
        children_str = ""
        for child_id in node.get("childIds", []):
            child_idx = next((i for i, n in enumerate(nodes) if n["nodeId"] == child_id), None)
            if child_idx is not None:
                children_str += flatten_axtree_to_str(axtree, ignored_roles, depth, child_idx)
        return children_str

    indent = "    " * depth

    props = []
    for prop in node.get("properties", []):
        p_name = prop.get("name", "")
        p_val_dict = prop.get("value")
        if p_val_dict is not None:
            if isinstance(p_val_dict, dict) and "value" in p_val_dict:
                p_value = p_val_dict["value"]
            else:
                p_value = p_val_dict
        else:
            p_value = None
        props.append(f"{p_name}={repr(p_value)}")
    prop_str = (", " + ", ".join(props)) if props else ""

    s = f"{indent}{role} {repr(name)}{prop_str}\n"

    for child_id in node.get("childIds", []):
        child_idx = next((i for i, n in enumerate(nodes) if n["nodeId"] == child_id), None)
        if child_idx is not None:
            s += flatten_axtree_to_str(axtree, ignored_roles, depth+1, child_idx)
    return s