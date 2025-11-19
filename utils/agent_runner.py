import asyncio
import os
import shutil
from pathlib import Path
from typing import Tuple, List

from .browser import BrowserUse
from .operations import IWR


def save_base64_image(img_b64: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(__import__('base64').b64decode(img_b64))


async def run_task(task: dict, model: str, task_index: int, total_tasks: int) -> Tuple[str, int, int, List[Path], list]:
    print(f"[{task_index}/{total_tasks}] Starting task: {task['name']}")

    eval_steps_path = Path(task['result_path']) / model / "eval_steps"
    if eval_steps_path.exists():
        shutil.rmtree(eval_steps_path)

    width, height = str(task['resolution']).split('x')
    window_size = {"width": int(width), "height": int(height)}

    browser = BrowserUse(window_size=window_size, headless=True)
    iwr = IWR(browser)

    await asyncio.wait_for(browser.go_to_url(task['start_url'], new_tab=False), timeout=60.0)
    await asyncio.sleep(1)

    success_count = 0
    screenshot_paths = []

    # Step 0: initial screenshot
    screenshot_b64 = await asyncio.wait_for(browser.get_clean_screenshot(), timeout=30.0)
    screenshot_path = Path(task['result_path']) / model / "eval_steps" / "step_0.png"
    save_base64_image(screenshot_b64, screenshot_path)
    screenshot_paths.append(screenshot_path)

    for i, step in enumerate(task.get('action_sequence', []), 1):
        action_type = step.get('type', '')
        params = step.get('parameters', {})
        print(f"[{task_index}/{total_tasks}] Executing step {i}/{len(task.get('action_sequence', []))}: {action_type}")

        action_map = {
            "Click": lambda: iwr.click(params.get("description", "")),
            "Scroll": lambda: iwr.scroll(params.get("description", "")),
            "Type": lambda: iwr.type(params),
            "Press": lambda: iwr.press(params)
        }

        if action_type in action_map:
            try:
                success = await asyncio.wait_for(action_map[action_type](), timeout=300.0)
            except Exception as e:
                print(f"[{task_index}/{total_tasks}] Action {i} raised: {e}")
                success = False
        else:
            print(f"Warning: Unknown action type '{action_type}'")
            success = False

        if success:
            success_count += 1
            print(f"[{task_index}/{total_tasks}] Step {i} succeeded")
        else:
            print(f"[{task_index}/{total_tasks}] Step {i} failed, stopping execution...")
            break

        await asyncio.sleep(0.5)

        screenshot_b64 = await asyncio.wait_for(browser.get_clean_screenshot(), timeout=30.0)
        screenshot_path = Path(task['result_path']) / model / "eval_steps" / f"step_{i}.png"
        save_base64_image(screenshot_b64, screenshot_path)
        screenshot_paths.append(screenshot_path)

    total_steps = len(task.get('action_sequence', []))
    print(f"[{task_index}/{total_tasks}] {task['name']}: {success_count}/{total_steps} steps successful")

    if hasattr(browser, 'browser_session') and browser.browser_session:
        try:
            await asyncio.wait_for(browser.browser_session.kill(), timeout=10.0)
        except Exception:
            pass

    return task['name'], success_count, total_steps, screenshot_paths, task.get('action_sequence', [])


__all__ = ['run_task']
