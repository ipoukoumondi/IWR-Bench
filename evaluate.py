import asyncio
import os
import argparse
import json
from pathlib import Path
from utils import core as utils
from utils import metrics as metrics_utils
from utils.agent_runner import run_task as agent_run_task
from utils.browser import BrowserUse
from utils.operations import IWR
from utils.visual_scorer import VisualScorer
from utils.assertion_scorer import AssertionScorer

visual_scorer = VisualScorer()

async def excute_action(iwr: IWR, action_type: str, params: dict) -> bool:
    action_map = {
        "Click": lambda: iwr.click(params.get("description", "")),
        "Scroll": lambda: iwr.scroll(params.get("description", "")),
        "Type": lambda: iwr.type(params),
        "Press": lambda: iwr.press(params)
    }
    action_func = action_map.get(action_type)
    if action_func:
        try:
            return await asyncio.wait_for(action_func(), timeout=30.0)
        except Exception as e:
            print(f"Action {action_type} failed with error: {e}")
            return False
    else:
        print(f"Unknown action type: {action_type}")
        return False

    #return await asyncio.wait_for(action_map[action_type](), timeout=300.0)

async def excute_single_task(task: dict, gt_data_dir: str, result_dir: str, model_name: str) -> tuple[str, int]:
    task_id = str(task.get('id'))
    
    result_html = os.path.join(result_dir, task_id, model_name.lower().replace('/', '__'), 'index.html')
    result_html = os.path.abspath(result_html)
    if not os.path.exists(result_html):
        print("Result file not found for task:", task_id, "at", result_html)
        return None,None
    
    print("Evaluating task:", task_id, "using result from", result_html)
    
    width, height = str(task['recording_details']['resolution']).split('x')
    window_size = {"width": int(width), "height": int(height)}
    browser = BrowserUse(window_size=window_size, headless=True)
    iwr = IWR(browser)
    
    await asyncio.wait_for(browser.go_to_url("file://" + result_html, new_tab=False), timeout=60.0)
    await asyncio.sleep(1)
    screenshot_b64 = await asyncio.wait_for(browser.get_clean_screenshot(), timeout=30.0)
    utils.save_base64_image(screenshot_b64, os.path.join(result_dir, task_id, model_name.lower().replace('/', '__'), "evalpoints", f'0.png'))
    
    idx = 1
    for i,step in enumerate(task.get('action_sequence', []), 1):
        is_success = await excute_action(iwr, step.get('type', ''), step.get('parameters', {}))
        if not is_success:
            print(f"Action {i} failed, stopping execution...")
            break
        else:
            print(f"Action {i} succeeded")
        await asyncio.sleep(0.5)
        screenshot_b64 = await asyncio.wait_for(browser.get_clean_screenshot(), timeout=30.0)
        utils.save_base64_image(screenshot_b64, os.path.join(result_dir, task_id, model_name.lower().replace('/', '__'), "evalpoints", f'{idx}.png'))
        idx += 1

    if hasattr(browser, 'browser_session') and browser.browser_session:
        try:
            await asyncio.wait_for(browser.browser_session.kill(), timeout=10.0)
        except Exception:
            pass
        
    success_count = idx-1
    return task_id, success_count

async def evaluate_single_task(task: dict, gt_data_dir: str, result_dir: str, model_name: str, success_step_count: int) -> tuple[str, str]:
    """Perform the 4-step evaluation for a single task and save eval.json.

    Steps:
    1) Assertion scoring using AssertionScorer.evaluate_all_assertions
    2) OCR + DINO visual scoring using VisualScorer.calculate_all_scores
    3) MLLM scoring results come from VisualScorer.calculate_all_scores (all_mllm_res)
    4) Aggregate scores, save eval.json to model result dir and return (task_id, eval_path)
    """
    task_id = str(task.get('id'))
    model_dir = Path(result_dir) / task_id / model_name.lower().replace('/', '__')
    eval_json_path = model_dir / 'eval.json'
    screenshots_dir = model_dir / 'evalpoints'

    screenshot_paths = sorted(
        [str(p) for p in screenshots_dir.glob('*.png')],
        key=lambda p: int(Path(p).stem) if Path(p).stem.isdigit() else Path(p).stem
    )

    data_task_dir = Path(gt_data_dir) / task_id
    action_sequence = task.get('action_sequence', [])

    # 1) Assertion scoring
    assertion_scorer = AssertionScorer()
    assertion_results = assertion_scorer.evaluate_all_assertions(action_sequence, screenshot_paths)
    # 2 & 3) Visual scores (OCR + DINO) and MLLM responses
    screenshot_paths_to_eval = []
    screenshot_map = {int(Path(p).stem) if Path(p).stem.isdigit() else p: p for p in screenshot_paths}
    screenshot_paths_to_eval.append(screenshot_map[0])

    for i, action in enumerate(action_sequence):
        ar = assertion_results[i]
        if ar and ar["result"] == 'No':
            success_step_count = i
            break
        include_flag = action.get('visual_evaluation_flag', False)
        if include_flag and i in screenshot_map:
            screenshot_paths_to_eval.append(screenshot_map[i])

    mllm_scores, dino_scores, ocr_scores, mllm_responses, ocr_texts = visual_scorer.calculate_all_scores(
        str(data_task_dir), screenshot_paths_to_eval
    )

    # 4) Aggregate and save
    normalized_mllm = [s / 100 for s in mllm_scores] if mllm_scores else []
    avg_mllm = sum(normalized_mllm) / len(normalized_mllm) if normalized_mllm else 0
    avg_dino = sum(dino_scores) / len(dino_scores) if dino_scores else 0
    avg_ocr = sum(ocr_scores) / len(ocr_scores) if ocr_scores else 0
    all_avg = [(m + d + o) / 3 for m, d, o in zip(normalized_mllm, dino_scores, ocr_scores)] if all([normalized_mllm, dino_scores, ocr_scores]) else []
    avg_score = sum(all_avg) / len(all_avg) if all_avg else 0

    results = {
        "total_step": len(action_sequence),
        "success_step": success_step_count,
        "screenshot_paths_to_eval": screenshot_paths_to_eval,
        "all_mllm_scores": normalized_mllm,
        "all_dino_scores": dino_scores,
        "all_ocr_scores": ocr_scores,
        "all_avg_scores": all_avg,
        "mllm_score": avg_mllm,
        "dino_score": avg_dino,
        "ocr_score": avg_ocr,
        "avg_score": avg_score,
        "all_mllm_res": mllm_responses,
        "all_ocr_texts": ocr_texts,
        "assertion_results": assertion_results
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    eval_json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"IWRBenchEval: Saved evaluation results to {eval_json_path}")
    return task_id, str(eval_json_path)



async def main():
    parser = argparse.ArgumentParser(description='IWR Evaluation Framework')
    parser.add_argument('--data', required=True, help='Path to dataset jsonl file (e.g. example.jsonl)')
    parser.add_argument('--result', required=True, help='Result path')
    parser.add_argument('--model', default='gpt-4o', help='Model name')
    parser.add_argument('--task', help='Specific task filter')
    parser.add_argument('--coverage', action='store_true', help='Re-evaluate existing results')
    args = parser.parse_args()
        
    jsonl_path = args.data
    jsonl_dir = os.path.dirname(jsonl_path)
    jsonl_base = os.path.splitext(os.path.basename(jsonl_path))[0]
    gt_data_dir = os.path.join(jsonl_dir, jsonl_base)
    
    result_dir = args.result
    model_name = args.model
    
    tasks = utils.load_examples(jsonl_path)
    if not tasks:
        print('No tasks found')
        return

    result_dir = args.result
    
    report_accumulator = {}

    for task in tasks:
        task_id, success_count = await excute_single_task(task, gt_data_dir, result_dir, model_name)
        if task_id is None:
            continue
        print(f"IWRBenchEval: Task {task_id} completed with {success_count} excute successful actions.")
        task_id, eval_path = await evaluate_single_task(task, gt_data_dir, result_dir, model_name, success_count)
        print(f"IWRBenchEval: Task {task_id} evaluation saved to {eval_path}")
        eval_data = metrics_utils.load_eval_for_reporting(result_dir, task_id, model_name)
        report_accumulator.setdefault(model_name, []).append(eval_data)
    # After running all tasks, write reports if we accumulated any results
    if report_accumulator:
        out_dir = Path(result_dir) / 'reports'
        task_list = report_accumulator.get(model_name, [])
        if task_list:
            metrics_utils.write_markdown_for_model(model_name, task_list, out_dir)
            print(f"Report for model '{model_name}' written to {out_dir / (model_name + '_metrics.md')}")

if __name__ == '__main__':
    asyncio.run(main())