from pathlib import Path
from collections import defaultdict
from typing import List, Dict
import json


def calculate_visual_score(ocr_score, dino_score, mllm_score):
    return (ocr_score + dino_score) / 2 * 0.7 + mllm_score * 0.3


def aggregate_requested_scores(task_metrics: Dict) -> Dict:
    total_steps = task_metrics.get('total_step', 0)
    success = task_metrics.get('success_step', 0)
    ifs = (success / total_steps) if total_steps > 0 else 0.0
    
    ocr_list = task_metrics.get('all_ocr_scores', [])
    dino_list = task_metrics.get('all_dino_scores', [])
    mllm_list = task_metrics.get('all_mllm_scores', [])
    
    ocr_list = ocr_list[:success+1]
    dino_list = dino_list[:success+1]
    mllm_list = mllm_list[:success+1]

    ocr_score = sum(ocr_list) / len(ocr_list) if ocr_list else 0.0
    dino_score = sum(dino_list) / len(dino_list) if dino_list else 0.0
    hfs = sum(mllm_list) / len(mllm_list) if mllm_list else 0.0

    lfs = (ocr_score + dino_score) / 2



    vfs = 0.5 * lfs + 0.5 * hfs

    final = 0.7 * ifs + 0.3 * vfs

    return {
        'IFS': ifs,
        'ocr_score': ocr_score,
        'dino_score': dino_score,
        'LFS': lfs,
        'HFS': hfs,
        'VFS': vfs,
        'Final': final
    }


def write_markdown_for_model(model_name: str, task_list: List[Dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_name.replace('/', '__')
    md_path = out_dir / f"{model_name}_metrics.md"

    lines = []
    lines.append(f"# model: {model_name} result\n")
    lines.append("| task | IFS(%) | OCR(%) | DINO(%) | LFS(%) | HFS(%) | VFS(%) | Final(%) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    agg_totals = defaultdict(float)
    count = 0

    for t in task_list:
        task_name = t.get('task_name', 'unknown')
        scores = aggregate_requested_scores(t)
        # Multiply displayed metrics by 100 to present percentages
        lines.append("| {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |".format(
            task_name,
            scores['IFS'] * 100, scores['ocr_score'] * 100, scores['dino_score'] * 100,
            scores['LFS'] * 100, scores['HFS'] * 100, scores['VFS'] * 100, scores['Final'] * 100
        ))

        for k, v in scores.items():
            agg_totals[k] += v

        count += 1

    if count > 0:
        # Append average as the final row of the same table (one-row summary)
        avg_vals = []
        for key in ['IFS', 'ocr_score', 'dino_score', 'LFS', 'HFS', 'VFS', 'Final']:
            avg = agg_totals[key] / count
            # convert to percentage for display
            avg_vals.append(avg * 100)
        lines.append('| avg | ' + ' | '.join(f"{v:.2f}" for v in avg_vals) + ' |')

    md_text = '\n'.join(lines)
    md_path.write_text(md_text, encoding='utf-8')
    return md_path


def load_eval_for_reporting(result_dir: Path, task_id: str, model_name: str) -> Dict:
    eval_path = Path(result_dir) / task_id / model_name.lower().replace('/', '__') / 'eval.json'
    text = eval_path.read_text(encoding='utf-8')
    data = json.loads(text)
    data['task_name'] = task_id
    return data
