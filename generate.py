#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

import os
from utils import core as utils

load_dotenv()


def process_task(task_item, data_dir: str, result_dir: str, model_name: str, coverage: bool = False):
	task_id = str(task_item.get('id'))


	task_path = os.path.join(data_dir, task_id)
	print(f"Processing task: {task_path}")
	video_path, stitched_images = utils.find_files(task_path)

	if not os.path.exists(video_path) and not stitched_images:
		return task_id, "No media found"

	if isinstance(task_item, dict):
		resolution = task_item.get('recording_details', {}).get('resolution') or utils.get_resolution(task_path)
	else:
		resolution = utils.get_resolution(task_path)

	model_dir = model_name.lower().replace('/', '__')
	output_dir = os.path.join(result_dir, task_id, model_dir)
	os.makedirs(output_dir, exist_ok=True)
	output_file = os.path.join(output_dir, 'index.html')

	if coverage and os.path.exists(output_file):
		try:
			os.remove(output_file)
		except Exception:
			pass

	if os.path.exists(output_file):
		return task_id, 'Already exists'

	temp_folder = os.path.join(output_dir, 'temp_frames')
	if os.path.exists(temp_folder):
		import shutil
		shutil.rmtree(temp_folder)

	print(f"Calling model for task {task_id} (model={model_name})")
	if model_name == 'mock':
		config = utils.get_config(model_name)
		temp_images = utils.prepare_images_in_temp_folder(stitched_images, video_path, temp_folder, config)
		raw = utils.mock_model_call("", temp_images, resolution)
	else:
		raw = utils.call_mllm_html(model_name, stitched_images, video_path, temp_folder, resolution)

	assets_dir = os.path.join(data_dir, task_id, 'assets')
	assets_path = os.path.relpath(assets_dir, output_dir).replace('\\', '/')
	html = utils.post_process(raw, assets_path)

	with open(output_file, 'w', encoding='utf-8') as f:
		f.write(html)

	return task_id, 'Done'



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', required=True, help='Path to dataset directory containing task folders or an example.jsonl file')
	parser.add_argument('--result', required=True, help='Path to write generated outputs')
	parser.add_argument('--model', default='mock', help='Model name (mock by default)')
	parser.add_argument('--workers', type=int, default=4)
	parser.add_argument('--coverage', action='store_true', help='If set, force regeneration by deleting existing index.html before generating')
	args = parser.parse_args()

	jsonl_path = args.data
	result_path = args.result

	base = os.path.splitext(os.path.basename(jsonl_path))[0]
	data_root = os.path.join(os.path.dirname(jsonl_path), base)
	if not os.path.isdir(data_root):
		print(f"Tasks directory not found next to {jsonl_path}: expected {data_root}")
		return

	tasks = utils.load_examples(jsonl_path)
	if not tasks:
		print('No tasks found')
		return

	print(f"Processing {len(tasks)} tasks with {args.workers} workers using model {args.model}")

	with ThreadPoolExecutor(max_workers=args.workers) as ex:
		ordered = sorted(tasks, key=lambda x: (x.get('id') if isinstance(x, dict) else x))
		futures = {ex.submit(process_task, t, data_root, result_path, args.model, args.coverage): t for t in ordered}
		completed = 0
		for future in as_completed(futures):
			task, status = future.result()
			completed += 1
			print(f"[{completed}/{len(tasks)}] {task}: {status}")


if __name__ == '__main__':
	main()
