import argparse
from datetime import datetime
import json
from pathlib import Path

import sys

# Ensure we can import the lead researcher module despite the hyphenated folder name
THIS_DIR = Path(__file__).resolve().parent
LR_DIR = THIS_DIR / "mas_research"
if str(LR_DIR) not in sys.path:
	sys.path.insert(0, str(LR_DIR))

from helpers import load_messages_from_json # type: ignore
from utils.message_serializer import serialize_message # type: ignore
from utils.metrics_analyzer import analyze_agent_metrics, print_metrics_report, get_metrics_summary # type: ignore



def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run the metric analysis with a filepath to a JSON of run result filepaths")
	parser.add_argument(
		"--results",
		type=str,
		default=str(THIS_DIR.parent / "configs/analysis_metrics_filepaths.json"),
		help="Path to JSON config (e.g., configs/analysis_metrics_filepaths.json)",
	)
	parser.add_argument(
		"--output",
		type=str,
		default='metrics_summary_current',
		help="Name to output JSON file for metrics summary (e.g., metrics_summary_current)",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	cfg_path = Path(args.results)

	if not cfg_path.exists():
		print(f"Path not found: {cfg_path}")
		sys.exit(1)

	with open(cfg_path, 'r') as f:
		run_results_paths = json.load(f)
		

	results = {}
	
	for run_type, runs in run_results_paths.items():
		results[run_type] = []
		for run_filepath in runs:
			data = load_messages_from_json(run_filepath)
			metrics = analyze_agent_metrics(data)
			results[run_type].append(metrics)
			
	if args.output:
		# with open('data'/ args.output/datetime.now().strftime("%Y%m%d%H%M%S") + ".json", 'w') as f:
		with open('data/' + args.output + ".json", 'w') as f:
			json.dump(results, f, indent=4)
			print(f"Metrics summary written to {args.output}")
	return results

if __name__ == "__main__":
	main()
