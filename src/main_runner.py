#!/usr/bin/env python3
"""
Simple entrypoint to run the lead researcher agent using a YAML config.

Usage examples:
  python -m src.main_runner --config configs/dummy.yaml
  python -m src.main_runner --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
import yaml  # type: ignore

# Ensure we can import the lead researcher module despite the hyphenated folder name
THIS_DIR = Path(__file__).resolve().parent
LR_DIR = THIS_DIR / "mas_research"
if str(LR_DIR) not in sys.path:
	sys.path.insert(0, str(LR_DIR))

import researcher_agent_script as lr  # type: ignore
import run_eval as reval  # type: ignore


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run the lead researcher with a YAML config")
	parser.add_argument(
		"--config",
		type=str,
		default=str(THIS_DIR.parent / "configs" / "default.yaml"),
		help="Path to YAML config (e.g., configs/dummy.yaml or configs/default.yaml)",
	)
	return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
	with open(path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}


def main() -> None:
	args = parse_args()
	cfg_path = Path(args.config)

	if not cfg_path.exists():
		print(f"Config not found: {cfg_path}")
		sys.exit(1)

	config = load_yaml(cfg_path)
	mode = config.get("mode", "default")

	if mode == "eval":
		print(f"Running in EVAL mode using config: {cfg_path}")
		result = asyncio.run(reval.main_browsecomp_eval(config))
	else:
		# Drive the existing pipeline via a tiny wrapper in the module
		result = lr.run_via_config(config)

	# Minimal, human-readable summary
	if True:
		print("\n=== Run Summary ===")
		print(f"Mode: {config.get('mode', 'default')}")

		if type(result) is not list:
			print(f"Question: {result.get('question', '')}")
			print(f"Expected: {result.get('expected_answer', '')}")
			print(f"Received: {result.get('recieved_answer', '')}")
		else:
			print(f"Total runs: {len(result)}")
			for idx, res in enumerate(result):
				print(f"\n--- Run {idx+1} ---")
				print(f"Question: {res.get('question', '')}")
				print(f"Expected: {res.get('expected_answer', '')}")
				print(f"Received: {res.get('recieved_answer', '')}")

if __name__ == "__main__":
	main()

