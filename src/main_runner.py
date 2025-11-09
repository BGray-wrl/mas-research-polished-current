#!/usr/bin/env python3
"""
Simple entrypoint to run the lead researcher agent using a YAML config.

Usage examples:
  python -m src.main_runner --config configs/dummy.yaml
  python -m src.main_runner --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml  # type: ignore

# Ensure we can import the lead researcher module despite the hyphenated folder name
THIS_DIR = Path(__file__).resolve().parent
LR_DIR = THIS_DIR / "mas-research"
if str(LR_DIR) not in sys.path:
	sys.path.insert(0, str(LR_DIR))

import lead_researcher_agent_script as lr  # type: ignore


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

	# Drive the existing pipeline via a tiny wrapper in the module
	result = lr.run_via_config(config)

	# Minimal, human-readable summary
	print("\n=== Run Summary ===")
	print(f"Mode: {config.get('mode', 'default')}")
	print(f"Question: {result.get('question')}")
	print(f"Expected: {result.get('expected_answer')}")
	print(f"Received: {result.get('recieved_answer')}")


if __name__ == "__main__":
	main()


# TODO