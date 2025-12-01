from yaml import warnings
import json
from pprint import pprint

import sys
from pathlib import Path

# Ensure we can import the lead researcher module despite the hyphenated folder name
THIS_DIR = Path(__file__).resolve().parent
LR_DIR = THIS_DIR / "mas_research"
if str(LR_DIR) not in sys.path:
	sys.path.insert(0, str(LR_DIR))

from helpers import load_messages_from_json # type: ignore
from utils.message_serializer import serialize_message # type: ignore
from utils.metrics_analyzer import analyze_agent_metrics, print_metrics_report, get_metrics_summary # type: ignore



def get_result_from_messages(messages = None, filepath = None):
    
    if messages is None:
        if filepath:
            messages = load_messages_from_json(filepath)
        else:
            print("No messages or filepath provided.")
            return None
        
    if type(messages) != list:
        warnings.warn("Messages is not a list.")
    
    elif type(messages[0]) != dict:
        warnings.warn("Messages do not appear to be serialized dictionaries. Attempting to serialize...")
        messages = [serialize_message(msg) for msg in messages]
    
    for msg in messages:
        if msg.get('type') == "ResultMessage":
            result = msg.get('data', {}).get('result', None)
            if result:
                return result
    return None


test_local_filepath = "results/dummy-multiagent/current.json"
test_local_filepath = "results/eval-singleagent/Nov-09-06-24/result14.json"
test_local_filepath = "results/eval-multiagent/Nov-09-06-54/result12.json"
test_local_filepath = "results/ww-test-multiagent/Nov-17-10-37/result0.json"
test_local_filepath = "results/ww-eval-multiagent/Nov-25-23-31/result76.json"






if __name__ == "__main__":
    # Load the data as a dict, not just messages
    with open(test_local_filepath, 'r') as f:
        data = json.load(f)
    
    # Analyze and print metrics
    metrics = analyze_agent_metrics(data)
    print_metrics_report(metrics)
    
    # Get condensed summary
    summary = get_metrics_summary(data)
    print("\n🎯 QUICK SUMMARY:")
    print(f"  Subagent Calls: {summary['subagent_calls']}")
    print(f"  Tool Calls: {summary['total_tool_calls']}")
    print(f"  Errors: {summary['errors']}")
    print(f"  Cost: ${summary['cost_usd']:.4f}")
    print(f"  Duration: {summary['duration_seconds']:.1f}s")
    print(f"  Tokens: {summary['total_tokens']:,}")
    print(f"  Grade: {summary['grade']}")
    print(f"  Correctness Score: {summary['correctness']}/10.0" if summary.get('correctness', None) is not None else "")


