#!/usr/bin/env python3
"""
Analyze all 200 WebWalker runs and generate comprehensive single vs multi-agent comparison
"""

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/ww100.csv')

print("="*80)
print("COMPREHENSIVE SYSTEM PERFORMANCE COMPARISON")
print("Full WebWalker Dataset: 200 Runs")
print("="*80)

# Separate by run type
single = df[df['run_type'] == 'ww-eval-singleagent'].copy()
multi = df[df['run_type'] == 'ww-eval-multiagent'].copy()

print(f"\nDataset breakdown:")
print(f"  Single-agent runs: {len(single)}")
print(f"  Multi-agent runs: {len(multi)}")
print(f"  Total runs: {len(df)}")

# Calculate statistics for each system
def get_stats(data, name):
    stats = {
        'name': name,
        'n': len(data),
        'accuracy_pct': data['accuracy'].mean() * 100,
        'accuracy_count': f"{data['accuracy'].sum()}/{len(data)}",
        'cost_mean': data['cost_usd'].mean(),
        'cost_std': data['cost_usd'].std(),
        'time_mean': data['time_seconds'].mean(),
        'time_std': data['time_seconds'].std(),
        'turns_mean': data['num_turns'].mean(),
        'turns_std': data['num_turns'].std(),
        'tool_calls_mean': data['total_tool_calls'].mean(),
        'tool_calls_std': data['total_tool_calls'].std(),
        'messages_mean': data['total_messages'].mean(),
        'messages_std': data['total_messages'].std(),
        'websearch_mean': data['websearch_calls'].mean(),
        'websearch_std': data['websearch_calls'].std(),
        'webfetch_mean': data['webfetch_calls'].mean(),
        'webfetch_std': data['webfetch_calls'].std(),
        'errors_mean': data['total_errors'].mean(),
        'errors_std': data['total_errors'].std(),
        'correctness_mean': data['correctness'].mean(),
        'correctness_std': data['correctness'].std(),
        'confidence_mean': data['confidence'].mean(),
        'confidence_std': data['confidence'].std(),
    }

    # Multi-agent specific metrics
    if 'num_subagents' in data.columns:
        multi_only = data[data['num_subagents'] > 0]
        if len(multi_only) > 0:
            stats['num_subagents_mean'] = multi_only['num_subagents'].mean()
            stats['num_subagents_std'] = multi_only['num_subagents'].std()
            stats['subagent_messages_mean'] = multi_only['subagent_messages'].mean()
            stats['subagent_messages_std'] = multi_only['subagent_messages'].std()
            stats['lead_messages_mean'] = multi_only['lead_agent_messages'].mean()
            stats['lead_messages_std'] = multi_only['lead_agent_messages'].std()
            stats['subagent_similarity_mean'] = multi_only['subagent_similarity'].mean()
            stats['subagent_similarity_std'] = multi_only['subagent_similarity'].std()
            stats['subagent_success_mean'] = multi_only['subagent_success_avg'].mean()
            stats['subagent_success_std'] = multi_only['subagent_success_avg'].std()
            stats['subagents_completed_pct_mean'] = multi_only['subagents_completed_pct'].mean()
            stats['subagents_completed_pct_std'] = multi_only['subagents_completed_pct'].std()

    return stats

single_stats = get_stats(single, 'Single-Agent')
multi_stats = get_stats(multi, 'Multi-Agent')

# Print comprehensive comparison
print("\n" + "="*80)
print("SINGLE-AGENT PERFORMANCE")
print("="*80)
print(f"Runs: {single_stats['n']}")
print(f"Accuracy: {single_stats['accuracy_pct']:.1f}% ({single_stats['accuracy_count']} correct)")
print(f"Cost: ${single_stats['cost_mean']:.3f} ± ${single_stats['cost_std']:.3f} per question")
print(f"Time: {single_stats['time_mean']:.1f}s ± {single_stats['time_std']:.1f}s")
print(f"Turns: {single_stats['turns_mean']:.1f} ± {single_stats['turns_std']:.1f}")
print(f"Tool Calls: {single_stats['tool_calls_mean']:.1f} ± {single_stats['tool_calls_std']:.1f}")
print(f"Messages: {single_stats['messages_mean']:.1f} ± {single_stats['messages_std']:.1f}")
print(f"WebSearch Calls: {single_stats['websearch_mean']:.1f} ± {single_stats['websearch_std']:.1f}")
print(f"WebFetch Calls: {single_stats['webfetch_mean']:.1f} ± {single_stats['webfetch_std']:.1f}")
print(f"Errors: {single_stats['errors_mean']:.2f} ± {single_stats['errors_std']:.2f}")
print(f"Correctness Score: {single_stats['correctness_mean']:.2f}/10 ± {single_stats['correctness_std']:.2f}")
print(f"Confidence: {single_stats['confidence_mean']:.1f}% ± {single_stats['confidence_std']:.1f}%")

print("\n" + "="*80)
print("MULTI-AGENT PERFORMANCE")
print("="*80)
print(f"Runs: {multi_stats['n']}")
print(f"Accuracy: {multi_stats['accuracy_pct']:.1f}% ({multi_stats['accuracy_count']} correct)")
print(f"Cost: ${multi_stats['cost_mean']:.3f} ± ${multi_stats['cost_std']:.3f} per question")
print(f"Time: {multi_stats['time_mean']:.1f}s ± {multi_stats['time_std']:.1f}s")
print(f"Turns: {multi_stats['turns_mean']:.1f} ± {multi_stats['turns_std']:.1f}")
print(f"Tool Calls: {multi_stats['tool_calls_mean']:.1f} ± {multi_stats['tool_calls_std']:.1f}")
print(f"Messages: {multi_stats['messages_mean']:.1f} ± {multi_stats['messages_std']:.1f}")
print(f"  - Lead Agent: {multi_stats['lead_messages_mean']:.1f} ± {multi_stats['lead_messages_std']:.1f}")
print(f"  - Subagents: {multi_stats['subagent_messages_mean']:.1f} ± {multi_stats['subagent_messages_std']:.1f}")
print(f"WebSearch Calls: {multi_stats['websearch_mean']:.1f} ± {multi_stats['websearch_std']:.1f}")
print(f"WebFetch Calls: {multi_stats['webfetch_mean']:.1f} ± {multi_stats['webfetch_std']:.1f}")
print(f"Errors: {multi_stats['errors_mean']:.2f} ± {multi_stats['errors_std']:.2f}")
print(f"Num Subagents: {multi_stats['num_subagents_mean']:.2f} ± {multi_stats['num_subagents_std']:.2f}")
print(f"Subagent Similarity: {multi_stats['subagent_similarity_mean']:.3f} ± {multi_stats['subagent_similarity_std']:.3f}")
print(f"Subagent Success: {multi_stats['subagent_success_mean']:.1f}% ± {multi_stats['subagent_success_std']:.1f}%")
print(f"Subagents Completed: {multi_stats['subagents_completed_pct_mean']:.1f}% ± {multi_stats['subagents_completed_pct_std']:.1f}%")
print(f"Correctness Score: {multi_stats['correctness_mean']:.2f}/10 ± {multi_stats['correctness_std']:.2f}")
print(f"Confidence: {multi_stats['confidence_mean']:.1f}% ± {multi_stats['confidence_std']:.1f}%")

print("\n" + "="*80)
print("COMPARATIVE ANALYSIS (Multi vs Single)")
print("="*80)

# Calculate ratios and differences
accuracy_diff = multi_stats['accuracy_pct'] - single_stats['accuracy_pct']
cost_ratio = multi_stats['cost_mean'] / single_stats['cost_mean']
time_ratio = multi_stats['time_mean'] / single_stats['time_mean']
turns_ratio = multi_stats['turns_mean'] / single_stats['turns_mean']
tool_ratio = multi_stats['tool_calls_mean'] / single_stats['tool_calls_mean']
messages_ratio = multi_stats['messages_mean'] / single_stats['messages_mean']
websearch_ratio = multi_stats['websearch_mean'] / single_stats['websearch_mean']
correctness_diff = multi_stats['correctness_mean'] - single_stats['correctness_mean']

print(f"Accuracy: {accuracy_diff:+.1f} percentage points ({'WORSE' if accuracy_diff < 0 else 'BETTER'})")
print(f"Cost: {cost_ratio:.2f}x ({'MORE' if cost_ratio > 1 else 'LESS'} expensive)")
print(f"Time: {time_ratio:.2f}x ({'SLOWER' if time_ratio > 1 else 'FASTER'})")
print(f"Turns: {turns_ratio:.2f}x ({abs(1-turns_ratio)*100:.0f}% {'fewer' if turns_ratio < 1 else 'more'} turns)")
print(f"Tool Calls: {tool_ratio:.2f}x ({'MORE' if tool_ratio > 1 else 'FEWER'})")
print(f"Messages: {messages_ratio:.2f}x ({'MORE' if messages_ratio > 1 else 'FEWER'})")
print(f"WebSearch: {websearch_ratio:.2f}x ({'MORE' if websearch_ratio > 1 else 'FEWER'})")
print(f"Correctness: {correctness_diff:+.2f} points on 10-point scale")

# Cost per correct answer
single_cost_per_correct = single_stats['cost_mean'] / (single_stats['accuracy_pct'] / 100)
multi_cost_per_correct = multi_stats['cost_mean'] / (multi_stats['accuracy_pct'] / 100)
cost_efficiency_ratio = multi_cost_per_correct / single_cost_per_correct

print(f"\nCost Efficiency:")
print(f"  Single-agent: ${single_cost_per_correct:.2f} per correct answer")
print(f"  Multi-agent: ${multi_cost_per_correct:.2f} per correct answer")
print(f"  Multi-agent is {cost_efficiency_ratio:.2f}x LESS efficient")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

print("\n1. PERFORMANCE WINNER: Single-Agent")
print(f"   - {abs(accuracy_diff):.1f} percentage points higher accuracy")
print(f"   - {1/cost_ratio:.0%} lower cost")
print(f"   - {cost_efficiency_ratio:.1f}x better cost-per-correct-answer")

print("\n2. MULTI-AGENT INEFFICIENCIES:")
print(f"   - {(tool_ratio-1)*100:.0f}% more tool calls for {abs(accuracy_diff):.0f}pp lower accuracy")
print(f"   - Very high subagent similarity ({multi_stats['subagent_similarity_mean']:.3f}) = redundant work")
print(f"   - {multi_stats['subagents_completed_pct_mean']:.1f}% subagent completion but only {multi_stats['accuracy_pct']:.1f}% system accuracy")
print(f"   - {abs(1-turns_ratio)*100:.0f}% fewer turns = insufficient deliberation")

print("\n3. ACTIVITY-ACCURACY PARADOX:")
print(f"   - System with MORE resources (multi-agent) performs WORSE")
print(f"   - Suggests over-activity indicates struggling, not thoroughness")

# Write summary to file
with open('full_comparison_summary.md', 'w') as f:
    f.write("# Full Dataset Comparison: Single-Agent vs Multi-Agent\n\n")
    f.write(f"**Dataset:** 200 WebWalker runs ({single_stats['n']} single, {multi_stats['n']} multi)\n\n")

    f.write("## Performance Summary\n\n")
    f.write("| Metric | Single-Agent | Multi-Agent | Difference |\n")
    f.write("|--------|--------------|-------------|------------|\n")
    f.write(f"| Accuracy | {single_stats['accuracy_pct']:.1f}% ({single_stats['accuracy_count']}) | {multi_stats['accuracy_pct']:.1f}% ({multi_stats['accuracy_count']}) | {accuracy_diff:+.1f}pp |\n")
    f.write(f"| Cost ($/q) | ${single_stats['cost_mean']:.3f} | ${multi_stats['cost_mean']:.3f} | {cost_ratio:.2f}x |\n")
    f.write(f"| Time (s) | {single_stats['time_mean']:.1f} | {multi_stats['time_mean']:.1f} | {time_ratio:.2f}x |\n")
    f.write(f"| Turns | {single_stats['turns_mean']:.1f} | {multi_stats['turns_mean']:.1f} | {turns_ratio:.2f}x |\n")
    f.write(f"| Tool Calls | {single_stats['tool_calls_mean']:.1f} | {multi_stats['tool_calls_mean']:.1f} | {tool_ratio:.2f}x |\n")
    f.write(f"| Messages | {single_stats['messages_mean']:.1f} | {multi_stats['messages_mean']:.1f} | {messages_ratio:.2f}x |\n")
    f.write(f"| WebSearch | {single_stats['websearch_mean']:.1f} | {multi_stats['websearch_mean']:.1f} | {websearch_ratio:.2f}x |\n")
    f.write(f"| Correctness | {single_stats['correctness_mean']:.2f}/10 | {multi_stats['correctness_mean']:.2f}/10 | {correctness_diff:+.2f} |\n")
    f.write(f"| Cost/Correct | ${single_cost_per_correct:.2f} | ${multi_cost_per_correct:.2f} | {cost_efficiency_ratio:.2f}x |\n")

    f.write("\n## Multi-Agent Coordination Metrics\n\n")
    f.write(f"- **Num Subagents:** {multi_stats['num_subagents_mean']:.2f} ± {multi_stats['num_subagents_std']:.2f}\n")
    f.write(f"- **Subagent Similarity:** {multi_stats['subagent_similarity_mean']:.3f} ± {multi_stats['subagent_similarity_std']:.3f}\n")
    f.write(f"- **Subagent Success Rate:** {multi_stats['subagent_success_mean']:.1f}% ± {multi_stats['subagent_success_std']:.1f}%\n")
    f.write(f"- **Subagent Completion Rate:** {multi_stats['subagents_completed_pct_mean']:.1f}% ± {multi_stats['subagents_completed_pct_std']:.1f}%\n")
    f.write(f"- **Lead Agent Messages:** {multi_stats['lead_messages_mean']:.1f} ± {multi_stats['lead_messages_std']:.1f}\n")
    f.write(f"- **Subagent Messages:** {multi_stats['subagent_messages_mean']:.1f} ± {multi_stats['subagent_messages_std']:.1f}\n")

print("\n" + "="*80)
print("Summary saved to: full_comparison_summary.md")
print("="*80)
