# Experiments and Analysis Results

This document summarizes the four statistical experiments conducted on the multi-agent system dataset using the **runww25** evaluation runs.

## Dataset Overview

- **Total Runs**: 50 (25 single-agent, 25 multi-agent)
- **Overall Accuracy**: 54% (27 successful runs)
- **Metrics**: 27 columns including accuracy, cost, time, correctness scores, and behavioral features
- **Key Finding**: Single-agent outperforms multi-agent in accuracy (60% vs 48%)

## Experiment 1: Logistic Regression - Success Probability

**Goal**: Predict probability of success using coordination and activity metrics.

**Features**:
- Core: `num_subagents`, `subagent_success_avg`, `subagent_similarity`
- Activity: `total_tool_calls`, `lead_agent_messages`, `subagent_messages`, `total_errors`, `num_turns`

**Results**:
- **Model Accuracy**: 72.00%
- **Status**: Successfully fitted model with both success and failure cases

**Top Feature Coefficients** (by absolute value):
1. `total_errors`: -1.22 (more errors → lower success probability)
2. `subagent_similarity`: -0.48 (higher similarity → lower success)
3. `subagent_success_avg`: -0.46 (paradoxically negative)
4. `subagent_messages`: -0.31 (more messages → lower success)
5. `total_tool_calls`: -0.30 (more tool calls → lower success)

**Key Insights**:
- Model achieves 72% accuracy in predicting run success
- Errors are the strongest predictor of failure
- Higher activity (messages, tool calls) correlates with lower success
- Suggests over-activity may indicate struggling systems rather than thoroughness
- High subagent similarity may indicate redundant work

## Experiment 2: Multiple Linear Regression - Cost & Time Analysis

**Goal**: Predict `cost_usd` and `time_seconds` using system activity metrics.

**Predictors**: `num_subagents`, `total_tool_calls`, `num_turns`, `total_errors`, `websearch_calls`, `webfetch_calls`

### Results for Cost (USD)

- **R² Score**: 0.982 (excellent fit)
- **RMSE**: $0.039

**Top Predictors**:
1. `websearch_calls`: +0.801 (strong positive)
2. `total_tool_calls`: -0.776 (negative, multicollinearity)
3. `webfetch_calls`: +0.279 (positive)
4. `num_subagents`: +0.038 (weak positive)
5. `num_turns`: -0.015 (negligible)

### Results for Time (seconds)

- **R² Score**: 0.633 (moderate fit)
- **RMSE**: 56.21 seconds

**Top Predictors**:
1. `total_tool_calls`: -232.91 (strong negative, multicollinearity)
2. `websearch_calls`: +210.49 (strong positive)
3. `webfetch_calls`: +76.04 (positive)
4. `total_errors`: +23.47 (positive)
5. `num_turns`: +15.01 (positive)

**Interpretation**: 
- Cost is highly predictable (R² = 0.98) from activity metrics
- Websearch calls are the primary cost driver (API call costs)
- Time is moderately predictable (R² = 0.63), suggesting other factors at play
- Multicollinearity between tool types (websearch dominates total tool calls)
- Errors add time but don't significantly impact cost

## Experiment 3: Correlation Analysis

**Goal**: Link structural and quality metrics to outcomes.

**Variables Analyzed**:
- Outcomes: `accuracy`, `cost_usd`, `time_seconds`
- Structural: `num_turns`, `total_tool_calls`, `subagent_messages`, `total_errors`, etc.
- Quality: `subagent_success_avg`, `subagent_similarity`, `subagents_completed_pct`

### Key Correlations with Accuracy

| Metric | Correlation |
|--------|-------------|
| `total_errors` | -0.420 |
| `time_seconds` | -0.406 |
| `cost_usd` | -0.379 |
| `total_tool_calls` | -0.372 |
| `subagent_messages` | -0.315 |
| `subagent_success_avg` | -0.248 |
| `num_subagents` | -0.201 |
| `subagents_completed_pct` | +0.196 |

### Key Correlations with Cost

| Metric | Correlation |
|--------|-------------|
| `total_tool_calls` | +0.984 |
| `subagent_messages` | +0.779 |
| `total_errors` | +0.735 |
| `time_seconds` | +0.714 |
| `num_subagents` | +0.597 |
| `subagents_completed_pct` | -0.515 |
| `subagent_similarity` | -0.501 |
| `accuracy` | -0.379 |

**Insights**:
- **Accuracy**: All major activity metrics negatively correlate with success
  - More errors, time, cost, and tool calls → lower accuracy
  - Suggests systems "thrash" when struggling with difficult questions
  - Only positive correlation: subagents_completed_pct (barely positive)
  
- **Cost Drivers**: 
  - Tool calls are the dominant factor (r = 0.984)
  - Subagent messages, errors, and time also drive up costs
  - Higher completion rates and similarity paradoxically reduce costs
  
- **Cost-Accuracy Trade-off**: 
  - Negative correlation between cost and accuracy (r = -0.379)
  - More spending does NOT lead to better results
  - Suggests inefficient resource allocation

**Visualization**: `experiments/correlation_heatmap.png`

## Experiment 4: PCA - Operational Modes

**Goal**: Identify emergent behavioral clusters in system operation.

**Variables**: 13 features including subagent counts, messages, tool calls, tokens, quality metrics, cost, and time.

### Principal Components

- **PC1 (71.38%)**: Activity Scale & System Complexity
  - Top loadings: `total_tool_calls` (0.321), `subagent_messages` (0.320), `cost_usd` (0.314), `websearch_calls` (0.312), `num_subagents` (0.301)
  - Interpretation: Overall system activity level and resource consumption

- **PC2 (12.31%)**: Quality & Efficiency Dimension
  - Top loadings: `subagent_success_avg` (0.484), `subagent_similarity` (0.474), `time_seconds` (0.378), `total_errors` (0.323), `lead_agent_messages` (0.271)
  - Interpretation: Coordination quality and error patterns

- **Total Variance Explained**: 83.68%

### Operational Clusters

The biplot reveals clear separation between run types:

1. **Multi-agent runs** (typically): 
   - Moderate to high PC1 (higher activity)
   - Variable PC2 (quality varies significantly)
   - 1.66x more tool calls than single-agent
   - 98.7% subagent completion rate
   
2. **Single-agent runs** (typically):
   - Lower PC1 (lower activity)
   - Different PC2 distribution
   - More turns but fewer tool calls
   - No subagent coordination overhead

**Key Observations**:
- PC1 captures the multi vs single-agent architectural difference
- PC2 shows that multi-agent quality varies more (some runs very efficient, others struggle)
- Multi-agent systems have higher activity but lower accuracy overall
- Successful runs (both types) tend toward lower PC1 scores (less activity)

**Visualization**: `experiments/pca_biplot.png`

## Summary of Findings

### System Performance Comparison

**Single-Agent Characteristics**:
- Accuracy: 60% (15/25 successful)
- Avg cost: $0.26 per question
- Avg time: 108.1 seconds
- Avg turns: 34.5
- Avg tool calls: 13.0
- Avg messages: 36.5
- Avg correctness score: 7.20/10
- Confidence: 100%

**Multi-Agent Characteristics**:
- Accuracy: 48% (12/25 successful)
- Avg cost: $0.42 per question
- Avg time: 128.1 seconds
- Avg turns: 4.5 (13% of single-agent!)
- Avg subagents: 1.2
- Avg tool calls: 21.5
- Avg messages: 47.0 (6.5 lead + 40.5 subagent)
- Avg websearch calls: 13.6 (vs 8.9 single-agent)
- Subagent similarity: 0.967 (very high redundancy)
- Subagent success rate: 75.5%
- Subagent completion rate: 98.7%
- Avg correctness score: 6.08/10
- Confidence: 100%

**Comparison (Multi vs Single)**:
- **Accuracy**: -12 percentage points (worse)
- **Cost**: 1.60x higher
- **Time**: 1.19x longer
- **Tool calls**: 1.66x more
- **Messages**: 1.29x more
- **Turns**: 0.13x (87% fewer turns!)
- **Correctness**: 1.12 points lower (on 10-point scale)

### Key Insights

1. **Single-Agent Outperforms Multi-Agent**:
   - Higher accuracy despite lower cost and activity
   - Better correctness scores (7.2 vs 6.1)
   - More deliberate approach with 7.7x more turns
   - Less resource-intensive overall

2. **Multi-Agent Inefficiencies**:
   - 66% more tool calls but 12% lower accuracy
   - Very high subagent similarity (0.967) suggests redundant work
   - 98.7% completion rate but 48% accuracy indicates integration/synthesis issues
   - Individual subagents succeed (75.5%) but system fails to leverage results
   - Fewer turns suggest rapid delegation without sufficient coordination

3. **Activity-Accuracy Paradox**:
   - More activity correlates with LOWER accuracy (r = -0.372 for tool calls)
   - Higher costs correlate with LOWER accuracy (r = -0.379)
   - Errors strongly predict failure (coefficient = -1.22)
   - Systems appear to "thrash" when struggling rather than succeed through thoroughness

4. **Cost Predictability**:
   - Cost is highly predictable (R² = 0.982) from activity metrics
   - Websearch calls are the primary driver
   - Multi-agent systems cost 60% more but deliver less value

5. **Quality Patterns**:
   - High subagent similarity indicates lack of diverse approaches
   - Subagent success doesn't translate to system success
   - Lead agent may not effectively synthesize subagent outputs
   - Turn economy matters: single-agent's iterative approach works better

### Recommendations for Future Research

1. **Improve Multi-Agent Coordination**:
   - Investigate why 75.5% subagent success → 48% system accuracy
   - Enhance lead agent's synthesis and integration capabilities
   - Reduce redundancy (current similarity = 0.967)
   - Add validation/review steps before final answers

2. **Optimize Resource Allocation**:
   - Reduce unnecessary tool calls in multi-agent (currently 66% more)
   - Improve search query quality over quantity
   - Consider early stopping when confidence is low
   - Balance turn count (not too few, not too many)

3. **Address Activity-Accuracy Paradox**:
   - High activity often indicates struggling, not effectiveness
   - Implement confidence-based early termination
   - Monitor error rates and adjust strategy dynamically
   - Consider single-agent approach for simpler questions

4. **Enhance Subagent Diversity**:
   - Current similarity of 0.967 is too high
   - Assign more distinct subtasks to reduce overlap
   - Encourage different search strategies across subagents
   - Improve task decomposition in lead agent

5. **Balance Turns vs Efficiency**:
   - Multi-agent uses only 13% of single-agent turns
   - May be delegating too quickly without sufficient thought
   - Explore hybrid approaches with more lead agent deliberation
   - Add checkpoints between delegation and synthesis

6. **Cost-Benefit Analysis**:
   - 60% higher cost for 12% lower accuracy is poor ROI
   - Consider adaptive strategy: start single-agent, escalate if needed
   - Optimize for cost-per-correct-answer, not just accuracy
   - Single-agent: $0.43/success, Multi-agent: $0.87/success (2x worse)

## Running the Experiments

```bash
# Run all experiments
uv run python experiments/experiments_and_visuals.py

# Or run individually
from experiments.experiments_and_visuals import *

df = load_dataset('data/runww25.csv')
results_1 = experiment_1_logistic_regression(df)
results_2 = experiment_2_linear_regression(df)
results_3 = experiment_3_correlation_analysis(df, 'experiments')
results_4 = experiment_4_pca_analysis(df, 'experiments')
```

## Output Files

- `data/runww25.csv` - Complete dataset with 50 runs
- `experiments/correlation_heatmap.png` - Correlation matrix visualization
- `experiments/pca_biplot.png` - PCA scatter plots (by accuracy and run_type)

## Conclusion

This analysis reveals that the multi-agent architecture, while theoretically promising, underperforms the simpler single-agent approach on this dataset. The key issues are:

1. **Poor ROI**: Multi-agent costs 60% more but achieves 12% lower accuracy
2. **Coordination failure**: Individual subagents succeed but system fails to synthesize
3. **Redundancy**: Very high subagent similarity (0.967) indicates duplicate work
4. **Turn imbalance**: Too few turns (4.5) suggests insufficient deliberation
5. **Resource waste**: 66% more tool calls without accuracy benefit

The single-agent system's iterative, deliberate approach (34.5 turns avg) outperforms the multi-agent's rapid delegation strategy. Future work should focus on improving multi-agent coordination, reducing redundancy, and adding synthesis validation steps.

