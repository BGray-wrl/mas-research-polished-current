# Experiments and Analysis Results

This document summarizes the four statistical experiments conducted on the multi-agent system dataset.

## Dataset Overview

- **Total Runs**: 9 (4 multiagent, 5 singleagent)
- **Metrics**: 26 columns including accuracy, cost, time, and behavioral features
- **Key Challenge**: All runs have accuracy = 0 (no successful completions)

## Experiment 1: Logistic Regression - Success Probability

**Goal**: Predict probability of success using coordination and activity metrics.

**Features**:
- Core: `num_subagents`, `subagent_success_avg`, `subagent_similarity`
- Activity: `total_tool_calls`, `lead_agent_messages`, `subagent_messages`, `total_errors`, `num_turns`

**Results**:
- Status: Could not fit model (only one class: all failures)
- Alternative: Feature statistics provided
- Key Observation: Despite 90.7% subagent completion rate, overall accuracy is 0%

**Feature Statistics (Multiagent runs only)**:
```
                            mean         std         min         max
num_subagents          11.000000    6.976150    5.000000   20.000000
subagent_success_avg    0.665821    0.014376    0.652462    0.680567
subagent_similarity     0.562066    0.103597    0.424514    0.673452
total_tool_calls      213.000000  133.868094   84.000000  378.000000
```

## Experiment 2: Multiple Linear Regression - Cost & Time Analysis

**Goal**: Predict `cost_usd` and `time_seconds` using system activity metrics.

**Predictors**: `num_subagents`, `total_tool_calls`, `num_turns`, `total_errors`, `websearch_calls`, `webfetch_calls`

### Results for Cost (USD)

- **R² Score**: 1.000 (perfect fit)
- **RMSE**: $0.063

**Top Predictors**:
1. `total_tool_calls`: +33.75 (strong positive)
2. `websearch_calls`: -26.92 (negative, multicollinearity effect)
3. `num_subagents`: -3.92 (negative, multicollinearity effect)

### Results for Time (seconds)

- **R² Score**: 0.999 (near perfect)
- **RMSE**: 15.95 seconds

**Top Predictors**:
1. `total_tool_calls`: +19,114.21 (extremely strong)
2. `websearch_calls`: -16,640.65 (negative, multicollinearity)
3. `num_subagents`: -2,317.63 (negative, multicollinearity)

**Interpretation**: 
- Tool call volume is the primary driver of both cost and time
- Nearly perfect predictions indicate strong linear relationships
- Some multicollinearity between tool types (websearch dominates tool calls)

## Experiment 3: Correlation Analysis

**Goal**: Link structural and quality metrics to outcomes.

**Variables Analyzed**:
- Outcomes: `accuracy`, `cost_usd`, `time_seconds`
- Structural: `num_turns`, `total_tool_calls`, `subagent_messages`, `total_errors`, etc.
- Quality: `subagent_success_avg`, `subagent_similarity`, `subagents_completed_pct`

### Key Correlations with Cost

| Metric | Correlation |
|--------|-------------|
| `total_tool_calls` | +0.998 |
| `num_subagents` | +0.989 |
| `subagent_messages` | +0.989 |
| `time_seconds` | +0.985 |
| `subagent_success_avg` | -0.934 |
| `total_errors` | +0.912 |

**Insights**:
- Tool calls and subagent activity strongly drive up costs
- Higher subagent success rate correlates with LOWER cost (efficiency)
- Time and cost are highly correlated (0.985)
- Accuracy correlations cannot be computed (all zeros)

**Visualization**: `experiments/correlation_heatmap.png`

## Experiment 4: PCA - Operational Modes

**Goal**: Identify emergent behavioral clusters in system operation.

**Variables**: 13 features including subagent counts, messages, tool calls, tokens, quality metrics, cost, and time.

### Principal Components

- **PC1 (61.44%)**: Activity Scale
  - Top loadings: `subagent_messages`, `num_subagents`, `websearch_calls`, `total_tool_calls`, `cost_usd`
  - Interpretation: Overall system activity level

- **PC2 (19.56%)**: Efficiency/Coordination Dimension
  - Top loadings: `total_tokens`, `subagent_success_avg`, `num_turns`, `lead_agent_messages`
  - Interpretation: Turn efficiency and token usage per action

- **Total Variance Explained**: 81.00%

### Operational Clusters

The biplot reveals clear separation:

1. **Multiagent runs** (blue): 
   - High PC1 (high activity)
   - Spread along PC2 (varying efficiency)
   
2. **Singleagent runs** (orange):
   - Low PC1 (low activity)
   - Different PC2 distribution (different turn patterns)

**Interpretation**:
- PC1 cleanly separates single vs multi-agent approaches by activity level
- PC2 captures efficiency variations within each approach
- Multiagent systems operate at ~5x activity level of single-agent

**Visualization**: `experiments/pca_biplot.png`

## Summary of Findings

### Key Insights

1. **Cost Predictability**: System costs are highly predictable (R² > 0.99) from activity metrics
   - Primary driver: total tool calls
   - Secondary: number of subagents and websearch volume

2. **Multiagent Characteristics**:
   - 5x more messages (435 vs 109)
   - 5x more tool calls (213 vs 42)
   - 3.2x higher cost per question ($7.10 vs $1.37)
   - 3.4x longer runtime (876s vs 261s)
   - 31% fewer turns (29 vs 92)

3. **Quality Paradox**:
   - 91% of individual subagents complete tasks successfully
   - Yet 0% overall system accuracy
   - Suggests coordination or integration issues

4. **Efficiency Patterns**:
   - Higher subagent success correlates with LOWER costs
   - Turn count inversely related to cost (fewer turns = higher cost)
   - Token usage per turn varies significantly (PC2 dimension)

### Recommendations for Future Research

1. **Investigate Accuracy Problem**: 
   - Why do successful subagents fail to produce correct final answers?
   - Check integration/synthesis step in lead agent

2. **Optimize Tool Usage**:
   - Reduce redundant searches (195 searches in multiagent vs 37 in singleagent)
   - Improve search query quality over quantity

3. **Balance Activity vs Accuracy**:
   - Current multiagent system optimizes for thoroughness, not correctness
   - Consider early stopping or answer validation mechanisms

4. **Collect More Data**:
   - Need runs with accuracy = 1 to fit classification models
   - Expand dataset for more robust statistical analysis

## Running the Experiments

```bash
# Run all experiments
uv run python experiments/experiments_and_visuals.py

# Or run individually
from experiments.experiments_and_visuals import *

df = load_dataset('data/dataset.csv')
results_1 = experiment_1_logistic_regression(df)
results_2 = experiment_2_linear_regression(df)
results_3 = experiment_3_correlation_analysis(df, 'experiments')
results_4 = experiment_4_pca_analysis(df, 'experiments')
```

## Output Files

- `experiments/correlation_heatmap.png` - Correlation matrix visualization
- `experiments/pca_biplot.png` - PCA scatter plots (by accuracy and run_type)
