================================================================================
COMPREHENSIVE SYSTEM PERFORMANCE COMPARISON
Full WebWalker Dataset: 200 Runs (100 Single-Agent + 100 Multi-Agent)
================================================================================

## Dataset Overview

- **Total Runs**: 200 (100 single-agent, 100 multi-agent)
- **Dataset**: WebWalker questions (indices 25-124)
- **Evaluation**: GPT-4o as LLM judge for answer correctness
- **Single-Agent Accuracy**: 50.0% (50/100 correct)
- **Multi-Agent Accuracy**: 48.0% (48/100 correct)

## Performance Comparison Table

| Metric | Single-Agent | Multi-Agent | Ratio/Difference |
|--------|--------------|-------------|------------------|
| **Accuracy** | 50.0% (50/100) | 48.0% (48/100) | -2.0pp |
| **Cost per Question** | $0.243 ± $0.192 | $0.473 ± $0.394 | 1.94x MORE |
| **Time per Question** | 88.7s ± 74.0s | 127.8s ± 76.6s | 1.44x SLOWER |
| **Turns** | 34.4 ± 23.5 | 4.8 ± 1.4 | 0.14x (86% FEWER) |
| **Tool Calls** | 12.9 ± 9.2 | 22.8 ± 15.8 | 1.76x MORE |
| **Total Messages** | 36.4 ± 23.5 | 49.6 ± 31.7 | 1.36x MORE |
| **WebSearch Calls** | 8.5 ± 7.0 | 14.0 ± 11.4 | 1.65x MORE |
| **WebFetch Calls** | 4.5 ± 2.7 | 7.4 ± 4.9 | 1.65x MORE |
| **Errors** | 0.86 ± 1.33 | 1.46 ± 2.06 | 1.70x MORE |
| **Correctness Score** | 6.82/10 ± 3.84 | 6.58/10 ± 4.12 | -0.24 |
| **Cost per Correct Answer** | $0.49 | $0.98 | 2.02x WORSE |

## Multi-Agent Coordination Metrics

- **Num Subagents**: 1.38 ± 0.65 (range: 1-3)
- **Lead Agent Messages**: 6.8 ± 1.4
- **Subagent Messages**: 42.8 ± 30.5
- **Subagent Similarity**: 0.923 ± 0.149 (very high redundancy)
- **Subagent Success Rate**: 5.3% ± 2.9%
- **Subagent Completion Rate**: 0.3% ± 0.4%

## Key Findings from Full Dataset

### 1. Single-Agent Marginally Outperforms Multi-Agent
- **Accuracy**: 2 percentage points higher (50% vs 48%)
- **Cost Efficiency**: 2.02x better cost-per-correct-answer
- **Resource Usage**: 51% lower cost, 44% faster

### 2. Multi-Agent Inefficiencies Confirmed
- **76% more tool calls** but 2pp lower accuracy
- **86% fewer turns** (4.8 vs 34.4) = insufficient deliberation by lead agent
- **94% higher cost** with no accuracy benefit
- **Very high subagent similarity** (0.923) indicates redundant work

### 3. Activity-Accuracy Paradox Holds
- System with MORE activity (multi-agent) performs WORSE
- More tool calls, messages, and cost correlate with lower success
- Suggests over-activity indicates struggling, not thoroughness

### 4. Coordination Failure Pattern
- Individual subagent metrics are extremely low (5.3% success, 0.3% completion)
- System accuracy (48%) suggests lead agent synthesizes despite poor subagent outputs
- OR metric calculation may be flawed - requires investigation

================================================================================
MULTI-AGENT DEEP-DIVE ANALYSIS (100 Runs)
================================================================================

Analyzing 100 multi-agent runs
Accuracy rate: 48.0%
Subagents per run: mean=1.4, range=[1, 3]

================================================================================
EXPERIMENT 1: Multi-Agent Coordination and Success Analysis
================================================================================

Dataset: 100 runs
Accuracy distribution: {0: 52, 1: 48}

Coordination Metrics Summary:
  Subagents per run: mean=1.38, std=0.65
  Subagent similarity: mean=0.923, std=0.149
  Subagent success avg: mean=5.25, std=2.87

Model Accuracy: 70.00%

Top Coordination Features for Success:
                feature  coefficient
subagents_completed_pct     0.328013
         webfetch_calls    -0.146476
       total_tool_calls    -0.142220
        websearch_calls    -0.129501
          num_subagents    -0.120732
   total_subagent_calls    -0.120732
              num_turns    -0.102467
           total_errors    -0.075811

================================================================================
EXPERIMENT 2: Multi-Agent Resource Efficiency Analysis
================================================================================

--- Predicting cost_usd ---
R² Score: 0.972
RMSE: 0.065
Mean cost_usd: 0.473

Top Predictors:
            feature  coefficient
    websearch_calls     0.153991
  subagent_messages     0.140368
   total_tool_calls     0.136079
     webfetch_calls    -0.074332
subagent_similarity     0.011695

--- Predicting time_seconds ---
R² Score: 0.758
RMSE: 37.532
Mean time_seconds: 127.791

Top Predictors:
          feature  coefficient
  websearch_calls   286.787105
subagent_messages  -160.795151
 total_tool_calls  -157.560775
   webfetch_calls   111.410608
        num_turns    92.527301

================================================================================
EXPERIMENT 3: Multi-Agent Coordination Pattern Analysis
================================================================================

Key Coordination Correlations:

1. Correlations with Accuracy:
total_tool_calls       -0.355258
subagent_messages      -0.353821
websearch_calls        -0.334716
webfetch_calls         -0.328400
cost_usd               -0.328054
num_subagents          -0.317936
total_subagent_calls   -0.317936
lead_agent_messages    -0.312771

2. Correlations with Cost:
websearch_calls         0.984573
subagent_messages       0.969603
total_tool_calls        0.968824
time_seconds            0.809682
lead_agent_messages     0.778322
num_turns               0.778322
num_subagents           0.756502
total_subagent_calls    0.756502

3. Subagent Coordination Relationships:

Subagent metrics intercorrelations:
                         num_subagents  subagent_similarity  subagent_success_avg  subagents_completed_pct  total_subagent_calls  subagent_messages  lead_agent_messages
num_subagents                 1.000000            -0.732635             -0.105551                -0.154373              1.000000           0.792135             0.984729
subagent_similarity          -0.732635             1.000000              0.095177                 0.135700             -0.732635          -0.485695            -0.708511
subagent_success_avg         -0.105551             0.095177              1.000000                 0.770253             -0.105551          -0.207672            -0.107778
subagents_completed_pct      -0.154373             0.135700              0.770253                 1.000000             -0.154373          -0.309586            -0.164392
total_subagent_calls          1.000000            -0.732635             -0.105551                -0.154373              1.000000           0.792135             0.984729
subagent_messages             0.792135            -0.485695             -0.207672                -0.309586              0.792135           1.000000             0.800616
lead_agent_messages           0.984729            -0.708511             -0.107778                -0.164392              0.984729           0.800616             1.000000

Heatmap saved to: experiments/visuals/multiagent_correlation_heatmap.png

================================================================================
EXPERIMENT 4: Multi-Agent Operational Modes (PCA & Clustering)
================================================================================

PCA Explained Variance:
  PC1: 63.77%
  PC2: 12.22%
  Total: 75.99%

Top PC1 Loadings:
total_tool_calls       0.312108
subagent_messages      0.310090
cost_usd               0.301613
websearch_calls        0.297851
lead_agent_messages    0.297598
num_turns              0.297598

Top PC2 Loadings:
subagents_completed_pct    0.570759
subagent_success_avg       0.555383
subagent_similarity        0.247805
time_seconds               0.227421
num_subagents              0.213780
total_subagent_calls       0.213780

Operational Mode Distribution:
cluster
0    12
1    62
2    26

Cluster Characteristics:
         num_subagents  subagent_messages  lead_agent_messages  total_subagent_calls  num_turns  total_tool_calls
cluster                                                                                                          
0             2.750000         108.833333             9.916667              2.750000   7.916667         57.166667
1             1.000000          24.580645             6.000000              1.000000   4.000000         13.290323
2             1.653846          55.769231             7.346154              1.653846   5.346154         29.538462

PCA clustering plot saved to: experiments/visuals/multiagent_pca_clustering.png

================================================================================
EXPERIMENT 5: Subagent Coordination Dynamics
================================================================================

Coordination dynamics plot saved to: experiments/visuals/multiagent_coordination_dynamics.png

Coordination Metric Statistics by Accuracy:
         num_subagents                 subagent_similarity                 subagent_success_avg                 subagents_completed_pct                 total_subagent_calls                
                  mean       std count                mean       std count                 mean       std count                    mean       std count                 mean       std count
accuracy                                                                                                                                                                                    
0             1.576923  0.750063    52            0.894995  0.159757    52             4.892308  2.422238    52                0.205128  0.366957    52             1.576923  0.750063    52
1             1.166667  0.429415    48            0.952823  0.131156    48             5.642708  3.272451    48                0.447917  0.486407    48             1.166667  0.429415    48

================================================================================
ALL MULTI-AGENT EXPERIMENTS COMPLETE
Visualizations saved to: experiments/visuals/
=====================================================================