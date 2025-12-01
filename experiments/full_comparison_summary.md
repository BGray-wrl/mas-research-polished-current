# Full Dataset Comparison: Single-Agent vs Multi-Agent

**Dataset:** 200 WebWalker runs (100 single, 100 multi)

## Performance Summary

| Metric | Single-Agent | Multi-Agent | Difference |
|--------|--------------|-------------|------------|
| Accuracy | 50.0% (50/100) | 48.0% (48/100) | -2.0pp |
| Cost ($/q) | $0.243 | $0.473 | 1.94x |
| Time (s) | 88.7 | 127.8 | 1.44x |
| Turns | 34.4 | 4.8 | 0.14x |
| Tool Calls | 12.9 | 22.8 | 1.76x |
| Messages | 36.4 | 49.6 | 1.36x |
| WebSearch | 8.5 | 14.0 | 1.65x |
| Correctness | 6.82/10 | 6.58/10 | -0.24 |
| Cost/Correct | $0.49 | $0.98 | 2.02x |

## Multi-Agent Coordination Metrics

- **Num Subagents:** 1.38 ± 0.65
- **Subagent Similarity:** 0.923 ± 0.149
- **Subagent Success Rate:** 5.3% ± 2.9%
- **Subagent Completion Rate:** 0.3% ± 0.4%
- **Lead Agent Messages:** 6.8 ± 1.4
- **Subagent Messages:** 42.8 ± 30.5
