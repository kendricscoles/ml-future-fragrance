# Fairness Analysis

## Key Metrics
| Metric | Value |
|--------|-------|
| Demographic Parity Gap | 0.239 |
| Equalized Odds Gap | 0.385 |
| TPR Gap | 0.385 |
| FPR Gap | 0.204 |

## By Group
| Group | N | Sel Rate | TPR | FPR | PPV |
|-------|---|----------|-----|-----|-----|
| 55+ | 19 | 0.000 | 0.000 | 0.000 | 0.000 |
| 25-34 | 67 | 0.239 | 0.385 | 0.204 | 0.312 |
| 18-24 | 36 | 0.083 | 0.000 | 0.115 | 0.000 |
| 35-44 | 46 | 0.022 | 0.000 | 0.027 | 0.000 |
| 45-54 | 32 | 0.000 | 0.000 | 0.000 | 0.000 |

## Notes
- Highest selection: 25-34, Lowest: 45-54
- Notable differences in error rates.
- Differences in selection rates.
- Threshold: top 10%, Protected: age_group
