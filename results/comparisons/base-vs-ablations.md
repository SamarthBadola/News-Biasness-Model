# Run comparison: base, no-shortcut-controls, no-class-weights

Reference: **base**. Same rows and folds in every run; differences are paired (see `newsbias/compare.py`). An interval that contains 0 means the data cannot tell the runs apart.

## Political

| Run | recall Left | recall Center | recall Right | precision Left | precision Center | precision Right | macro-F1 | balanced acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 37.8 | 85.0 | 46.4 | 51.9 | 74.6 | 54.2 | 57.7 | 56.4 |
| no-shortcut-controls | 51.4 | 79.0 | 50.0 | 50.0 | 79.8 | 50.0 | 60.0 | 60.1 |
| no-class-weights | 37.8 | 88.0 | 32.1 | 58.3 | 72.1 | 47.4 | 54.5 | 52.7 |

**no-shortcut-controls minus base** (percentage points, 95% paired interval, share of resamples above 0):

| Metric | Difference | Interval | P(> 0) |
|---|---:|---:|---:|
| macro_f1 | +2.3 | -4.3 to +9.4 | 74.0% |
| balanced_accuracy | +3.7 | -3.2 to +10.9 | 84.0% |
| recall Left | +13.5 | +0.0 to +28.2 | 96.7% |
| recall Center | -6.0 | -10.8 to -1.9 | 0.0% |
| recall Right | +3.6 | -11.8 to +19.2 | 58.6% |

Rows only base got right: 9; only no-shortcut-controls: 9 (McNemar p = 1.00). Per gold class (base-only → no-shortcut-controls-only): Left 1→6, Center 6→0, Right 2→3.

**no-class-weights minus base** (percentage points, 95% paired interval, share of resamples above 0):

| Metric | Difference | Interval | P(> 0) |
|---|---:|---:|---:|
| macro_f1 | -3.2 | -8.7 to +1.8 | 10.2% |
| balanced_accuracy | -3.8 | -9.0 to +0.9 | 6.5% |
| recall Left | +0.0 | -8.1 to +7.5 | 34.0% |
| recall Center | +3.0 | +0.0 to +6.9 | 95.0% |
| recall Right | -14.3 | -28.0 to -3.3 | 0.0% |

Rows only base got right: 5; only no-class-weights: 4 (McNemar p = 1.00). Per gold class (base-only → no-class-weights-only): Left 1→1, Center 0→3, Right 4→0.

## Gender

| Run | recall Misogynistic | recall Misandrist | recall Neutral | precision Misogynistic | precision Misandrist | precision Neutral | macro-F1 | balanced acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 83.3 | 44.4 | 95.3 | 78.9 | 72.7 | 91.1 | 76.5 | 74.4 |
| no-shortcut-controls | 83.3 | 66.7 | 93.0 | 78.9 | 66.7 | 93.8 | 80.4 | 81.0 |
| no-class-weights | 83.3 | 27.8 | 96.1 | 83.3 | 62.5 | 89.2 | 71.4 | 69.1 |

**no-shortcut-controls minus base** (percentage points, 95% paired interval, share of resamples above 0):

| Metric | Difference | Interval | P(> 0) |
|---|---:|---:|---:|
| macro_f1 | +3.9 | -2.1 to +10.9 | 88.9% |
| balanced_accuracy | +6.6 | +0.7 to +13.6 | 98.0% |
| recall Misogynistic | +0.0 | +0.0 to +0.0 | 0.0% |
| recall Misandrist | +22.2 | +5.0 to +42.9 | 98.2% |
| recall Neutral | -2.3 | -5.3 to +0.0 | 0.0% |

Rows only base got right: 3; only no-shortcut-controls: 4 (McNemar p = 1.00). Per gold class (base-only → no-shortcut-controls-only): Misogynistic 0→0, Misandrist 0→4, Neutral 3→0.

**no-class-weights minus base** (percentage points, 95% paired interval, share of resamples above 0):

| Metric | Difference | Interval | P(> 0) |
|---|---:|---:|---:|
| macro_f1 | -5.0 | -12.2 to +0.8 | 4.2% |
| balanced_accuracy | -5.3 | -11.9 to +0.2 | 3.0% |
| recall Misogynistic | +0.0 | +0.0 to +0.0 | 0.0% |
| recall Misandrist | -16.7 | -36.4 to +0.0 | 0.0% |
| recall Neutral | +0.8 | +0.0 to +2.9 | 62.6% |

Rows only base got right: 3; only no-class-weights: 1 (McNemar p = 0.62). Per gold class (base-only → no-class-weights-only): Misogynistic 0→0, Misandrist 3→0, Neutral 0→1.

