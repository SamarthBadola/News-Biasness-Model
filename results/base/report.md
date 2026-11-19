# Evaluation report: base

Out-of-fold predictions from 5-fold cross-validation on 165 rows (seed 13). Every row was predicted by a model that never saw it. Ranges in brackets are 95% bootstrap intervals; with 18 rows in a class, one prediction moves recall by about 5.6 points, so read rare-class numbers as rough.

Model `ai4bharat/IndicBERTv2-MLM-only`, 8 epochs, batch 8, lr 3e-05 (layer decay 0.9), head dropout 0.2, frozen embeddings True, class-weight power 1.0, role masking p 0.5, Neutral gender swap True, confidence weighting False.
Data SHA-256 `620cd785ca7bc57a`, code `ac6556a`, torch 2.15.0+cpu, transformers 5.23.0, training time 48 min.

## Political

Macro-F1 **57.7** (48.8–66.7), balanced accuracy **56.4** (48.3–65.1), accuracy 67.9.

| Class | Rows | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Left | 37 | 51.9 (32.1–70.6) | **37.8** (21.9–54.1) | 43.8 |
| Center | 100 | 74.6 (66.7–82.8) | **85.0** (78.0–91.7) | 79.4 |
| Right | 28 | 54.2 (33.3–75.0) | **46.4** (28.6–65.8) | 50.0 |

Confusion matrix (rows: true label, columns: predicted):

| true \ predicted | Left | Center | Right |
|---|---:|---:|---:|
| Left | 14 | 18 | 5 |
| Center | 9 | 85 | 6 |
| Right | 4 | 11 | 13 |

Against the baselines on the same folds (recall per class, then macro-F1 and balanced accuracy):

| Model | recall Left | recall Center | recall Right | macro-F1 | balanced acc. |
|---|---:|---:|---:|---:|---:|
| Majority class | 0.0 | 100.0 | 0.0 | 25.2 | 33.3 |
| TF-IDF + logistic regression | 18.9 | 86.0 | 46.4 | 51.1 | 50.4 |
| **Fine-tuned model** | 37.8 | 85.0 | 46.4 | 57.7 | 56.4 |

## Gender

Macro-F1 **76.5** (66.1–86.0), balanced accuracy **74.4** (64.8–84.8), accuracy 88.5.

| Class | Rows | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Misogynistic | 18 | 78.9 (58.8–95.2) | **83.3** (64.7–100.0) | 81.1 |
| Misandrist | 18 | 72.7 (41.6–100.0) | **44.4** (21.0–69.2) | 55.2 |
| Neutral | 129 | 91.1 (86.1–95.9) | **95.3** (91.5–98.5) | 93.2 |

Confusion matrix (rows: true label, columns: predicted):

| true \ predicted | Misogynistic | Misandrist | Neutral |
|---|---:|---:|---:|
| Misogynistic | 15 | 0 | 3 |
| Misandrist | 1 | 8 | 9 |
| Neutral | 3 | 3 | 123 |

Against the baselines on the same folds (recall per class, then macro-F1 and balanced accuracy):

| Model | recall Misogynistic | recall Misandrist | recall Neutral | macro-F1 | balanced acc. |
|---|---:|---:|---:|---:|---:|
| Majority class | 0.0 | 0.0 | 100.0 | 29.3 | 33.3 |
| TF-IDF + logistic regression | 38.9 | 55.6 | 96.9 | 70.0 | 63.8 |
| **Fine-tuned model** | 83.3 | 44.4 | 95.3 | 76.5 | 74.4 |

## Shortcut checks

These look for signs that the model learned something other than framing.

**Names replaced by roles.** The held-out rows predicted again with every listed party, leader and organisation name replaced by its role ("an opposition leader"). Large drops mean the model leans on names.

| Axis | Macro-F1 normal | Macro-F1 names masked | Recall change per class (points) |
|---|---:|---:|---:|
| political | 57.7 | 58.4 | Left -2.7, Center +0.0, Right +7.1 |
| gender | 76.5 | 76.5 | Misogynistic +0.0, Misandrist +0.0, Neutral +0.0 |

**Outlet majority.** Accuracy on rows whose label matches the usual label of the *other* rows from the same outlet, and on rows where it does not. A big gap means outlet style is doing the work.

| Axis | Model | label = outlet's usual | label differs |
|---|---:|---:|---:|
| political | model | 80.9 (72/89) | 49.0 (25/51) |
| political | tfidf_logreg | 76.4 (68/89) | 43.1 (22/51) |
| gender | model | 94.3 (115/122) | 63.6 (14/22) |
| gender | tfidf_logreg | 93.4 (114/122) | 68.2 (15/22) |

**Top TF-IDF n-grams per class** (logistic regression fitted on all rows). Names here are cues a model could memorise.

- political / Left: don, as, men, women, freedom, had, our, laws, tata, if, that, ramesh
- political / Center: her, why, from, ai, new, mumbai, every, look, kerala, film, jail, into
- political / Right: congress, opposition, political, politics, modi, opinion, india, protests, security, the modi, electoral, party
- gender / Misogynistic: her, look, bikini, actress, beauty, queen, sunny, sexy, leone, sunny leone, bollywood, mallika
- gender / Misandrist: men, women, men are, male, all men, all, do, are, be, masculinity, because, my
- gender / Neutral: congress, its, his, over, minister, opposition, bjp, india, to the, cm, party, to

## Calibration

A single temperature per axis fitted on these out-of-fold logits. The final model applies it, so the confidence the platform shows is closer to the real hit rate. ECE is the gap between confidence and accuracy (lower is better).

| Axis | Temperature | ECE before | ECE after | NLL before | NLL after |
|---|---:|---:|---:|---:|---:|
| political | 1.74 | 13.2 | 4.9 | 0.749 | 0.676 |
| gender | 1.55 | 7.9 | 4.3 | 0.362 | 0.317 |

## By article type

| Axis | Type | Rows | Accuracy | Balanced acc. |
|---|---:|---:|---:|---:|
| political | analysis | 23 | 56.5 | 52.2 |
| political | news | 92 | 72.8 | 48.8 |
| political | opinion | 46 | 60.9 | 58.8 |
| political | wire | 4 | 100.0 | 100.0 |
| gender | analysis | 23 | 82.6 | 33.3 |
| gender | news | 92 | 93.5 | 71.9 |
| gender | opinion | 46 | 82.6 | 84.0 |
| gender | wire | 4 | 75.0 | 50.0 |

## Folds

0 near-duplicate pair(s) (TF-IDF cosine ≥ 0.5) were kept in the same fold.

| Fold | rows | political:Left | political:Center | political:Right | gender:Misogynistic | gender:Misandrist | gender:Neutral |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 32 | 7 | 20 | 5 | 4 | 3 | 25 |
| 1 | 34 | 8 | 20 | 6 | 3 | 4 | 27 |
| 2 | 33 | 7 | 20 | 6 | 4 | 3 | 26 |
| 3 | 33 | 7 | 20 | 6 | 3 | 4 | 26 |
| 4 | 33 | 8 | 20 | 5 | 4 | 4 | 25 |

## Limits

- One annotator, so no inter-annotator agreement; the model can at best learn that annotator's judgement.
- Gender-focused articles were sourced on purpose, so class proportions are not real-world rates and precision on the platform will be lower than here.
- No independent test set. These are cross-validated estimates on 165 rows.
- The labels describe the headline and lede only, and the model is evaluated on that input only.
