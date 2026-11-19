# Evaluation report: no-class-weights

Out-of-fold predictions from 5-fold cross-validation on 165 rows (seed 13). Every row was predicted by a model that never saw it. Ranges in brackets are 95% bootstrap intervals; with 18 rows in a class, one prediction moves recall by about 5.6 points, so read rare-class numbers as rough.

Model `ai4bharat/IndicBERTv2-MLM-only`, 8 epochs, batch 8, lr 3e-05 (layer decay 0.9), head dropout 0.2, frozen embeddings True, class-weight power 0.0, role masking p 0.5, Neutral gender swap True, confidence weighting False.
Data SHA-256 `a7451ed5d3013919`, code `b279199`, torch 2.15.0+cpu, transformers 5.23.0, training time 99 min.

## Political

Macro-F1 **54.5** (45.1–62.8), balanced accuracy **52.7** (44.6–60.8), accuracy 67.3.

| Class | Rows | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Left | 37 | 58.3 (36.3–79.2) | **37.8** (20.6–53.8) | 45.9 |
| Center | 100 | 72.1 (64.5–79.7) | **88.0** (81.2–94.1) | 79.3 |
| Right | 28 | 47.4 (25.0–72.2) | **32.1** (15.4–51.4) | 38.3 |

Confusion matrix (rows: true label, columns: predicted):

| true \ predicted | Left | Center | Right |
|---|---:|---:|---:|
| Left | 14 | 19 | 4 |
| Center | 6 | 88 | 6 |
| Right | 4 | 15 | 9 |

Against the baselines on the same folds (recall per class, then macro-F1 and balanced accuracy):

| Model | recall Left | recall Center | recall Right | macro-F1 | balanced acc. |
|---|---:|---:|---:|---:|---:|
| Majority class | 0.0 | 100.0 | 0.0 | 25.2 | 33.3 |
| TF-IDF + logistic regression | 18.9 | 86.0 | 46.4 | 51.1 | 50.4 |
| **Fine-tuned model** | 37.8 | 88.0 | 32.1 | 54.5 | 52.7 |

## Gender

Macro-F1 **71.4** (61.1–81.3), balanced accuracy **69.1** (60.5–79.2), accuracy 87.3.

| Class | Rows | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Misogynistic | 18 | 83.3 (64.7–100.0) | **83.3** (64.7–100.0) | 83.3 |
| Misandrist | 18 | 62.5 (25.0–100.0) | **27.8** (7.7–50.0) | 38.5 |
| Neutral | 129 | 89.2 (83.9–94.3) | **96.1** (92.4–99.2) | 92.5 |

Confusion matrix (rows: true label, columns: predicted):

| true \ predicted | Misogynistic | Misandrist | Neutral |
|---|---:|---:|---:|
| Misogynistic | 15 | 0 | 3 |
| Misandrist | 1 | 5 | 12 |
| Neutral | 2 | 3 | 124 |

Against the baselines on the same folds (recall per class, then macro-F1 and balanced accuracy):

| Model | recall Misogynistic | recall Misandrist | recall Neutral | macro-F1 | balanced acc. |
|---|---:|---:|---:|---:|---:|
| Majority class | 0.0 | 0.0 | 100.0 | 29.3 | 33.3 |
| TF-IDF + logistic regression | 38.9 | 55.6 | 96.9 | 70.0 | 63.8 |
| **Fine-tuned model** | 83.3 | 27.8 | 96.1 | 71.4 | 69.1 |

## Shortcut checks

These look for signs that the model learned something other than framing.

**Names replaced by roles.** The held-out rows predicted again with every listed party, leader and organisation name replaced by its role ("an opposition leader"). Large drops mean the model leans on names.

| Axis | Macro-F1 normal | Macro-F1 names masked | Recall change per class (points) |
|---|---:|---:|---:|
| political | 54.5 | 56.1 | Left -5.4, Center +1.0, Right +10.7 |
| gender | 71.4 | 71.4 | Misogynistic +0.0, Misandrist +0.0, Neutral +0.0 |

**Outlet majority.** Accuracy on rows whose label matches the usual label of the *other* rows from the same outlet, and on rows where it does not. A big gap means outlet style is doing the work.

| Axis | Model | label = outlet's usual | label differs |
|---|---:|---:|---:|
| political | model | 80.9 (72/89) | 43.1 (22/51) |
| political | tfidf_logreg | 76.4 (68/89) | 43.1 (22/51) |
| gender | model | 95.1 (116/122) | 54.5 (12/22) |
| gender | tfidf_logreg | 93.4 (114/122) | 68.2 (15/22) |

**Top TF-IDF n-grams per class** (logistic regression fitted on all rows). Names here are cues a model could memorise.

- political / Left: don, as, men, women, freedom, had, our, laws, tata, if, that, ramesh
- political / Center: her, why, from, ai, new, mumbai, every, look, kerala, film, jail, into
- political / Right: congress, opposition, political, politics, modi, opinion, india, protests, security, the modi, electoral, party
- gender / Misogynistic: her, look, bikini, actress, beauty, queen, sunny, sexy, leone, sunny leone, bollywood, mallika
- gender / Misandrist: men, women, men are, male, all men, all, do, are, be, masculinity, because, my
- gender / Neutral: congress, its, his, over, minister, opposition, bjp, india, to the, cm, party, to

## Where the errors are

**By annotator confidence.** Recall on biased rows (Left/Right, Misogynistic/Misandrist), split by the confidence the annotator gave the label. If the model misses mostly low-confidence rows, much of the error is in cases the annotator also found borderline.

| Axis | Annotator confidence | Recall on biased rows | Accuracy on all rows |
|---|---:|---:|---:|
| political | high | 53.8 (14/26) | 81.3 (87/107) |
| political | medium | 24.2 (8/33) | 44.0 (22/50) |
| political | low | 16.7 (1/6) | 25.0 (2/8) |
| gender | high | 69.6 (16/23) | 92.0 (138/150) |
| gender | medium | 44.4 (4/9) | 54.5 (6/11) |
| gender | low | 0.0 (0/4) | 0.0 (0/4) |

**Direction errors.** 8 political row(s) were predicted on the opposite side (Left as Right or Right as Left): art-0071, art-0075, art-0077, art-0132, art-0174, art-0197, art-0289, art-0299. These are worse than calling a biased row Center: the model saw partisan framing but read its direction wrongly.

Every misclassified row is listed by id in `metrics.json` under `diagnostics.errors`. To read them with their text from your local copy of the dataset: `python -m newsbias.errors no-class-weights`.

## Calibration

A single temperature per axis fitted on these out-of-fold logits. The final model applies it, so the confidence the platform shows is closer to the real hit rate. ECE is the gap between confidence and accuracy (lower is better).

| Axis | Temperature | ECE before | ECE after | NLL before | NLL after |
|---|---:|---:|---:|---:|---:|
| political | 1.81 | 14.1 | 6.7 | 0.747 | 0.664 |
| gender | 1.60 | 7.2 | 2.5 | 0.375 | 0.325 |

## By article type

| Axis | Type | Rows | Accuracy | Balanced acc. |
|---|---:|---:|---:|---:|
| political | analysis | 23 | 52.2 | 48.1 |
| political | news | 92 | 69.6 | 42.5 |
| political | opinion | 46 | 67.4 | 64.1 |
| political | wire | 4 | 100.0 | 100.0 |
| gender | analysis | 23 | 82.6 | 33.3 |
| gender | news | 92 | 93.5 | 71.9 |
| gender | opinion | 46 | 78.3 | 75.8 |
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
