# Evaluation report: no-shortcut-controls

Out-of-fold predictions from 5-fold cross-validation on 165 rows (seed 13). Every row was predicted by a model that never saw it. Ranges in brackets are 95% bootstrap intervals; with 18 rows in a class, one prediction moves recall by about 5.6 points, so read rare-class numbers as rough.

Model `ai4bharat/IndicBERTv2-MLM-only`, 8 epochs, batch 8, lr 3e-05 (layer decay 0.9), head dropout 0.2, frozen embeddings True, class-weight power 1.0, role masking p 0.0, Neutral gender swap False, confidence weighting False.
Data SHA-256 `a7451ed5d3013919`, code `b279199`, torch 2.15.0+cpu, transformers 5.23.0, training time 35 min.

## Political

Macro-F1 **60.0** (51.2–68.4), balanced accuracy **60.1** (51.5–68.6), accuracy 67.9.

| Class | Rows | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Left | 37 | 50.0 (34.5–65.9) | **51.4** (35.3–67.6) | 50.7 |
| Center | 100 | 79.8 (71.2–88.0) | **79.0** (71.0–86.8) | 79.4 |
| Right | 28 | 50.0 (31.0–69.2) | **50.0** (31.0–66.7) | 50.0 |

Confusion matrix (rows: true label, columns: predicted):

| true \ predicted | Left | Center | Right |
|---|---:|---:|---:|
| Left | 19 | 11 | 7 |
| Center | 14 | 79 | 7 |
| Right | 5 | 9 | 14 |

Against the baselines on the same folds (recall per class, then macro-F1 and balanced accuracy):

| Model | recall Left | recall Center | recall Right | macro-F1 | balanced acc. |
|---|---:|---:|---:|---:|---:|
| Majority class | 0.0 | 100.0 | 0.0 | 25.2 | 33.3 |
| TF-IDF + logistic regression | 18.9 | 86.0 | 46.4 | 51.1 | 50.4 |
| **Fine-tuned model** | 51.4 | 79.0 | 50.0 | 60.0 | 60.1 |

## Gender

Macro-F1 **80.4** (71.0–88.3), balanced accuracy **81.0** (71.0–90.0), accuracy 89.1.

| Class | Rows | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Misogynistic | 18 | 78.9 (58.8–95.2) | **83.3** (64.7–100.0) | 81.1 |
| Misandrist | 18 | 66.7 (43.7–88.9) | **66.7** (44.0–88.9) | 66.7 |
| Neutral | 129 | 93.8 (89.2–97.7) | **93.0** (88.6–97.0) | 93.4 |

Confusion matrix (rows: true label, columns: predicted):

| true \ predicted | Misogynistic | Misandrist | Neutral |
|---|---:|---:|---:|
| Misogynistic | 15 | 0 | 3 |
| Misandrist | 1 | 12 | 5 |
| Neutral | 3 | 6 | 120 |

Against the baselines on the same folds (recall per class, then macro-F1 and balanced accuracy):

| Model | recall Misogynistic | recall Misandrist | recall Neutral | macro-F1 | balanced acc. |
|---|---:|---:|---:|---:|---:|
| Majority class | 0.0 | 0.0 | 100.0 | 29.3 | 33.3 |
| TF-IDF + logistic regression | 38.9 | 55.6 | 96.9 | 70.0 | 63.8 |
| **Fine-tuned model** | 83.3 | 66.7 | 93.0 | 80.4 | 81.0 |

## Shortcut checks

These look for signs that the model learned something other than framing.

**Names replaced by roles.** The held-out rows predicted again with every listed party, leader and organisation name replaced by its role ("an opposition leader"). Large drops mean the model leans on names.

| Axis | Macro-F1 normal | Macro-F1 names masked | Recall change per class (points) |
|---|---:|---:|---:|
| political | 60.0 | 61.5 | Left -2.7, Center -1.0, Right +10.7 |
| gender | 80.4 | 80.4 | Misogynistic +0.0, Misandrist +0.0, Neutral +0.0 |

**Outlet majority.** Accuracy on rows whose label matches the usual label of the *other* rows from the same outlet, and on rows where it does not. A big gap means outlet style is doing the work.

| Axis | Model | label = outlet's usual | label differs |
|---|---:|---:|---:|
| political | model | 79.8 (71/89) | 54.9 (28/51) |
| political | tfidf_logreg | 76.4 (68/89) | 43.1 (22/51) |
| gender | model | 93.4 (114/122) | 77.3 (17/22) |
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
| political | high | 69.2 (18/26) | 80.4 (86/107) |
| political | medium | 39.4 (13/33) | 48.0 (24/50) |
| political | low | 33.3 (2/6) | 25.0 (2/8) |
| gender | high | 82.6 (19/23) | 91.3 (137/150) |
| gender | medium | 66.7 (6/9) | 72.7 (8/11) |
| gender | low | 50.0 (2/4) | 50.0 (2/4) |

**Direction errors.** 12 political row(s) were predicted on the opposite side (Left as Right or Right as Left): art-0029, art-0071, art-0075, art-0077, art-0132, art-0143, art-0174, art-0197, art-0202, art-0249, art-0289, art-0299. These are worse than calling a biased row Center: the model saw partisan framing but read its direction wrongly.

Every misclassified row is listed by id in `metrics.json` under `diagnostics.errors`. To read them with their text from your local copy of the dataset: `python -m newsbias.errors no-shortcut-controls`.

## Calibration

A single temperature per axis fitted on these out-of-fold logits. The final model applies it, so the confidence the platform shows is closer to the real hit rate. ECE is the gap between confidence and accuracy (lower is better).

| Axis | Temperature | ECE before | ECE after | NLL before | NLL after |
|---|---:|---:|---:|---:|---:|
| political | 1.35 | 8.7 | 6.5 | 0.707 | 0.689 |
| gender | 1.33 | 4.4 | 2.5 | 0.331 | 0.314 |

## By article type

| Axis | Type | Rows | Accuracy | Balanced acc. |
|---|---:|---:|---:|---:|
| political | analysis | 23 | 60.9 | 72.1 |
| political | news | 92 | 71.7 | 52.8 |
| political | opinion | 46 | 60.9 | 60.1 |
| political | wire | 4 | 100.0 | 100.0 |
| gender | analysis | 23 | 82.6 | 42.7 |
| gender | news | 92 | 93.5 | 79.7 |
| gender | opinion | 46 | 84.8 | 89.0 |
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
