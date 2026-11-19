# News Bias Model

Fine-tuning, evaluation and serving code for the **political** (Left / Center / Right) and **gender**
(Misogynistic / Misandrist / Neutral) bias classifier of the news bias platform. One IndicBERTv2 encoder
(`ai4bharat/IndicBERTv2-MLM-only`) with two classification heads, trained on the A1 corpus in
[news-bias-dataset](https://github.com/SamarthBadola/news-bias-dataset).

Communal/caste and cultural bias are not here: they are comparative axes handled by few-shot LLM calls in the
platform backend.

This repo holds two things:

1. **The training pipeline**: preprocessing, five-fold cross-validation, the evaluation report and the final fit
   (`newsbias/`, `configs/`).
2. **The API contract** the platform uses to call the model: [`docs/api.md`](docs/api.md),
   [`newsbias/api/schema.py`](newsbias/api/schema.py) and the generated [`api/openapi.json`](api/openapi.json),
   with a reference server in [`newsbias/api/server.py`](newsbias/api/server.py).

## How the pipeline works

```
art_A1.jsonl ──> check rows ──> headline + lede only ──> normalise ──> mask outlet/agency names
 (dataset repo,     (stop on any     (outlet, URL and the       (NFC, single     (+ drop "- The Wire"
  read-only)         bad row)         other two axes dropped)     spaces)          headline suffixes)
                                                                                         │
            near-duplicate stories grouped ◄─────────────────────────────────────────────┘
                         │
         5 folds, balanced on both axes at once, groups kept together
                         │
   for each fold: train on 4/5 ──> predict the held-out 1/5 (also with names role-masked)
                         │
   pooled out-of-fold predictions ──> per-class precision / recall / F1 with bootstrap intervals,
                                      confusion matrices, macro-F1, balanced accuracy, baselines,
                                      shortcut checks, calibration ──> results/<run>/report.md
                         │
   final fit on all 165 rows with the same settings ──> models/<name>/ (served by the API)
```

The out-of-fold numbers are the evaluation. The final model's own training scores are never reported.

### What is done against shortcut learning

With 165 rows the model can easily learn a cue that happens to line up with the labels instead of the framing
the labels describe. These measures are in place, each with a reason:

| Measure | What it stops | Where |
|---|---|---|
| **Source masking**: outlet and agency names become "a news outlet" / "a news agency"; site-name suffixes on headlines are removed. Train and serve. | Labels were made blind to the outlet, so an outlet name in a quote or a "(PTI)" tag must not become the cue. | `newsbias/text.py` |
| **Role masking** of party, leader and organisation names, at random (p = 0.5) each epoch, train only. "Rahul Gandhi" becomes "an opposition leader", "BJP" becomes "the ruling party". | Memorising names instead of framing. The side is kept because the guideline defines Left and Right by framing towards the ruling side and the opposition. | `newsbias/augment.py`, `newsbias/resources/political_entities.toml` |
| **Gender-swapped copies of Neutral rows** (he↔she, men↔women, ...), label kept Neutral. | "Mentions women" → biased. Straight reporting stays straight whichever gender it is about. Biased rows are **not** swapped: the guideline defines Misogynistic and Misandrist differently, so a swapped Misogynistic article is not a Misandrist one. | `newsbias/augment.py` |
| **Dropout** 0.2 before the heads, on top of the encoder's own 0.1. | Co-adapted features on a tiny dataset. | `newsbias/model.py` |
| **Frozen word embeddings** (192M of the 278M parameters). | Memorising rare words. Also roughly halves CPU training time and memory. | `newsbias/model.py` |
| **Layer-wise learning-rate decay** (0.9 per layer) and weight decay 0.01. | Overwriting the pretrained lower layers. | `newsbias/model.py` |
| **Fixed epochs, no early stopping, no tuning on held-out folds.** | Leaking test rows into the estimate. | `configs/base.toml` |

Diagnostics in every report check whether these worked: predictions with all names role-masked, accuracy on rows
whose label differs from their outlet's usual label, the TF-IDF model's top n-grams, and (in the final-report
config) how well the highlighted phrases match the annotator's evidence spans.

### What is done for per-class recall

- **Class-weighted loss** (`class_weight_power = 1`, fully balanced): a mistake on a rare class costs as much,
  in total, as mistakes on the common class. This is the main lever.
- **Mean pooling** instead of [CLS], because IndicBERTv2-MLM-only never trained its [CLS] vector.
- **Balanced folds**, so every test fold has 3–4 rows of each rare gender class.
- Recall is reported per class with a 95% interval, next to a majority baseline and a TF-IDF baseline on the
  same folds, so you can see whether a change helped or is within noise.

### Training input versus what the platform sends

The labels describe the headline and lede (annotation guideline §2a). The platform sends the **full article**;
the model service cuts the lede from it with the dataset's own rule (first two paragraphs after boilerplate and
dateline removal, at most 120 words; `newsbias/text.py::build_lede`) so the model reads exactly the kind of text
it was trained and evaluated on. See [`docs/api.md`](docs/api.md#input-full-article-in-headline--lede-classified).

## Setup (on your laptop)

Python 3.11 or newer. Clone this repo and the dataset repo **side by side** (the config expects
`../news-bias-dataset`):

```
some-folder/
├── News-Biasness-Model/
└── news-bias-dataset/
```

PowerShell:

```powershell
cd News-Biasness-Model
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m pytest
```

Linux/macOS: `source .venv/bin/activate`. On Linux without a GPU, run
`pip install torch --index-url https://download.pytorch.org/whl/cpu` first to skip the CUDA libraries.

`python -m pytest` must pass before you start. The tests train a tiny stand-in model, so they need no download.

## Running it

Run these in order. The first command that touches IndicBERTv2 downloads it (about 1.1 GB) into the Hugging
Face cache.

**1. Check the data and preprocessing (a minute).**

```
python -m newsbias.prepare
```

Prints class counts, near-duplicate pairs, the planned folds, token lengths, and examples of every masking and
swap step. Read the role-masking and gender-swap examples once; if a swap reads wrong, tell me.

**2. Smoke test with the real model (5–10 minutes).**

```
python -m newsbias.cv --run smoke --epochs 1 --limit 40
```

Confirms the download, memory and speed. Note the seconds per epoch in the output. `results/smoke*` is ignored
by git; delete it afterwards.

**3. Main cross-validation run (estimate: 30–60 minutes).**

```
python -m newsbias.cv --config configs/base.toml --run base
```

Stopping it (Ctrl+C) is safe. Run the same command again and it continues from the last finished fold.

**4. Ablations (same time each).** These show what the anti-shortcut measures and the class weights do:

```
python -m newsbias.cv --config configs/no_shortcut_controls.toml --run no-shortcut-controls
python -m newsbias.cv --config configs/no_class_weights.toml --run no-class-weights
```

**5. Numbers for the write-up (about 3× the main run).** Three training seeds and the highlight check:

```
python -m newsbias.cv --config configs/final_report.toml --run final-report
```

**6. Final model (one fold's time, slightly longer).**

```
python -m newsbias.final --config configs/final_report.toml --cv-run final-report --out models/bias-v1
```

Trains on all 165 rows, copies the calibration temperatures from the cross-validation run and saves the
model to `models/bias-v1/`. That folder is ignored by git because it is about 1.1 GB. Share it through
Hugging Face Hub (a private model repo) or a drive link, not GitHub.

**7. Send the results back.** Commit the `results/<run>/` folders (they hold ids, labels, logits and the
report, never article text) on a branch and push:

```
git checkout -b results/base-run
git add results/base results/no-shortcut-controls results/no-class-weights
git commit -m "results: base run and ablations"
git push -u origin results/base-run
```

### Speed tips

- Close other heavy programs; training uses all physical cores.
- `cpu_bf16 = true` in the config uses bfloat16 on the CPU. Ryzen AI 300-series (Zen 5) chips should support it in hardware, and
  it is often 1.5–2× faster. Run it as a separate `--run` name and check the report matches the float32 run.
- `freeze_layers = 4` freezes the lowest four encoder layers as well, which is faster and sometimes no worse
  on small data. It changes the model, so it is a new run, not a speed-up of the same one.

## Reading the report

`results/<run>/report.md`:

- **Per-class recall with its interval** is the number to watch. With 18 rows, an interval of ±20 points is
  normal; two runs whose intervals overlap heavily are not reliably different.
- **Against the baselines.** The fine-tuned model should beat TF-IDF + logistic regression on macro-F1. If it
  does not, the fine-tuning is not adding anything yet.
- **Shortcut checks.** A big drop when names are masked, or a big gap between "label = outlet's usual" and "label
  differs", means the model is leaning on names or outlet style.
- **Calibration.** The temperature is applied by the final model; ECE after should be lower than before.

If rare-class recall is still low after the base run:

1. **More labeled rows of the rare classes.** With 18 examples per class, nothing else comes close.
2. **Read the confusion matrix.** Rare rows called Neutral or Center means the model is underfitting them; try
   `epochs = 12` as a new run. Misogynistic and Misandrist confused with each other is more likely a data or
   definition problem than a training one.
3. Class weighting is already fully balanced. Pushing it further trades a lot of precision for little recall.

Any change is a new run with a new name, compared on the same folds.

## Repository layout

| Path | What it is |
|---|---|
| `newsbias/text.py` | Lede rule (copy of the dataset's), normalisation, source masking with offset maps |
| `newsbias/data.py` | Loading, row checks, near-duplicate grouping |
| `newsbias/augment.py` | Role masking and Neutral gender swaps |
| `newsbias/folds.py` | Grouped five-fold split balanced on both axes |
| `newsbias/model.py` | Encoder + two heads, freezing, layer-wise learning rates, save/load |
| `newsbias/train.py` | Training loop, class weights, prediction |
| `newsbias/metrics.py` | Per-class metrics, bootstrap intervals, calibration |
| `newsbias/baselines.py` | Majority and TF-IDF baselines |
| `newsbias/attribution.py` | Integrated Gradients highlights |
| `newsbias/cv.py`, `report.py`, `final.py`, `prepare.py` | Commands |
| `newsbias/predictor.py`, `newsbias/api/` | Serving: full article in, labels and highlights out |
| `configs/` | Run configurations |
| `results/` | Committed evaluation results, one folder per run |
| `docs/api.md`, `api/openapi.json` | API contract |

## Limits

- One annotator, no inter-annotator agreement. The model can at best reproduce A1's judgement.
- Rare-class proportions were raised by targeted sourcing, so precision on real traffic will be lower than in the
  report.
- No independent test set: cross-validated estimates on 165 rows.
- The labels and the evaluation cover the headline and lede, not whole articles.
