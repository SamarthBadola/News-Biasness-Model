# Classifier API (version 1)

The contract between the platform backend and the model service, for the political and gender axes. The
machine-readable version is [`api/openapi.json`](../api/openapi.json), generated from
[`newsbias/api/schema.py`](../newsbias/api/schema.py) by `python scripts/export_openapi.py`; CI fails if the two
drift apart.

This replaces `classify(headline, lede)` in section 7 of the Web Application Design: the backend sends the full
article body instead of a lede it cut itself. Everything else in that section stands (HTTPS, API key in Render's
environment variables, a timeout and one retry, degrade to the LLM-scored axes if the service is down).

## Input: full article in, headline + lede classified

The backend sends the headline and the **full body** the page fetcher extracted, one paragraph per line. The
service derives the lede itself with the dataset's rule: the first two body paragraphs after removing bylines,
datelines and boilerplate, cut to 120 words at the last full sentence if that keeps at least 60 words.

Why the model does not read the whole article:

- **The labels describe the headline and lede only** (annotation guideline §2a: "It describes the headline and
  lede, not the whole article, and the platform's output says 'headline and opening'"). A model fed whole
  articles would be producing a judgement nobody labeled and nothing evaluated.
- **The evaluation covers only that input.** There is no measurement of how the model behaves on paragraph 9 of
  an opinion piece; text it never saw in training shifts its predictions in ways the report cannot show.
- **The encoder reads at most 512 tokens.** Whole articles would need chunking and a rule for combining chunk
  predictions, which also needs labeled whole-article data to evaluate.

Sending the full body keeps the contract stable if that changes later: whole-article classification can be added
behind the same request once there is labeled data to evaluate it, with `input.scope` in the response saying
which was used. Doing the cut in the service also guarantees the platform and the training data use the same
rule.

## Authentication

Every `/v1/*` call needs the header `X-API-Key: <key>`. The key lives in the backend's Render environment
variables and the service's own environment (`NEWSBIAS_API_KEY`). The service refuses to start without one.
Use HTTPS only.

## `POST /v1/classify`

Request:

```json
{
  "headline": "After delimitation bill fails in Lok Sabha, Modi now using women card in poll-bound Bengal",
  "body": "By Staff Reporter\nNEW DELHI: Prime Minister Narendra Modi on Sunday ...\nThe Opposition said ...\n...",
  "explain": true,
  "request_id": "9f1c2e"
}
```

| Field | Type | Notes |
|---|---|---|
| `headline` | string, 1–1,000 chars | As published. |
| `body` | string, 1–200,000 chars | Full body, paragraphs separated by newlines. Boilerplate may be left in. |
| `explain` | bool, default true | Compute highlights for biased labels. False saves several seconds per call. |
| `request_id` | string, optional | Echoed back for logs. |

Unknown fields are rejected (422).

Response (200):

```json
{
  "api_version": "1",
  "model_version": "indicbertv2-bias-20261201-a7451e3c9d0f",
  "request_id": "9f1c2e",
  "input": {
    "headline": "After delimitation bill fails in Lok Sabha, Modi now using women card in poll-bound Bengal",
    "lede": "Prime Minister Narendra Modi on Sunday ... The Opposition said ...",
    "lede_words": 96,
    "truncated": false,
    "scope": "headline+lede"
  },
  "political": {
    "label": "Left",
    "confidence": 0.71,
    "probabilities": {"Left": 0.71, "Center": 0.24, "Right": 0.05},
    "highlights": [
      {"field": "headline", "start": 59, "end": 69, "text": "women card", "weight": 0.42}
    ]
  },
  "gender": {
    "label": "Neutral",
    "confidence": 0.88,
    "probabilities": {"Misogynistic": 0.07, "Misandrist": 0.05, "Neutral": 0.88},
    "highlights": []
  }
}
```

- `confidence` is the calibrated probability of the label (temperature scaling fitted on cross-validation
  predictions). With 165 training rows, treat it as rough.
- `highlights` are the phrases that pushed the model towards the label, strongest first, at most five. `start` and
  `end` are character offsets into `input.headline` or `input.lede` (whichever `field` names), and `text` equals
  that slice. `weight` is the phrase's share of all the evidence for the label (0–1), from Integrated Gradients.
  They are empty for Center and Neutral, because there is no bias finding to explain.
- The labels describe the headline and lede, so the platform should say "headline and opening", as the guideline
  requires.

### How the explanation step should use this

The highlights are evidence, not an explanation. The backend passes the label, confidence, `input.headline`,
`input.lede` and the highlighted phrases to Gemini, which writes the plain-language rationale. Keep the fetched
text in the data part of the structured prompt, never in the instructions (threat model M-002). If
`highlights` is empty for a biased label (rare; `explain` false), the rationale should say the model gave no
specific phrases rather than inventing some.

### Errors

| Status | When |
|---|---|
| 401 | Missing or wrong `X-API-Key` |
| 422 | Invalid request, or no usable paragraph in `body` |
| 503 | Model not loaded (see `/health`) |

On 503 or a timeout, the backend retries once, then returns the LLM-scored axes and marks political and gender as
unavailable (design §7).

## `GET /health`

No key. `{"status": "ok" | "loading" | "error", "model_version": "...", "api_version": "1"}`. Use it for the
host's health check.

## Running the reference server

```
NEWSBIAS_MODEL_DIR=models/bias-v1 NEWSBIAS_API_KEY=<long random string> uvicorn newsbias.api.server:app --port 8000
```

`NEWSBIAS_IG_STEPS` (default 16) sets the Integrated Gradients steps; 8 halves highlight time with slightly
noisier weights. Requests are served one at a time.

## Hosting requirements (open decision 3)

Measured with a model of IndicBERTv2's size on 2 CPU threads:

| | |
|---|---|
| Peak memory | about 2.9 GB |
| Labels only | about 0.15 s |
| Labels + highlights for one biased axis | about 4 s; both axes biased about 8 s |
| Model files | about 1.1 GB |

**Render's free web service (512 MB RAM) cannot hold this model.** A host needs at least 4 GB of RAM. One free
option that fits is a Hugging Face Space on the free CPU tier (16 GB RAM, 2 vCPU), which also sleeps when idle, so
the first request after a quiet period is slow (the same cold-start pattern as Render). The backend's timeout for
this call should allow for that, and for highlights.
