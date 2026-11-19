"""Inference used by the API: full article in, labels, calibrated confidences and highlighted phrases out."""
import numpy as np

from . import model as model_io
from .attribution import explain
from .labels import AXES, NEUTRAL
from .text import build_lede, mask_sources, normalize
from .train import predict_logits, softmax


class _Cfg:
    def __init__(self, max_length, eval_batch_size=1):
        self.max_length = max_length
        self.eval_batch_size = eval_batch_size


class Predictor:
    def __init__(self, model_dir, ig_steps=16, top_phrases=5):
        self.model, self.tokenizer, self.meta = model_io.load(model_dir)
        self.cfg = _Cfg(self.meta.get("max_length", 256))
        self.temperatures = self.meta.get("temperatures") or {a: 1.0 for a in AXES}
        self.ig_steps = ig_steps
        self.top_phrases = top_phrases

    @property
    def version(self):
        return self.meta["model_version"]

    def prepare(self, headline, body):
        """The exact text the model reads: normalised headline and the dataset-rule lede, with sources masked."""
        title = normalize(headline)
        lede = normalize(build_lede(body))
        return title, lede, mask_sources(title, title=True), mask_sources(lede)

    def classify(self, headline, body, explain_axes=True):
        title, lede, m_title, m_lede = self.prepare(headline, body)
        if not lede:
            raise ValueError("no usable paragraphs in body")
        logits = predict_logits(self.model, self.tokenizer, [(m_title.text, m_lede.text)], self.cfg)
        n_tokens = len(self.tokenizer(m_title.text, m_lede.text)["input_ids"])
        result = {"headline": title, "lede": lede, "lede_words": len(lede.split()),
                  "truncated": n_tokens > self.cfg.max_length, "axes": {}}
        masked = (m_title, m_lede)
        for axis, classes in AXES.items():
            probs = softmax(logits[axis], self.temperatures.get(axis, 1.0))[0]
            idx = int(np.argmax(probs))
            highlights = []
            # Highlights explain a bias finding; for Center / Neutral there is no finding to explain.
            if explain_axes and classes[idx] != NEUTRAL[axis]:
                for ph in explain(self.model, self.tokenizer, m_title.text, m_lede.text, axis, idx, self.ig_steps,
                                  self.cfg.max_length, self.top_phrases):
                    mt = masked[ph["field"]]
                    start, end = mt.to_original(ph["start"], ph["end"])
                    highlights.append({"field": ("headline", "lede")[ph["field"]], "start": start, "end": end,
                                       "text": mt.original[start:end], "weight": round(float(ph["weight"]), 4)})
            result["axes"][axis] = {"label": classes[idx], "confidence": round(float(probs[idx]), 4),
                                    "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
                                    "highlights": highlights}
        return result
