"""IndicBERTv2 encoder with two classification heads (political, gender) on a shared mean-pooled representation.

Design choices:

* **Mean pooling, not [CLS].** IndicBERTv2-MLM-only was pretrained with masked-language modelling only, so its
  [CLS] vector was never trained to summarise the input. Averaging the token vectors is the more stable choice
  with little data.
* **One encoder, two heads.** The axes share what the encoder learns about framing; each head has its own loss.
* **Dropout before the heads** (``head_dropout``) on top of the encoder's own 0.1 dropout.
* **Frozen word embeddings.** The 250,000-token embedding matrix is two thirds of the 278M parameters. With 165
  rows, updating it mostly memorises rare words; freezing it is a regulariser, and it also removes ~1.5 GB of
  optimiser state and a large share of the per-step CPU work.
"""
import json
import time
from pathlib import Path

import torch
from torch import nn

from .labels import AXES

META_FILE = "model_meta.json"


def _quiet_transformers():
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()


def load_encoder(name_or_path):
    from transformers import AutoModel
    _quiet_transformers()
    try:
        return AutoModel.from_pretrained(name_or_path, add_pooling_layer=False)
    except TypeError:
        return AutoModel.from_pretrained(name_or_path)


def load_tokenizer(name_or_path):
    from transformers import AutoTokenizer
    _quiet_transformers()
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if not tok.is_fast:
        raise RuntimeError(f"{name_or_path} has no fast tokenizer; highlighting needs offset mappings")
    return tok


class BiasClassifier(nn.Module):
    def __init__(self, encoder, head_dropout=0.2):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(head_dropout)
        self.heads = nn.ModuleDict({axis: nn.Linear(hidden, len(classes)) for axis, classes in AXES.items()})
        for head in self.heads.values():
            nn.init.normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)

    @classmethod
    def from_base(cls, name_or_path, head_dropout=0.2):
        return cls(load_encoder(name_or_path), head_dropout)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, inputs_embeds=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                           inputs_embeds=inputs_embeds)
        hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        pooled = self.dropout(pooled)
        return {axis: head(pooled) for axis, head in self.heads.items()}


def encoder_layers(model):
    enc = model.encoder
    return list(enc.encoder.layer) if hasattr(enc, "encoder") and hasattr(enc.encoder, "layer") else []


def freeze(model, embeddings=True, layers=0):
    if embeddings and hasattr(model.encoder, "embeddings"):
        for p in model.encoder.embeddings.parameters():
            p.requires_grad = False
    for layer in encoder_layers(model)[:layers]:
        for p in layer.parameters():
            p.requires_grad = False


def param_groups(model, lr, head_lr, layer_decay, weight_decay):
    """Layer-wise learning-rate decay: the top encoder layer gets ``lr``, each lower layer ``layer_decay`` times
    less, so the general-purpose lower layers move least. No weight decay on biases and LayerNorm weights."""
    groups = []

    def add(params, group_lr):
        decay, no_decay = [], []
        for name, p in params:
            if not p.requires_grad:
                continue
            (no_decay if p.ndim == 1 or "LayerNorm" in name or name.endswith("bias") else decay).append(p)
        if decay:
            groups.append({"params": decay, "lr": group_lr, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "lr": group_lr, "weight_decay": 0.0})

    layers = encoder_layers(model)
    n = len(layers)
    for i, layer in enumerate(layers):
        add(layer.named_parameters(), lr * layer_decay ** (n - 1 - i))
    if hasattr(model.encoder, "embeddings"):
        add(model.encoder.embeddings.named_parameters(), lr * layer_decay ** n)
    covered = {id(p) for g in groups for p in g["params"]}
    rest = [(name, p) for name, p in model.encoder.named_parameters() if id(p) not in covered
            and not name.startswith(("embeddings.", "encoder.layer."))]
    add(rest, lr)
    add(model.heads.named_parameters(), head_lr)
    return groups


def encode(tokenizer, titles, ledes, max_length, offsets=False):
    return tokenizer(list(titles), list(ledes), truncation="longest_first", max_length=max_length, padding=True,
                     return_tensors="pt", return_offsets_mapping=offsets)


def model_inputs(batch):
    keys = ("input_ids", "attention_mask", "token_type_ids")
    return {k: batch[k] for k in keys if k in batch}


def save(model, tokenizer, out_dir, meta):
    out = Path(out_dir)
    (out / "encoder").mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(out / "encoder")
    tokenizer.save_pretrained(out / "encoder")
    torch.save({"heads": model.heads.state_dict(), "head_dropout": model.dropout.p}, out / "heads.pt")
    meta = dict(meta, labels={a: list(c) for a, c in AXES.items()}, saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (out / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load(model_dir):
    d = Path(model_dir)
    meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
    for axis, classes in AXES.items():
        if tuple(meta["labels"][axis]) != classes:
            raise RuntimeError(f"{d}: label order for {axis} differs from this code: {meta['labels'][axis]}")
    state = torch.load(d / "heads.pt", map_location="cpu", weights_only=True)
    model = BiasClassifier(load_encoder(d / "encoder"), state["head_dropout"])
    model.heads.load_state_dict(state["heads"])
    model.eval()
    return model, load_tokenizer(d / "encoder"), meta
