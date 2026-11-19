"""Fixtures: a synthetic corpus in the dataset's schema and a tiny BERT that stands in for IndicBERTv2.

The tiny model has the same architecture family and tokenizer layout (cased WordPiece, [CLS] A [SEP] B [SEP])
as ai4bharat/IndicBERTv2-MLM-only, so the whole pipeline runs end to end in seconds without a download.
"""
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

POL = {
    "Left": ["Opposition slams government over failed scheme", "Critics say the government betrayed farmers"],
    "Right": ["Government's bold reform wins praise", "Historic step by the government, says Modi"],
    "Center": ["Committee meets to review water project", "Officials said the meeting was held on Monday"],
}
GEN = {
    "Misogynistic": ["glamorous actress in scandal, her outfit", "the woman's bedroom secrets revealed"],
    "Misandrist": ["men are all the same, toxic men again", "fathers are useless, say men haters"],
    "Neutral": ["he said the report was filed", "the court heard the case on Tuesday"],
}
PLAN = ([("Center", "Neutral")] * 14 + [("Left", "Neutral")] * 6 + [("Right", "Neutral")] * 6 +
        [("Center", "Misogynistic")] * 5 + [("Center", "Misandrist")] * 3 + [("Left", "Misandrist")] * 2)
OUTLETS = ["NDTV", "OpIndia", "The Wire", "Times of India", "Scroll"]


def make_rows(seed=0):
    rng = random.Random(seed)
    rows = []
    for n, (pol, gen) in enumerate(PLAN, 1):
        title = f"{rng.choice(POL[pol])} story {n}"
        lede = (f"NEW DELHI: {rng.choice(POL[pol])}. {rng.choice(GEN[gen])} in case {n}. "
                f"The report from PTI added details number {n * 7}.")
        rows.append({
            "id": f"art-{n:04d}", "url": f"https://example.com/{n}", "outlet": OUTLETS[n % len(OUTLETS)],
            "article_type": rng.choice(["news", "opinion", "analysis"]), "event_id": "", "title": title,
            "standfirst": "", "lede": lede, "lede_status": "ok", "pilot": n <= 3, "split": "eval",
            "labels": {
                "political": {"label": pol, "confidence": "high", "rationale": "r",
                              "spans": [{"field": "title", "start": 0, "end": 6, "text": title.split()[0]}]},
                "gender": {"label": gen, "confidence": rng.choice(["high", "medium"]), "rationale": "r",
                           "spans": [] if gen == "Neutral" else [{"field": "lede", "start": 0, "end": 3,
                                                                  "text": GEN[gen][0].split()[0]}]},
                "communal_caste": {"status": "not_applicable"}, "cultural": {"status": "not_applicable"},
            },
        })
    return rows


@pytest.fixture
def data_file(tmp_path):
    path = tmp_path / "art_A1.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in make_rows()) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    out = tmp_path_factory.mktemp("tiny-bert")
    corpus = [r["title"] + " " + r["lede"] for r in make_rows()] + \
             [" ".join(v) for d in (POL, GEN) for v in d.values()] + ["a news outlet an opposition leader agency"]
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    specials = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    tok.train_from_iterator(corpus, trainers.WordPieceTrainer(vocab_size=400, special_tokens=specials))
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]",
                                   pad_token="[PAD]", mask_token="[MASK]")
    fast.save_pretrained(out)
    cfg = BertConfig(vocab_size=tok.get_vocab_size(), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=128)
    BertModel(cfg).save_pretrained(out)
    return str(out)


@pytest.fixture
def tiny_config(tmp_path, data_file, tiny_model):
    path = tmp_path / "tiny.toml"
    path.write_text(f'''data_path = "{data_file.as_posix()}"
model_name = "{Path(tiny_model).as_posix()}"
max_length = 96
epochs = 2
batch_size = 8
head_lr = 1e-2
lr = 1e-3
folds = 3
bootstrap = 50
explain_eval = true
ig_steps = 4
threads = 1
''', encoding="utf-8")
    return path
