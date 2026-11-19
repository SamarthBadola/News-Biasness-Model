import torch

from newsbias.attribution import explain
from newsbias.model import BiasClassifier, load_tokenizer
from newsbias.text import mask_sources


def test_highlight_offsets_map_back_through_masking(tiny_model):
    torch.manual_seed(0)
    model = BiasClassifier.from_base(tiny_model).eval()
    tok = load_tokenizer(tiny_model)
    title = mask_sources("Glamorous actress in scandal - The Wire", title=True)
    lede = mask_sources("NDTV said the woman's bedroom secrets revealed, her outfit discussed (PTI).")
    masked = (title, lede)
    found = 0
    for target in range(3):
        # a random head can have no positive evidence for some class, so only require phrases for one of them
        phrases = explain(model, tok, title.text, lede.text, "gender", target, steps=4, max_length=64)
        found += len(phrases)
        assert sum(p["weight"] for p in phrases) <= 1 + 1e-6
        for p in phrases:
            m = masked[p["field"]]
            assert m.text[p["start"]:p["end"]].strip() == m.text[p["start"]:p["end"]]
            s, e = m.to_original(p["start"], p["end"])
            assert 0 <= s < e <= len(m.original)
    assert found
