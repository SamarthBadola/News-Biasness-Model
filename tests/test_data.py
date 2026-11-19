import json

import pytest

from newsbias.data import DataError, assign_groups, load_examples
from tests.conftest import make_rows


def test_load_and_mask(data_file):
    ex = load_examples(data_file)
    assert len(ex) == 36
    assert all("PTI" not in e.lede for e in ex)
    assert all(e.masked_sources > 0 for e in ex)


def test_bad_label_stops_the_run(tmp_path):
    rows = make_rows()
    rows[3]["labels"]["gender"]["label"] = "Misogynist"
    rows[4]["lede_status"] = "too_short"
    path = tmp_path / "bad.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(DataError) as e:
        load_examples(path)
    assert "Misogynist" in str(e.value) and "too_short" in str(e.value)


def test_near_duplicates_grouped(data_file):
    ex = load_examples(data_file)
    ex[1].title, ex[1].lede = ex[0].title, ex[0].lede + " extra"
    assign_groups(ex, threshold=0.85)
    assert ex[0].group == ex[1].group
