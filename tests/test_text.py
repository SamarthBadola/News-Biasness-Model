from newsbias.text import build_lede, mask_sources, normalize


def test_lede_skips_boilerplate_and_dateline():
    body = "\n".join([
        "By Asha Rao",
        "Updated : Oct 10, 2026",
        "NEW DELHI: The committee met on Monday to discuss the new water policy for the city.",
        "Members asked for a review of the budget before any decision was taken.",
        "A third paragraph that must not appear in the lede at all.",
    ])
    lede = build_lede(body)
    assert lede.startswith("The committee met on Monday")
    assert "Members asked" in lede
    assert "third paragraph" not in lede
    assert "Asha" not in lede


def test_lede_cut_at_sentence_when_long():
    para = " ".join(["This sentence has exactly ten words in it right here."] * 15)
    lede = build_lede(para + "\nSecond paragraph.")
    assert len(lede.split()) <= 120
    assert lede.endswith(".")


def test_agency_dateline_is_stripped():
    assert build_lede("Mumbai, Oct 3 (PTI): The court granted bail to the accused on Friday evening.").startswith(
        "The court granted bail")


def test_normalize_collapses_space_and_controls():
    assert normalize(" a  b\tc\x00d ") == "a b c d"


def test_mask_sources():
    m = mask_sources("NDTV reported that (PTI) sources and Reuters said so; The Hindu reported it too.")
    assert "NDTV" not in m.text and "PTI" not in m.text and "Reuters" not in m.text and "The Hindu" not in m.text
    assert m.text.startswith("a news outlet reported")


def test_mask_sources_leaves_ordinary_words():
    text = "On Republic Day the Hindu community and The Hindu festival crowds gathered with a mint plant."
    assert mask_sources(text).text == text


def test_offsets_map_back_to_original():
    original = "Story by NDTV about the women's march, says ANI today"
    m = mask_sources(original)
    i = m.text.index("women's march")
    s, e = m.to_original(i, i + len("women's march"))
    assert original[s:e] == "women's march"
    j = m.text.index("a news outlet")
    s, e = m.to_original(j, j + len("a news outlet"))
    assert original[s:e] == "NDTV"
    k = m.text.index("today")
    s, e = m.to_original(k, k + 5)
    assert original[s:e] == "today"


def test_title_site_suffix_removed():
    m = mask_sources("Sexist Remarks Miss the Point - The Wire", title=True)
    assert m.text == "Sexist Remarks Miss the Point"
    assert m.to_original(0, len(m.text)) == (0, len(m.text))
    assert mask_sources("Rain - The Wire", title=False).text == "Rain - a news outlet"
