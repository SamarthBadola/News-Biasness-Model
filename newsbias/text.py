"""Text rules shared by training and serving.

Two things live here so that the model sees exactly the same kind of input in production as in training:

* ``build_lede`` is a copy of the lede rule in news-bias-dataset/scripts/collect.py (annotation guideline
  section 2a): the first two body paragraphs after boilerplate and dateline removal, cut to 120 words at the
  last full sentence if that keeps at least 60 words, otherwise at 120 words. The labels describe the headline
  and this lede, not the whole article, so the API derives the lede from the full article with this function.
* ``mask_sources`` replaces outlet and news-agency names with neutral words. Annotators labeled with the outlet
  hidden, so an outlet name that slips into the text (a quote, an "(PTI)" tag) must not become a cue.

``MaskedText`` keeps an offset map so that highlighted phrases found in the masked text can be reported at their
position in the text the caller sent.
"""
import re
import unicodedata
from dataclasses import dataclass, field

# --- lede rule (keep in step with news-bias-dataset/scripts/collect.py) -----------------------------------------

LEDE_WORDS, LEDE_MIN_KEEP, LEDE_PARAS = 120, 60, 2

BOILER_LINE = re.compile(
    r"^(updated\s*:|published\s*:|also read|read more|advertisement|listen to this|subscribe|follow us|"
    r"disclaimer|this (report|story|article) (is|has been) (auto|automatically)|"
    r"\(this story|\(with inputs|photo\s*:|image\s*:|\(file photo)", re.I)
BYLINE = re.compile(r"^(by|written by|edited by)\s", re.I)
AGENCY_TAG = r"\((?:PTI|ANI|IANS|Reuters|AFP|UNI)\)"
CITIES = ("New Delhi|Delhi|Mumbai|Kolkata|Chennai|Bengaluru|Bangalore|Hyderabad|Lucknow|Patna|Chandigarh|"
          "Jaipur|Bhopal|Guwahati|Srinagar|Thiruvananthapuram")
DATE_PART = r"(?:,\s*[A-Za-z]{3,9}\.?\s*\d{1,2})?"
DATELINES = [
    re.compile(rf"^\s*[A-Za-z. ]{{2,30}}{DATE_PART}\s*{AGENCY_TAG}\s*(?::|—|–|-)?\s*"),
    re.compile(rf"^\s*[A-Z][A-Z .]{{2,30}}{DATE_PART}\s*(?::|—|–)\s+"),
    re.compile(rf"^\s*(?:{CITIES}){DATE_PART}\s*(?::|—|–)\s+"),
]


def cap_words(text, limit, min_keep):
    words = text.split()
    if len(words) <= limit:
        return " ".join(words)
    cut = " ".join(words[:limit])
    if cut.endswith((".", "?", "!")):
        return cut
    last = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if last != -1 and len(cut[:last + 1].split()) >= min_keep:
        return cut[:last + 1]
    return cut


def strip_dateline(paragraph):
    for pattern in DATELINES:
        m = pattern.match(paragraph)
        if m and len(paragraph[m.end():].split()) >= 4:
            return paragraph[m.end():].strip()
    return paragraph


def is_boilerplate(paragraph):
    words = paragraph.split()
    if len(words) < 5 or BOILER_LINE.match(paragraph):
        return True
    return bool(BYLINE.match(paragraph)) and len(words) <= 12


def build_lede(text):
    """Lede from a full article body whose paragraphs are separated by newlines."""
    kept = []
    for para in (p.strip() for p in text.split("\n")):
        if not para or is_boilerplate(para):
            continue
        kept.append(strip_dateline(para) if not kept else para)
        if len(kept) == LEDE_PARAS:
            break
    return cap_words(" ".join(kept), LEDE_WORDS, LEDE_MIN_KEEP)


# --- normalisation ----------------------------------------------------------------------------------------------

_SPACE = re.compile(r"\s+")


def normalize(text):
    """NFC, no control characters, single spaces. Applied to headline and lede before anything else."""
    text = unicodedata.normalize("NFC", text or "")
    text = "".join(ch if unicodedata.category(ch)[0] != "C" else " " for ch in text)
    return _SPACE.sub(" ", text).strip()


# --- masking with an offset map ---------------------------------------------------------------------------------

@dataclass
class MaskedText:
    """Text after replacements, with enough bookkeeping to map offsets back to the original."""
    original: str
    text: str
    # (orig_start, orig_end, new_start, new_end) for every replacement, in order
    edits: list = field(default_factory=list)

    def to_original(self, start, end):
        """Map a [start, end) span of ``text`` back to ``original``. A span touching a replacement widens to
        cover the whole replaced original string."""
        return self._map(start, side="start"), self._map(end, side="end")

    def _map(self, pos, side):
        shift = 0
        for o_s, o_e, n_s, n_e in self.edits:
            if pos <= n_s:
                break
            if pos < n_e or (side == "end" and pos == n_e):
                return o_s if side == "start" else o_e
            shift = o_e - n_e
        return pos + shift


def apply_replacements(text, patterns, keep=None):
    """Apply ``(compiled_regex, replacement)`` pairs left to right in a single pass over ``text``.

    Matches never overlap: at each position the earliest-starting, then longest, match wins. ``keep``, if given,
    is called once per surviving match and can veto it (used for random masking during training).
    """
    found = []
    for rx, repl in patterns:
        for m in rx.finditer(text):
            found.append((m.start(), -(m.end() - m.start()), m.end(), repl))
    found.sort()
    out, edits, cursor, new_len = [], [], 0, 0
    for start, _, end, repl in found:
        if start < cursor:
            continue
        if keep is not None and not keep():
            cursor_skip = end
            out.append(text[cursor:cursor_skip])
            new_len += cursor_skip - cursor
            cursor = cursor_skip
            continue
        out.append(text[cursor:start])
        new_len += start - cursor
        out.append(repl)
        edits.append((start, end, new_len, new_len + len(repl)))
        new_len += len(repl)
        cursor = end
    out.append(text[cursor:])
    return MaskedText(original=text, text="".join(out), edits=edits)


# Outlets seen in the dataset or listed in its outlets.txt, plus other large Indian outlets. Names that are also
# ordinary words ("Republic", "Scroll", "Mint", "Outlook", "Organiser", "The Hindu", ...) are masked only when they
# appear in an attribution-like position, so "Republic Day" or "the Hindu community" are left alone.
OUTLETS = (
    "NDTV Profit", "NDTV", "ThePrint", "The Print", "Indian Express", "The New Indian Express", "Times of India",
    "TOI", "Hindustan Times", "India Today", "News18", "Republic TV", "Republic World", "Scroll.in", "The Wire",
    "OpIndia", "Firstpost", "Oneindia", "TFIPOST", "TFI Post", "Free Press Journal", "The Quint",
    "Times Now", "India TV", "DNA India", "National Herald", "ABP News", "Business Standard", "Deutsche Welle",
    "The Diplomat", "Moneycontrol", "LiveLaw", "Feminism in India", "Telegraph India", "The Telegraph",
    "The Tribune", "Deccan Herald", "Swarajya", "Newslaundry", "The Statesman", "Zee News",
    "Aaj Tak", "Economic Times", "Livemint", "Organiser Weekly", "The Mooknayak", "Alt News", "BOOM Live", "Deccan Chronicle",
)
AMBIGUOUS_OUTLETS = ("The Hindu", "Republic", "Scroll", "Mint", "The Week", "Frontline", "Outlook", "Organiser")
AGENCIES = ("PTI", "ANI", "IANS", "Reuters", "AFP", "UNI", "Bloomberg", "Press Trust of India",
            "Asian News International", "Associated Press")

_ATTRIBUTION = r"(?=\s*(?:[,.;:)'’]|$|\s+(?:reported|reports|said|says|newspaper|daily|told|has|had|report)\b))"


def _names(names):
    return "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))


SOURCE_PATTERNS = [
    (re.compile(rf"\(\s*(?:{_names(AGENCIES)})\s*\)"), "(agency)"),
    (re.compile(rf"(?<![\w.]){_names(AGENCIES)}(?![\w])"), "a news agency"),
    (re.compile(rf"(?<![\w.])(?:{_names(OUTLETS)})(?![\w])"), "a news outlet"),
    (re.compile(rf"(?<![\w.])(?:{_names(AMBIGUOUS_OUTLETS)}){_ATTRIBUTION}"), "a news outlet"),
]


# "Headline - The Wire" / "Headline | NDTV": a site-name suffix is removed outright, because even masked to
# "- a news outlet" it would remain a style cue that only some outlets' headlines carry.
_ALL_OUTLETS = _names(OUTLETS + AMBIGUOUS_OUTLETS)
TITLE_SUFFIX = (re.compile(rf"\s+[-|–—:]\s+(?:{_ALL_OUTLETS})(?:\s+(?:Mobile|News|Online))?\s*$"), "")


def mask_sources(text, title=False):
    """Replace outlet and agency names (and, for a headline, drop a trailing site name). Returns a MaskedText."""
    return apply_replacements(text, ([TITLE_SUFFIX] if title else []) + SOURCE_PATTERNS)
