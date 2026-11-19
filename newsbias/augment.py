"""Training-time augmentations that discourage shortcut features.

Role masking (``RoleMasker``)
    Replaces party, leader and organisation names with a role phrase ("an opposition leader") at random, with
    probability ``p`` per mention, freshly each epoch. The side is kept, because the political labels are
    defined by framing towards the ruling side or the opposition; the identity is hidden, so the model cannot
    lean on a memorised name. Used only in training. ``mask_all`` is used by the evaluation diagnostic.

Gender swap of Neutral rows (``gender_swap``)
    Adds a copy of each Neutral row with gendered words swapped (he/she, men/women, ...). The label stays Neutral:
    straight reporting is straight whichever gender it is about. This teaches the gender head that a gendered
    word on its own is not evidence of bias. Biased rows are *not* swapped: in the guideline Misogynistic and
    Misandrist are defined differently (sexualising or judging women vs generalising about men as a category),
    so swapping the words of a Misogynistic article does not produce a Misandrist one.
"""
import random
import re
import tomllib
from pathlib import Path

from .text import apply_replacements

RESOURCES = Path(__file__).parent / "resources"


class RoleMasker:
    def __init__(self, path=None, p=0.5):
        path = Path(path) if path else RESOURCES / "political_entities.toml"
        groups = tomllib.loads(path.read_text(encoding="utf-8"))["group"]
        self.p = p
        self.patterns = []
        for g in groups:
            names = sorted(g["names"], key=len, reverse=True)
            alt = "|".join(re.escape(n) for n in names)
            self.patterns.append((re.compile(rf"(?<![\w-])(?:{alt})(?![\w-])"), g["replacement"]))

    def mask(self, text, rng):
        return apply_replacements(text, self.patterns, keep=lambda: rng.random() < self.p).text

    def mask_all(self, text):
        return apply_replacements(text, self.patterns).text

    def count(self, text):
        return len(apply_replacements(text, self.patterns).edits)


_PAIRS = [
    ("he", "she"), ("him", "her"), ("himself", "herself"), ("his", "her"), ("hers", "his"),
    ("man", "woman"), ("men", "women"), ("boy", "girl"), ("boys", "girls"), ("male", "female"),
    ("males", "females"), ("husband", "wife"), ("husbands", "wives"), ("father", "mother"),
    ("fathers", "mothers"), ("son", "daughter"), ("sons", "daughters"), ("brother", "sister"),
    ("brothers", "sisters"), ("uncle", "aunt"), ("uncles", "aunts"), ("nephew", "niece"),
    ("grandfather", "grandmother"), ("grandson", "granddaughter"), ("king", "queen"), ("prince", "princess"),
    ("groom", "bride"), ("boyfriend", "girlfriend"), ("gentleman", "lady"), ("gentlemen", "ladies"),
    ("mr", "ms"), ("mr.", "ms."), ("sir", "madam"), ("fatherhood", "motherhood"), ("paternal", "maternal"),
    ("paternity", "maternity"), ("masculine", "feminine"), ("masculinity", "femininity"),
    ("manhood", "womanhood"), ("spokesman", "spokeswoman"), ("chairman", "chairwoman"),
    ("businessman", "businesswoman"), ("businessmen", "businesswomen"), ("policeman", "policewoman"),
    ("sportsman", "sportswoman"), ("schoolboy", "schoolgirl"), ("widower", "widow"),
]
SWAP = {}
for a, b in _PAIRS:
    SWAP.setdefault(a, b)
    SWAP.setdefault(b, a)
# one-way mappings where the reverse would be wrong or ambiguous
SWAP.update({"mrs": "mr", "mrs.": "mr.", "actress": "actor", "actresses": "actors", "her": "his"})

_WORD = re.compile(r"(?<![\w])(" + "|".join(re.escape(w) for w in sorted(SWAP, key=len, reverse=True)) +
                   r")(?![\w])", re.I)
# after "her", these words mean it was an object pronoun ("told her to") rather than a possessive ("her book")
_NOT_A_NOUN = {"and", "or", "to", "in", "on", "at", "for", "with", "from", "by", "as", "that", "the", "a", "an",
               "is", "was", "were", "has", "had", "who", "but", "if", "when", "after", "before", "about", "of",
               "into", "over", "out", "up", "down", "off", "back", "again", "away", "too", "also"}


def _case_like(src, word):
    if src.isupper() and len(src) > 1:
        return word.upper()
    if src[0].isupper():
        return word[0].upper() + word[1:]
    return word


def gender_swap(text):
    """Return ``(swapped_text, number_of_swaps)``. Personal names are not changed."""
    count = 0

    def repl(m):
        nonlocal count
        src = m.group(1)
        low = src.lower()
        target = SWAP[low]
        if low == "her":
            rest = text[m.end():]
            nxt = re.match(r"\s+([A-Za-z]+)", rest)
            target = "his" if nxt and nxt.group(1).lower() not in _NOT_A_NOUN else "him"
        count += 1
        return _case_like(src, target)

    return _WORD.sub(repl, text), count


def make_rng(seed):
    return random.Random(seed)
