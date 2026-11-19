import random

from newsbias.augment import RoleMasker, gender_swap


def test_gender_swap_words_and_case():
    text, n = gender_swap("He told her that Men and women gave her book to his sister.")
    assert text == "She told him that Women and men gave his book to her brother."
    assert n == 7


def test_gender_swap_leaves_other_words():
    text, n = gender_swap("The committee met on Monday; themselves, shell, menu, Herald.")
    assert n == 0
    assert text == "The committee met on Monday; themselves, shell, menu, Herald."


def test_role_mask_all_and_longest_match():
    rm = RoleMasker(p=1.0)
    out = rm.mask_all("Rahul Gandhi of the Congress met Shiv Sena (UBT) chief; Modi and BJP responded.")
    assert "Rahul" not in out and "Modi" not in out and "BJP" not in out
    assert "an opposition leader of the an opposition party" in out
    assert "met an opposition party chief" in out
    assert "a ruling-party leader and the ruling party responded" in out


def test_role_mask_probability_zero_and_ordinary_words():
    rm = RoleMasker(p=0.0)
    text = "Modi spoke to the CPI data team in Congress Nagar."
    assert rm.mask(text, random.Random(0)) == text
    assert "CPI" in RoleMasker(p=1.0).mask_all(text)
