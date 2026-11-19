from newsbias import baselines
from newsbias.data import load_examples
from newsbias.labels import label_index


def test_single_class_training_fold_does_not_crash(data_file):
    ex = load_examples(data_file)
    # one Misogynistic row and the rest Neutral, like the first 40 rows of the real corpus
    ex = [e for e in ex if e.gender == "Neutral"][:20] + [e for e in ex if e.gender == "Misogynistic"][:1]
    folds = [[e.id for e in ex[i::3]] for i in range(3)]
    folds = [f for f in folds if ex[-1].id not in f] + [[x for f in folds if ex[-1].id in f for x in f]]
    preds = baselines.out_of_fold(ex, folds)
    neutral = label_index("gender", "Neutral")
    assert preds["tfidf_logreg"]["gender"][-1] == neutral   # trained on Neutral only, predicts it
    feats = baselines.top_features(ex)
    assert set(feats["gender"]) == {"Misogynistic", "Neutral"}


def test_top_features_binary_sign(data_file):
    ex = [e for e in load_examples(data_file) if e.gender in ("Misogynistic", "Neutral")]
    feats = baselines.top_features(ex, n=30)["gender"]
    # the synthetic Misogynistic rows use these words; they must not be listed under Neutral
    assert {"glamorous", "actress", "scandal", "bedroom", "outfit"} & set(feats["Misogynistic"])
    assert not {"glamorous", "actress", "scandal", "bedroom"} & set(feats["Neutral"])
