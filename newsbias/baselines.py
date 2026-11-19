"""Reference models run on exactly the same folds and the same masked text.

* Majority class: what you get by always saying Center / Neutral. Its balanced accuracy is 1/3.
* TF-IDF + logistic regression (word 1-2 grams, class-balanced): a bag-of-words model. If IndicBERTv2 does not
  beat it, the fine-tuning is not adding anything. Its highest-weighted n-grams, fitted on all rows, are listed
  in the report as a cheap look at which surface words separate the classes in this data (party and person
  names near the top are a warning sign of shortcuts the transformer could also pick up).
"""
import numpy as np

from .labels import AXES, label_index


def _texts(examples):
    return [e.title + " " + e.lede for e in examples]


def _pipeline():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))


def out_of_fold(examples, folds):
    """Returns {"majority": {axis: preds}, "tfidf_logreg": {axis: preds}} aligned with ``examples``."""
    index = {e.id: i for i, e in enumerate(examples)}
    preds = {"majority": {a: np.zeros(len(examples), int) for a in AXES},
             "tfidf_logreg": {a: np.zeros(len(examples), int) for a in AXES}}
    for test_ids in folds:
        test = [index[x] for x in test_ids]
        test_set = set(test)
        train = [i for i in range(len(examples)) if i not in test_set]
        tr_ex = [examples[i] for i in train]
        te_ex = [examples[i] for i in test]
        for axis in AXES:
            y = np.array([label_index(axis, getattr(e, axis)) for e in tr_ex])
            majority = np.bincount(y, minlength=len(AXES[axis])).argmax()
            preds["majority"][axis][test] = majority
            if len(set(y.tolist())) < 2:
                # a training fold with one class (only possible on a small --limit subset): nothing to learn
                preds["tfidf_logreg"][axis][test] = majority
                continue
            model = _pipeline().fit(_texts(tr_ex), y)
            preds["tfidf_logreg"][axis][test] = model.predict(_texts(te_ex))
    return preds


def top_features(examples, n=12):
    out = {}
    for axis, classes in AXES.items():
        y = np.array([label_index(axis, getattr(e, axis)) for e in examples])
        out[axis] = {}
        if len(set(y.tolist())) < 2:
            continue
        model = _pipeline().fit(_texts(examples), y)
        vec, lr = model.steps[0][1], model.steps[1][1]
        names = np.array(vec.get_feature_names_out())
        for row, cls_idx in enumerate(lr.classes_):
            # with two classes sklearn keeps one coefficient row, pointing towards classes_[1]
            coef = lr.coef_[row] if lr.coef_.shape[0] > 1 else (lr.coef_[0] if row == 1 else -lr.coef_[0])
            out[axis][classes[cls_idx]] = names[np.argsort(-coef)[:n]].tolist()
    return out
