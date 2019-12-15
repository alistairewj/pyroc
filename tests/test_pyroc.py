from pytest import approx
from collections import OrderedDict

import numpy as np

import pyroc


def test_pyroc_init():
    roc = pyroc.ROC([0, 0, 1, 1], [0.4, 0.3, 0.8, 0.7])
    assert roc is not None


def test_pyroc_input_parsing(ovarian_cancer_dataset):
    df = ovarian_cancer_dataset

    expected_values = OrderedDict(
        [
            ('albumin', np.asarray([3., 3.2, 3.9])),
            ('total_protein', np.asarray([5.8, 6.3, 6.8])),
            # Krebs-Goplerud score
            ('total_score', np.asarray([0, 5, 7])),
        ]
    )
    n = 3

    # pandas series // pandas dataframe
    roc = pyroc.ROC(
        df['outcome'],
        df.drop('outcome', axis=1),
    )
    for i, p in enumerate(expected_values):
        assert (roc.preds[p][:n] == expected_values[p][:n]).all()

    # from now on we only pass target as numpy array
    target = df['outcome'].values
    df = df.drop('outcome', axis=1)

    # single numpy array of preds
    preds = df['albumin'].values
    roc = pyroc.ROC(
        target,
        preds,
    )
    # since roc wasn't provided predictor labels,
    # keys are integers (i)
    assert (roc.preds[0][:n] == expected_values['albumin'][:n]).all()

    # numpy arrays
    preds = df.values
    roc = pyroc.ROC(
        target,
        preds,
    )
    for i, p in enumerate(expected_values):
        # since roc wasn't provided predictor labels,
        # keys are integers (i)
        assert (roc.preds[i][:n] == expected_values[p][:n]).all()

    # list
    preds = [df[c].values for c in df.columns]
    roc = pyroc.ROC(
        target,
        preds,
    )
    for i, p in enumerate(expected_values):
        # since roc wasn't provided predictor labels,
        # keys are integers (i)
        assert (roc.preds[i][:n] == expected_values[p][:n]).all()

    # dict
    preds = {c: df[c].values for c in df.columns}
    roc = pyroc.ROC(
        target,
        preds,
    )
    for i, p in enumerate(expected_values):
        assert p in roc.preds.keys()
        assert (roc.preds[p][:n] == expected_values[p][:n]).all()


def test_pyroc_compare(ovarian_cancer_dataset):
    roc = pyroc.ROC(
        ovarian_cancer_dataset['outcome'],
        ovarian_cancer_dataset.drop('outcome', axis=1),
    )
    p, ci = roc.compare(np.array([[1, -1, 0], [1, 0, -1]]))

    assert p == approx(0.42291256338064165)
    assert ci[0] == approx(0.05063562)
    assert ci[1] == approx(7.37775891)
