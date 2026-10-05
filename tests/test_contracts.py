import numpy as np
import pandas as pd
import pytest

from dat_parkinsons.contracts import align_predictions, patient_folds, submission
from dat_parkinsons.ensemble import blend, fit_blend


def table():
    return pd.DataFrame(
        {
            "patient_id": ["a", "b", "c", "d"],
            "label": [0, 1, 0, 1],
            "pred_probability": [0.1, 0.8, 0.2, 0.9],
            "fold": [0, 0, 1, 1],
        }
    )


def test_alignment_is_by_id_not_row_order():
    a = table()
    ids, y, p = align_predictions({"a": a, "b": a.iloc[::-1]})
    assert ids.tolist() == ["a", "b", "c", "d"]
    assert np.array_equal(p[:, 0], p[:, 1])
    assert y.tolist() == [0, 1, 0, 1]


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "label", "fold", "nan", "range"])
def test_alignment_rejects_corrupt_inputs(corruption):
    a, b = table(), table()
    if corruption == "duplicate":
        b = pd.concat([b, b.iloc[:1]])
    elif corruption == "missing":
        b = b.iloc[1:]
    elif corruption == "label":
        b.loc[0, "label"] = 1
    elif corruption == "fold":
        b.loc[0, "fold"] = 1
    elif corruption == "nan":
        b.loc[0, "pred_probability"] = np.nan
    else:
        b.loc[0, "pred_probability"] = 1.2
    with pytest.raises(ValueError):
        align_predictions({"a": a, "b": b})


def test_folds_stable_under_reordering():
    a = patient_folds(table(), 2)
    b = patient_folds(table().iloc[::-1], 2)
    pd.testing.assert_frame_equal(a, b)
    assert a.patient_id.is_unique


def test_logit_blend_analytic_result_and_zero_weight():
    # Odds 1/4 and 4, equally weighted in log space, give probability 1/2.
    assert blend([[0.2, 0.8]], [1, 1])[0] == pytest.approx(0.5)
    assert blend([[0.2, 0.8]], [0, 1])[0] == pytest.approx(0.8)
    assert np.isfinite(blend([[0, 1]], [1, 1])).all()
    with pytest.raises(ValueError):
        blend([[0.2, 0.8]], [0, 0])


def test_blend_prefers_informative_branch():
    y = [0, 1, 0, 1]
    p = np.array([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
    weights = fit_blend(p, y)
    assert weights[0] > 0.99
    assert weights.sum() == pytest.approx(1)


def test_submission_contract():
    frame = submission(["a", "b"], [0.2, 0.8])
    assert frame.columns.tolist() == ["uid", "is_pathologic"]
    with pytest.raises(ValueError):
        submission(["a", "a"], [0.2, 0.8])
