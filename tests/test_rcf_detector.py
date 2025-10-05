import numpy as np
from log_anomaly_detection.model.rcf_detector import RCFDetector
import pytest
import os

def _rows_match(e_rows, a_rows, rtol=1e-9, atol=1e-12):
    return all(any(np.allclose(e, a, rtol=rtol, atol=atol) for a in a_rows) for e in e_rows)

def test_fit_sets_flag_and_forests():
    X = np.random.default_rng(0).normal(size=(100, 3))
    det = RCFDetector().fit(X, n_trees=5, tree_size=32, random_state=1)
    assert det.is_fitted == True
    assert det.forest is not None and len(det.forest) == 5
    assert det.n_features_ == 3

def test_score_shapes_and_outliers():
    rng = np.random.default_rng(1)
    X_train = rng.normal(size=(100, 3))
    det = RCFDetector().fit(X_train, n_trees=5, tree_size=32, random_state=1)
    rng = np.random.default_rng(2)
    X_score = np.vstack([rng.normal(size=(30, 3)), [80,80,80], [-7,-7,-7]])
    scores = det.score(X_score)
    top_2_outlier_indices = np.argsort(scores)[-2:][::-1]
    expected = np.array([[80.,80.,80.], [-7.,-7.,-7.]])
    assert _rows_match(expected, X_score[top_2_outlier_indices])

def test_save_and_load(tmp_path):
    rng = np.random.default_rng(3)
    X_train = rng.normal(size=(100, 3))
    X_score = np.vstack([rng.normal(size=(30, 3)), [80,80,80], [-7,-7,-7]])
    path = str(tmp_path / "rcf.joblib")
    # train & save
    det = RCFDetector().fit(X_train, n_trees=5, tree_size=32, random_state=1)
    det.save(str(path))
    assert os.path.exists(path)
    # reload and validate expected outlier is caught
    det = RCFDetector.load(str(path))
    assert len(det.forest) == 5
    scores_after_reload = det.score(X_score)
    top_2_outlier_indices = np.argsort(scores_after_reload)[-2:][::-1]
    expected = np.array([[80.,80.,80.], [-7.,-7.,-7.]])
    assert _rows_match(expected, X_score[top_2_outlier_indices])

def test_score_or_save_before_fit(tmp_path):
    det = RCFDetector()
    rng = np.random.default_rng(4)
    X = rng.random(size=(100,3))
    with pytest.raises(RuntimeError):
        det.score(X)
    with pytest.raises(RuntimeError):
        det.save(path=str(tmp_path / "rcf_test_score_or_save_before_fit.joblib"))

def test_score_expects_same_shape_as_train():
    rng = np.random.default_rng(5)
    X_train = rng.normal(size=(100, 3))
    det = RCFDetector().fit(X_train, n_trees=5, tree_size=32, random_state=1)
    rng = np.random.default_rng(6)
    X_score_diff_shape = np.vstack([rng.normal(size=(30, 4)), [80,80,80,80], [-7,-7,-7,-7]])
    with pytest.raises(ValueError):
        det.score(X_score_diff_shape)
