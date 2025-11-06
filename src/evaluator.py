import numpy as np
from sklearn.metrics import roc_auc_score

import config as cfg


from typing import Any, Dict
import numpy as np
from sklearn.metrics import roc_auc_score
import config as cfg

def _coerce_probs(answer: Any, n_expected: int) -> np.ndarray:
    def _extract_vec(x):
        arr = np.asarray(list(x) if isinstance(x, (list, tuple)) else x)
        if arr.ndim == 2:
            if arr.shape[1] == 2:
                arr = arr[:, 1]
            elif arr.shape[1] == 1:
                arr = arr[:, 0]
        return arr

    vec = None
    if isinstance(answer, dict):
        for k in ("y_pred_proba", "probas", "proba", "pred_proba", "y_pred"):
            if k in answer:
                vec = answer[k]; break
        if vec is None and "answer" in answer and isinstance(answer["answer"], dict):
            for k in ("y_pred_proba", "probas", "proba", "pred_proba", "y_pred"):
                if k in answer["answer"]:
                    vec = answer["answer"][k]; break
    else:
        vec = answer

    if vec is None:
        raise ValueError("No probabilities found in submission")

    if isinstance(vec, list) and vec and isinstance(vec[0], tuple):
        vec = [t[-1] for t in vec]
    if isinstance(vec, list) and vec and isinstance(vec[0], dict):
        key = "prob_1" if "prob_1" in vec[0] else ("1" if "1" in vec[0] else None)
        if key:
            vec = [d[key] for d in vec]

    arr = _extract_vec(vec).astype(float)

    if arr.shape[0] != n_expected:
        raise ValueError(f"length {arr.shape[0]} != expected {n_expected}")
    return arr

def grade_submission(submission: Dict[str, Any], y_test, spec, test_df) -> Dict[str, Any]:
    try:
        y_pred = _coerce_probs(submission, n_expected=len(y_test))
    except Exception:
        return {"passed": False, "auc": float("nan"), "checks": {"submission_format_ok": False}}

    if not np.isfinite(y_pred).all():
        return {"passed": False, "auc": float("nan"), "checks": {"submission_numeric_ok": False}}

    y_pred = np.clip(y_pred, 0.0, 1.0)

    if len(np.unique(y_test)) < 2:
        return {"passed": False, "auc": float("nan"), "checks": {"auc_defined": False}}

    auc = float(roc_auc_score(y_test, y_pred))
    passed = auc >= cfg.THRESHOLD
    return {"passed": passed, "auc": auc,
            "checks": {"submission_format_ok": True, "submission_numeric_ok": True, "metric_ok": passed}}
