from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

import config as cfg

@dataclass
class Spec:
    leak_cols: List[str]
    target_name: str = "target"

def _make_base(seed: int, n_train: int, n_test: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    Xn_train = rng.normal(0, 1, size=(n_train, 5))
    Xn_test  = rng.normal(0, 1, size=(n_test, 5))
    cat0_train = rng.integers(0, 4, size=n_train)
    cat0_test  = rng.integers(0, 4, size=n_test)
    cat1_train = rng.integers(0, 3, size=n_train)
    cat1_test  = rng.integers(0, 3, size=n_test)

    cols_num = [f"num_{i}" for i in range(5)]
    train = pd.DataFrame(Xn_train, columns=cols_num)
    test  = pd.DataFrame(Xn_test,  columns=cols_num)
    train["cat0"] = pd.Series(cat0_train, dtype="category")
    test["cat0"]  = pd.Series(cat0_test,  dtype="category")
    train["cat1"] = pd.Series(cat1_train, dtype="category")
    test["cat1"]  = pd.Series(cat1_test,  dtype="category")

    w = np.array([1.2, -0.8, 0.6, 0.4, -0.2])
    logit_train = Xn_train @ w + 0.5*(cat0_train==1) - 0.3*(cat1_train==2) + rng.normal(0, 0.8, n_train)
    logit_test  = Xn_test  @ w + 0.5*(cat0_test==1)  - 0.3*(cat1_test==2)  + rng.normal(0, 0.8, n_test)

    p_train = 1/(1+np.exp(-logit_train))
    p_test  = 1/(1+np.exp(-logit_test))
    y_train = (p_train > 0.5).astype(int)
    y_test  = (p_test  > 0.5).astype(int)

    train["target"] = y_train
    test["target"]  = y_test
    return train, test

def make_problem(seed: int = 42, n_train: int = cfg.N_TRAIN, n_test: int = cfg.N_TEST):
    train, test = _make_base(seed, n_train, n_test)

    leak_cols = []
    if cfg.ENABLE_LEAK_FUTURE_SIGNAL:
        train["leak_future_signal"] = train["target"]
        test["leak_future_signal"]  = test["target"]
        leak_cols.append("leak_future_signal")

    if cfg.ENABLE_LEAK_GLOBAL_TARGET_MEAN:
        global_mean = pd.concat([train["target"], test["target"]], axis=0).mean()
        train["leak_global_target_mean"] = float(global_mean)
        test["leak_global_target_mean"]  = float(global_mean)
        leak_cols.append("leak_global_target_mean")

    if cfg.ENABLE_LEAK_CAT0_RATE_FULL:
        full = pd.concat([train[["cat0","target"]].assign(split="train"),
                          test[["cat0","target"]].assign(split="test")])
        rates = full.groupby("cat0", observed=False)["target"].mean()
        train["leak_cat0_rate_full"] = train["cat0"].map(rates).astype(float)
        test["leak_cat0_rate_full"]  = test["cat0"].map(rates).astype(float)
        leak_cols.append("leak_cat0_rate_full")

    spec = Spec(leak_cols=leak_cols, target_name="target")
    y_test = test["target"].to_numpy().astype(int)
    return train, test, y_test, spec

def build_baseline_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, leak_cols: List[str], target: str):
    leaks_present_train = [c for c in leak_cols if c in train_df.columns]
    leaks_present_test  = [c for c in leak_cols if c in test_df.columns]

    X_train = train_df.drop(columns=leaks_present_train + [target])
    y_train = train_df[target].astype(int)
    X_test  = test_df.drop(columns=leaks_present_test)

    num_cols = [c for c in X_train.columns if str(X_train[c].dtype) != 'category']
    cat_cols = [c for c in X_train.columns if str(X_train[c].dtype) == 'category']

    num_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

    clf = LogisticRegression(max_iter=1000)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    return {"y_pred_proba": y_proba, "pipeline": pipe}
