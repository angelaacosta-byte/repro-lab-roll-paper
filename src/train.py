"""Reproducible pipeline on real customs data: which structured customs fields predict a
proxy-indicated sustainable parent-roll import?

Run:  PYTHONHASHSEED=0 python src/train.py --seed 42

Model: logistic regression (one-hot + median imputation + scaling inside a Pipeline).
Validation: 5-fold StratifiedGroupKFold grouped by importer, so one importer is never in train and test at once.
Metric: PR-AUC (average precision), because only 4.4% of records are positive.
"""
import argparse, hashlib, json, os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "bobinas_model_data.csv")
CAT = ["hs", "origin", "customs_office", "transport", "channel", "product"]
NUM = ["log_kg", "log_usd_kg", "gsm", "supplier_known", "month"]
TARGET, GROUP = "sust", "importer_code"


def load():
    md5 = hashlib.md5(open(DATA, "rb").read()).hexdigest()
    df = pd.read_csv(DATA, dtype={"hs": str})
    df["log_kg"] = np.log10(df.net_kg)
    df["log_usd_kg"] = np.log10(df.fob_usd / df.net_kg)
    return df, md5


def preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CAT),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM)])


def make_model(name="logistic", seed=0, pos_weight=1.0):
    if name == "logistic":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
    elif name == "majority":
        from sklearn.dummy import DummyClassifier
        clf = DummyClassifier(strategy="prior")
    elif name == "xgboost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                            scale_pos_weight=pos_weight, eval_metric="logloss", random_state=seed, n_jobs=1)
    else:
        raise ValueError(name)
    return Pipeline([("pre", preprocessor()), ("clf", clf)])


def folds(df, seed):
    return list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                .split(df, df[TARGET], groups=df[GROUP]))


def evaluate(df, seed, model="logistic"):
    """Per-fold PR-AUC and balanced accuracy for one seed."""
    pr, ba = [], []
    for tr, te in folds(df, seed):
        y_tr = df[TARGET].iloc[tr]
        m = make_model(model, seed, pos_weight=(y_tr == 0).sum() / y_tr.sum())
        m.fit(df.iloc[tr], y_tr)
        p = m.predict_proba(df.iloc[te])[:, 1]
        pr.append(average_precision_score(df[TARGET].iloc[te], p))
        ba.append(balanced_accuracy_score(df[TARGET].iloc[te], p >= 0.5))
    return pr, ba


def main(seed=42, model="logistic"):
    np.random.seed(seed)
    df, md5 = load()
    pr, ba = evaluate(df, seed, model)
    return {"seed": seed, "model": model, "data_md5": md5,
            "pr_auc_mean": round(float(np.mean(pr)), 5), "pr_auc_sd": round(float(np.std(pr, ddof=1)), 5),
            "balanced_acc_mean": round(float(np.mean(ba)), 5)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    print(json.dumps(main(ap.parse_args().seed)))
