"""Fair comparison on identical folds: majority class vs logistic regression vs XGBoost.

Run:  PYTHONHASHSEED=0 python src/compare_models.py

All models see the same features and the same 20 seeds x 5 grouped folds (100 scores each).
XGBoost vs logistic regression is tested with the corrected resampled t-test (Nadeau & Bengio, 2003).
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import load, evaluate  # noqa: E402
from src.run_seeds import SEEDS       # noqa: E402

MODELS = ["majority", "logistic", "xgboost"]


def compare():
    df, _ = load()
    scores = {m: [] for m in MODELS}
    for s in SEEDS:
        for m in MODELS:
            scores[m] += evaluate(df, s, m)[0]
    d = np.array(scores["xgboost"]) - np.array(scores["logistic"])
    n_test = len(df) / 5
    t = d.mean() / np.sqrt(d.var(ddof=1) * (1 / len(d) + n_test / (len(df) - n_test)))
    p = 2 * stats.t.sf(abs(t), len(d) - 1)
    table = pd.DataFrame({"model": MODELS,
                          "pr_auc_mean": [round(float(np.mean(scores[m])), 4) for m in MODELS],
                          "pr_auc_sd": [round(float(np.std(scores[m], ddof=1)), 4) for m in MODELS]})
    return table, round(float(t), 3), round(float(p), 4)


if __name__ == "__main__":
    table, t, p = compare()
    table.to_csv("model_comparison.csv", index=False)
    print(table.to_string(index=False))
    print(f"\nXGBoost vs logistic regression, corrected resampled t-test: t = {t}, p = {p}")
    print("-> no significant difference: the simpler model is preferred" if p >= 0.05 else "-> significant difference")
