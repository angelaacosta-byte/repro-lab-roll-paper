"""Regenerate results_summary.csv: the same pipeline run with 20 seeds (0-19).

Run:  PYTHONHASHSEED=0 python src/run_seeds.py
      PYTHONHASHSEED=0 python src/run_seeds.py --mlflow   # also log every run to MLflow (pip install mlflow)

Each row is one seed (5-fold grouped CV). The headline result is the mean and SD of
pr_auc_mean ACROSS the 20 seeds. No single seed is reported as the result.
With --mlflow, every run is logged with its seed, model, data MD5 and code commit,
and the MLflow table is exported to mlflow_runs.csv (the MLflow database itself is not tracked by Git).
"""
import argparse, os, subprocess, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import main  # noqa: E402

SEEDS = list(range(20))   # 20 seeds: 0..19


def code_commit(path="src/train.py"):
    """Short hash of the last commit that changed the training code."""
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%h", "--", path],
                             capture_output=True, text=True, check=True).stdout.strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def run_all(seeds=SEEDS):
    rows = [main(s) for s in seeds]
    return pd.DataFrame(rows)[["seed", "pr_auc_mean", "pr_auc_sd", "balanced_acc_mean"]], rows[0]["data_md5"]


def log_to_mlflow(table, data_md5, out_csv="mlflow_runs.csv"):
    import mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("repro-lab-roll-paper")
    commit = code_commit()
    for _, r in table.iterrows():
        with mlflow.start_run(run_name=f"seed-{int(r.seed)}"):
            mlflow.log_params({"model": "logistic_regression", "cv": "5-fold grouped by importer",
                               "seed": int(r.seed), "data_md5": data_md5, "git_commit": commit})
            mlflow.log_metrics({"pr_auc_mean": r.pr_auc_mean, "pr_auc_sd": r.pr_auc_sd,
                                "balanced_acc_mean": r.balanced_acc_mean})
    runs = mlflow.search_runs(experiment_names=["repro-lab-roll-paper"])
    cols = ["tags.mlflow.runName", "params.seed", "params.model", "params.cv", "params.data_md5", "params.git_commit",
            "metrics.pr_auc_mean", "metrics.pr_auc_sd", "metrics.balanced_acc_mean"]
    runs = runs[cols].assign(seed=lambda d: d["params.seed"].astype(int)).sort_values("seed").drop(columns="seed")
    runs.to_csv(out_csv, index=False)
    print(f"MLflow: {len(table)} runs logged to mlflow.db and exported to {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_summary.csv")
    ap.add_argument("--mlflow", action="store_true", help="also log the runs to MLflow")
    args = ap.parse_args()
    table, md5 = run_all()
    table.to_csv(args.out, index=False)
    if args.mlflow:
        log_to_mlflow(table, md5)
    print(table.to_string(index=False))
    m, s = table.pr_auc_mean.mean(), table.pr_auc_mean.std(ddof=1)
    print(f"\nHeadline result (20 seeds): PR-AUC = {m:.4f} ± {s:.4f} (mean ± SD); "
          f"base rate = 0.0444  |  data MD5 {md5}")
