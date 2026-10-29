"""Automated reproducibility checks. Run from the repository root:  PYTHONHASHSEED=0 pytest -v"""
import hashlib, os, sys
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.train import main, load, folds, CAT, NUM   # noqa: E402
from src.run_seeds import run_all                   # noqa: E402

EXPECTED_MD5 = "bb033998c58693e41dba8b8f8046aded"   # data/bobinas_model_data.csv
EXPECTED_SEED42 = 0.46319                            # README "Expected output"
EXPECTED_HEADLINE = (0.4686, 0.0212)                 # PR-AUC mean, SD across 20 seeds
TOL = 1e-3


def test_data_hash_matches():
    """Data-version check: the committed data file is exactly the one used for the results."""
    md5 = hashlib.md5(open("data/bobinas_model_data.csv", "rb").read()).hexdigest()
    assert md5 == EXPECTED_MD5, f"Data file changed: MD5 {md5}"


def test_data_shape_and_label():
    df = pd.read_csv("data/bobinas_model_data.csv")
    assert len(df) == 2994
    assert set(df["sust"].unique()) == {0, 1}
    assert df["sust"].sum() == 133            # 4.4% proxy-indicated sustainable
    assert df["year"].between(2020, 2025).all()


def test_no_identities_or_label_text_in_data():
    """Privacy and leakage: no firm names and no description text are published or used as features."""
    df = pd.read_csv("data/bobinas_model_data.csv")
    banned = {"Importador", "Embarcador / Exportador", "Descripcion Comercial", "Cod. Tributario"}
    assert not banned & set(df.columns)
    assert df["importer_code"].str.fullmatch(r"IMP_\d{3}").all()
    assert "sust" not in CAT + NUM


def test_folds_never_share_an_importer():
    df, _ = load()
    for tr, te in folds(df, 42):
        assert not set(df.importer_code.iloc[tr]) & set(df.importer_code.iloc[te])


def test_same_seed_gives_identical_output():
    assert main(42) == main(42)


def test_seed42_matches_expected():
    assert main(42)["pr_auc_mean"] == pytest.approx(EXPECTED_SEED42, abs=TOL)


def test_results_table_is_reproduced_by_script():
    committed = pd.read_csv("results_summary.csv").sort_values("seed").reset_index(drop=True)
    fresh, _ = run_all()
    fresh = fresh.sort_values("seed").reset_index(drop=True)
    assert list(fresh.seed) == list(committed.seed)
    for col in ["pr_auc_mean", "pr_auc_sd", "balanced_acc_mean"]:
        assert fresh[col].to_numpy() == pytest.approx(committed[col].to_numpy(), abs=TOL)


def test_headline_mean_and_sd():
    fresh, _ = run_all()
    assert fresh.pr_auc_mean.mean() == pytest.approx(EXPECTED_HEADLINE[0], abs=TOL)
    assert fresh.pr_auc_mean.std(ddof=1) == pytest.approx(EXPECTED_HEADLINE[1], abs=TOL)
