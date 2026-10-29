# Reproducible ML Pipeline on Real Customs Data · Tissue Parent Rolls, Peru 2020–2025

[![Reproducibility check](https://github.com/angelaacosta-byte/repro-lab-roll-paper/actions/workflows/reproducibility.yml/badge.svg)](https://github.com/angelaacosta-byte/repro-lab-roll-paper/actions/workflows/reproducibility.yml)

**EN:** The Session 5 reproducibility lab, rebuilt on my own thesis data: real SUNAT/Veritrade import records of tissue parent rolls (*bobinas*). Question: which structured customs fields predict a proxy-indicated sustainable import (recycled-fibre or FSC/PEFC/ISO 14001 wording)? Built so a stranger gets the same numbers.
**ES:** El laboratorio de reproducibilidad de la Sesión 5, rehecho con los datos reales de mi tesis: importaciones de bobinas de papel tissue (SUNAT/Veritrade). Pregunta: ¿qué campos aduaneros predicen una importación con señal de sostenibilidad? Hecho para que un extraño obtenga los mismos números.

Author: Angela Milén Acosta Ticona · Doctoral Program in Deep Tech, AI & Emerging Technologies, UNMSM · Course: Research Methods & Scientific Integrity in AI (Dr. Loveleen Gaur)

## Main result · Resultado principal

| Model | PR-AUC, mean ± SD across 20 seeds | Base rate (random guess) |
|---|---|---|
| Logistic regression (one-hot + imputation + scaling inside a Pipeline, 5-fold CV grouped by importer) | **0.4686 ± 0.0212** | 0.0444 |

**EN:** PR-AUC is used instead of accuracy because only 4.4% of records are positive: a model that always says "not sustainable" would have 95.6% accuracy and be useless. The headline is the mean and SD across 20 seeds (0–19), the minimum recommended in class; no single seed is reported as the result. Seed 42 is kept only as a single-run check.
**ES:** Se usa PR-AUC y no exactitud porque solo el 4.4% de los registros son positivos: un modelo que siempre dice "no sostenible" tendría 95.6% de exactitud y no serviría. El resultado es la media y la desviación estándar entre 20 semillas (0–19).

### Fair comparison · Comparación justa (`src/compare_models.py`)

| Model | PR-AUC (20 seeds × 5 folds = 100 scores) |
|---|---|
| Majority class | 0.0444 ± 0.0062 |
| Logistic regression | 0.4686 ± 0.2326 |
| XGBoost | 0.4282 ± 0.2057 |

XGBoost vs logistic regression, corrected resampled t-test (Nadeau & Bengio, 2003): t = −0.60, **p = 0.55 → no significant difference; the simpler model is preferred.**

## How to reproduce · Cómo reproducir

```bash
git clone https://github.com/angelaacosta-byte/repro-lab-roll-paper.git && cd repro-lab-roll-paper
pip install -r requirements-lock.txt          # exact versions (Python 3.13). On macOS replace xgboost-cpu by xgboost.

PYTHONHASHSEED=0 python src/run_seeds.py      # 1) main result: regenerates results_summary.csv (20 seeds)
PYTHONHASHSEED=0 python src/train.py --seed 42   # 2) single-run check for one seed
PYTHONHASHSEED=0 python src/compare_models.py # 3) fair comparison of three models
pip install pytest==8.4.2 && PYTHONHASHSEED=0 pytest -v   # 4) automated checks (8 tests)
pip install mlflow && PYTHONHASHSEED=0 python src/run_seeds.py --mlflow   # 5) optional: log all 20 runs in MLflow
```

`PYTHONHASHSEED=0` goes **before** `python` on purpose: this seed only works if it is set when Python starts.

## Expected output · Resultado esperado

```
 seed  pr_auc_mean  pr_auc_sd  balanced_acc_mean
    0      0.47264    0.17237            0.79864
    1      0.47885    0.20198            0.79547
    2      0.46526    0.21007            0.80063
    3      0.44791    0.26578            0.78470
    4      0.47205    0.29288            0.81093
    5      0.49633    0.30079            0.81323
    6      0.42115    0.27752            0.77372
    7      0.48604    0.25848            0.79973
    8      0.51656    0.21703            0.80070
    9      0.46908    0.29699            0.80694
   10      0.45615    0.28224            0.79655
   11      0.47605    0.21257            0.79916
   12      0.45705    0.21590            0.79745
   13      0.49037    0.22602            0.80077
   14      0.44201    0.26886            0.78242
   15      0.47257    0.27911            0.78541
   16      0.44929    0.23661            0.79353
   17      0.46756    0.28313            0.79705
   18      0.48277    0.30396            0.78842
   19      0.45251    0.29407            0.79259

Headline result (20 seeds): PR-AUC = 0.4686 ± 0.0212 (mean ± SD); base rate = 0.0444
```

## Reproduction report (3 lines) · Informe de reproducción (3 líneas)

| Question (Session 5, NB3) | Answer |
|---|---|
| Same number again? · ¿Mismo número? | **Yes.** Seed 42 twice gives identical output; the 20-seed table is regenerated exactly by `src/run_seeds.py` and re-checked by GitHub Actions on every push. |
| Is the result valid? · ¿Es válido? | **Yes, with limits.** No leakage (preprocessing inside a Pipeline, description excluded, folds grouped by importer); compared with a trivial and a simple baseline; the label is a text proxy still to be validated by two raters (Cohen's kappa). |
| Problems found · Problemas | Rare positive class (4.4%) → PR-AUC, not accuracy; high fold-to-fold spread (SD ≈ 0.2); XGBoost does not beat logistic regression, so the simpler model is kept. |

## Session 5 toolkit: five layers · Las cinco capas de la Sesión 5

| Layer · Capa | Tool | In this repository |
|---|---|---|
| 1 · Code | Git | Full commit history on GitHub |
| 2 · Data | Fingerprint | `data/bobinas_model_data.csv`, MD5 `bb033998c58693e41dba8b8f8046aded`, checked by the tests |
| 3 · Experiments | MLflow | `src/run_seeds.py --mlflow` logs seed, model, data MD5, code commit and metrics for every run; exported table: `mlflow_runs.csv` |
| 4 · Environment | Pinned versions | `requirements-lock.txt`, `.python-version` |
| 5 · Reporting | README | mean ± SD over 20 seeds, never one lucky number |

## Data · Datos

- **Source:** Veritrade (SUNAT customs import declarations), HS 4803.00.90, 4818.20 and 4818.90, January 2020 – December 2025, cleaned to 2,994 parent-roll records (cleaned file MD5 `f2d7e5e54cc455749e50e57bb96dbfea`, see my thesis Datasheet).
- **Published file:** `data/bobinas_model_data.csv`, built by `src/make_dataset.py`. It contains **no firm names and no description text**: importers are replaced by codes (IMP_001…), suppliers by a yes/no flag. MD5 `bb033998c58693e41dba8b8f8046aded`; the tests fail if it changes.
- **Label `sust`:** 1 if the commercial description mentions recycled fibre (RECICLAD*, RECYCL*) or a certification (FSC, PEFC, ISO 14001, Ecolabel); 133 positives (4.4%). The description itself is not a feature, to avoid leakage.
- **Features:** HS subheading, origin country (top 8 + other), customs office, transport mode, inspection channel, product type, log net kg, log FOB per kg, grammage, supplier reported, month.

## Automated check · Verificación automática

`tests/test_reproducibility.py` checks that: (1) the data MD5 is unchanged; (2) the data have 2,994 rows and 133 positives; (3) no firm names or description text are present; (4) no importer appears in both training and test folds; (5) the same seed gives identical output; (6) seed 42 matches the expected value; (7) `results_summary.csv` (20 seeds) is regenerated by the script; (8) the headline mean and SD match. GitHub Actions runs all of this on every push.

## Difference with Deliverable 5 · Diferencia con el Entregable 5

| | Deliverable 5 (repro-lab) | This repository |
|---|---|---|
| Data | Public breast-cancer data (UCI) | **My real thesis data** (Veritrade/SUNAT) |
| Metric | Accuracy (balanced classes) | PR-AUC (rare positive class, 4.4%) |
| Validation | Stratified 5-fold CV | **Grouped** 5-fold CV by importer (avoids leakage between firms) |
| Extra checks | — | Privacy test (no firm names) and leakage test (no shared importer) |

## Licence
Code: MIT. Data: derived from Veritrade records under their terms of use; published only in anonymised, feature-level form for reproducibility.
