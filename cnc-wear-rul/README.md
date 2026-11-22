# AI-Based CNC Tool Wear Classification and Remaining Useful Life (RUL) Prediction

Predictive-maintenance mini project (Practical No. 07). A milling cutter wears out with use. Replace it too
early and you waste tool life; too late and you get scrap parts and machine downtime. From sensor signals this
project

1. **classifies the wear state** of the tool (Healthy / Moderate / Worn), and
2. **predicts the Remaining Useful Life (RUL)** in runs,

and serves both through a Streamlit dashboard.

The code follows the course's **8-stage ML pipeline**. Stages 1 to 6 use only the training tools; the held-out
test tools are used **once**, in stage 7.

---

## Rubric to project map

| Stage | What the rubric asks | What this project does | Where |
|---|---|---|---|
| 1 Data and EDA | Missing values, class balance, correlations | Dataset summary table, wear curves, class-balance and correlation plots | `cncwear/data.py` → `outputs/stage1_*` |
| 2 Split | Stratified split, test set untouched, fixed seed | Whole **tools** are held out (about 25%), stratified by wear class with `StratifiedGroupKFold`, seed 42 | `cncwear/split.py` → `outputs/stage2_*` |
| 3 Preprocess | Imputation, scaling inside a Pipeline | `SimpleImputer` + constant-feature filter + `StandardScaler` inside every model's `Pipeline`, so each CV fold fits them on its own training part | `cncwear/models.py` → `outputs/stage3_pipeline.txt` |
| 4 Baselines | 3+ models plus a simple baseline, repeated stratified CV, mean ± std | 6 models per task plus 2 baselines, 4 folds x 3 repeats, every metric as mean ± std | `cncwear/validation.py` → `outputs/stage4_*` |
| 5 Tuning | GridSearch/Optuna, nested CV, fixed seed | Grid search inside **nested** CV (inner folds tune, outer folds score), then best parameters on the full training set | `cncwear/validation.py` → `outputs/stage5_*` |
| 6 Ablation | Remove one component at a time | Drops one sensor group / feature type at a time on identical folds, reports the performance drop | `cncwear/ablation.py` → `outputs/stage6_*` |
| 7 Test | Evaluate once: metrics, confusion matrix, ROC | One evaluation on the test tools: accuracy / precision / recall / F1 / ROC-AUC, confusion matrix, ROC curves, per-class and per-tool error analysis | `cncwear/pipeline.py` → `outputs/stage7_*` |
| 8 Deliver | Explainability, repo, README, demo | SHAP explanations, this repo with tests and CI, a 7-tab dashboard | `cncwear/explain.py`, `app.py` |

The marking scheme's "Code quality, reproducibility, demo" row is covered by: modular `cncwear/` package,
`config.yaml` (all settings in one place), `requirements.txt` + `requirements-lock.txt`, a pytest suite
(`tests/`), a GitHub Actions workflow and the dashboard.

## Quick start

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

1. Download the **NASA / UC Berkeley Milling Wear** dataset (search "NASA milling data set" on the NASA
   Prognostics Data Repository, or "Milling Tool Wear" on Kaggle) and put it in `data/` as `mill.mat`
   (recommended) or `mill.csv` (columns `case`, `run`, `VB` plus numeric sensor columns).
2. Train and evaluate everything: `python train.py` (a few minutes). Read the console tables and
   `outputs/results_tables.md`.
3. Open the dashboard: `streamlit run app.py`.
4. Check the code: `pip install -r requirements-dev.txt` then `pytest`.

`python train.py --quick` is a fast smoke test with reduced settings. **Never report numbers from quick mode.**

**Google Colab:** upload the project zip and use `colab_run.ipynb` (one cell per step, including the app).

## About the data

16 tools ("cases") were run until worn, 167 runs in total. Each run has the cutting parameters (depth of cut
`DOC`, `feed`, `material`), six sensor signals (spindle motor current `smcAC` / `smcDC`, table and spindle
vibration, table and spindle acoustic emission) and the measured flank wear `VB` (mm). Each raw signal (9000
samples) is summarised into 8 statistics (mean, std, RMS, max, min, peak-to-peak, skewness, kurtosis).

- **Wear state:** VB below 0.20 mm = Healthy, 0.20 to 0.40 = Moderate, 0.40 or more = Worn.
- **RUL:** the last recorded run of a tool minus the current run.
- Both definitions, the seed, the split size, CV settings and ablation groups are in `config.yaml`.
- The elapsed-time column `time` is **not** used as a feature because it reveals the tool's life directly.

## Why the split is by tool (important for the viva)

Runs of the same tool are highly correlated. A random row split would put near-copies of the same tool in
train and test and report fake-high scores. So the test set, every CV fold and every tuning fold keep each tool
on one side only, while `StratifiedGroupKFold` keeps the wear-class mix similar across folds. With only 16 tools,
the test set is small (about 4 tools), so the test numbers are indicative; the repeated CV mean ± std on the
training tools shows how stable they are.

## Dashboard tabs

| Tab | What it shows |
|---|---|
| Overview | Key results, wear curves (train vs test tools), class balance, the 8 stages |
| Live prediction | Fresh / mid-life / worn presets, sliders, RUL gauge, replace-or-keep advice, SHAP "why this prediction" |
| Tool life replay | Step through one tool's life: actual vs predicted RUL and wear state (test tools are truly unseen) |
| Validation (stages 4-6) | Baselines vs models, tuned models, ablation, all with mean ± std |
| Test results (stage 7) | Test metrics vs baselines, confusion matrix, ROC, predicted vs actual RUL, per-tool errors |
| Explainability | SHAP and Random Forest importance, signal explorer |
| Batch | Upload a CSV, download predictions |

## Project structure

```
cnc-wear-rul/
├── config.yaml            # thresholds, seed, split, CV, ablation groups
├── train.py               # entry point: runs the 8 stages
├── app.py                 # Streamlit dashboard
├── cncwear/
│   ├── data.py            # loading, labels, EDA          (stage 1)
│   ├── split.py           # stratified group hold-out     (stage 2)
│   ├── models.py          # Pipeline, models, baselines   (stages 3-4)
│   ├── validation.py      # repeated + nested CV, tuning  (stages 4-5)
│   ├── ablation.py        #                               (stage 6)
│   ├── metrics.py, plots.py
│   ├── explain.py         # SHAP                          (stage 8)
│   └── pipeline.py        # orchestrates all stages
├── tests/                 # pytest suite
├── docs/REPORT_GUIDE.md   # how to turn the outputs into the IEEE paper
├── colab_run.ipynb        # step-by-step Colab runner
├── data/                  # put mill.mat here (not committed)
├── models/                # created by train.py
└── outputs/               # created by train.py: tables, figures, results_tables.md
```

## Results (fill in after `python train.py`)

Copy the final numbers from `outputs/results_tables.md` (it already contains every table in markdown).

| Task | Chosen model | Nested-CV (mean ± std) | Held-out test |
|---|---|---|---|
| RUL regression | | RMSE = | MAE = , RMSE = , R² = |
| Wear-state classification | | Macro-F1 = | Accuracy = , Macro-F1 = , ROC-AUC = |

## Limitations (say these yourself in the viva)

- Only 16 tools and one machine setup, so results may not transfer to other machines, materials or tools.
- The test set is only about 4 tools; scores on it have wide uncertainty.
- Features are summary statistics; frequency-domain features (FFT) or a deep model (LSTM / 1D-CNN) could help.
- RUL is the number of runs until the end of the recorded experiment, not until a physical failure.
- The dashboard works on entered or uploaded readings, not on a live sensor stream.

## Ideas to extend

FFT / wavelet features, an LSTM or 1D-CNN on raw signals, prediction intervals for RUL, a cost-based
replace-now vs keep-running recommendation, validation on a second dataset (PHM 2010).

## Suggested Git history (the rubric looks at it)

Commit in small steps so the history tells the story, for example:
`Add data loading and EDA` → `Add stratified group split` → `Add preprocessing pipeline and baselines` →
`Add nested CV tuning` → `Add ablation study` → `Add test evaluation and ROC` → `Add SHAP explanations` →
`Add Streamlit dashboard` → `Add tests and CI` → `Final results and README`.
