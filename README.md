# Credit Risk Scorecard: Evaluation, Calibration, Explainability and Fairness

An end-to-end credit risk project on the German Credit dataset (1,000 applicants, 30% bad). It starts from
a tutorial model and extends it with the steps used in credit risk work: sound evaluation, a WoE logistic
scorecard, calibration, cut-off strategy, reason codes and a fairness audit. A Streamlit app scores an
applicant and explains a decline.

> Built on top of the tutorial by Onur Baltaci,
> ["Credit Risk Modeling Using Machine Learning - Full Python Data Science Project"](https://www.youtube.com/watch?v=lhI999ETwmA).
> The tutorial covers EDA, label encoding and accuracy-based model comparison. Everything from Phase 1 onwards
> in this repository is my own extension.

**Final model:** WoE logistic scorecard, 6 variables, no `Sex` or `Age`. Score: 600 points at 50:1 good:bad odds, 20 points to double the odds.

## Live Demo

You can try it here: https://phuminan-credit-risk-scorecard.streamlit.app

## Key results

All numbers come from the notebooks in this repository (5-fold stratified CV on 800 training rows; 200-row held-out test set, used only for confirmation).

|                                                           | Value                                  |
| --------------------------------------------------------- | -------------------------------------- |
| Champion (scorecard, no Age), OOF AUC                     | 0.748                                  |
| Challenger (XGBoost), CV AUC                              | 0.769 +/- 0.052                        |
| Scorecard with Age vs XGBoost, paired fold AUC difference | -0.018 (std 0.019)                     |
| Champion test AUC                                         | 0.763                                  |
| Cost-optimal cut-off (chosen out-of-fold)                 | 536.5                                  |
| Test approval rate / bad rate among approved              | 33.5% / 13.4% (portfolio bad rate 30%) |

Main findings:

1. **Missing is information.** The tutorial dropped rows with missing account data (478 of 1,000). Applicants
   without a checking account default at 11.7% vs 41.9% for the rest, so dropping them removed the lowest-risk group.
2. **Transparency costs little here.** The scorecard matches a one-hot logistic regression (CV AUC 0.751 vs 0.750) and
   trails XGBoost by about 0.02 AUC, which is within fold-to-fold noise. XGBoost is kept as a challenger.
3. **The PDs are close to calibrated.** The cost-optimal cut-off chosen out-of-fold (PD 0.158) is within 2 score points of
   the theoretical optimum for a calibrated model (PD 1/6). Calibration slope 0.83; Platt scaling did not improve the Brier score.
4. **Reason codes depend on the reference.** Max-points and average-applicant references give the same top reason for
   only 64% of declined test applicants, about the same agreement as scorecard vs XGBoost SHAP (67%).
5. **Age, not sex, drove the disparity.** Applicants under 25 had a disparate impact ratio of 0.38, and creditworthy young
   applicants were approved at half the rate (22% vs 48%). Removing `Age` raised the ratio to 0.61 at a cost of 0.002 OOF AUC,
   so the final model excludes it.

## Project structure

```
.
├── notebooks/
│   ├── 01_evaluation.ipynb                  # Phase 1
│   ├── 02_scorecard_calibration.ipynb       # Phase 2
│   └── 03_explainability_fairness.ipynb     # Phase 3
├── app/
│   ├── app.py                               # Streamlit scorecard app
│   ├── credit_scorecard.pkl                 # final champion (created by notebook 03)
│   └── requirements.txt                     # app-only dependencies (deployment)
├── requirements.txt
└── README.md
```

## Phase 1: Sound evaluation

- Kept all 1,000 rows; missing `Saving accounts` / `Checking account` became a `none` category.
- Removed `Sex` from the inputs (kept aside for the fairness audit) and added `Purpose`.
- Selected models with stratified 5-fold CV on ROC-AUC on the training set. The test set was used once.
- Reported AUC, Gini and KS with a bootstrap CI, against a majority-class baseline (accuracy 0.70).
- Tuned a decision threshold out-of-fold with the dataset's cost matrix (approving a bad applicant costs 5, rejecting a good one costs 1).

| Model               | CV AUC (mean +/- std) |
| ------------------- | --------------------- |
| XGBoost             | 0.769 +/- 0.052       |
| Random Forest       | 0.763 +/- 0.054       |
| Logistic Regression | 0.750 +/- 0.055       |
| Extra Trees         | 0.746 +/- 0.060       |

XGBoost on the test set: AUC 0.785 (95% bootstrap CI 0.713-0.855), Gini 0.57, KS 0.45. The cost-optimal threshold (PD 0.15)
cut expected cost per applicant from 0.885 (threshold 0.5) to 0.55 (approve all: 1.5), but it rejected 64% of applicants.
That trade-off is what Phase 2 addresses.

## Phase 2: WoE scorecard and calibration

- Optimal binning with [optbinning](https://github.com/guillermo-navas-palencia/optbinning), with at least 5% of rows per bin and monotonic trends
  for numerical variables. Binning sits inside the CV pipeline.
- Information Value used for selection (IV >= 0.02). `Checking account` (IV 0.62) is above the usual 0.5 "suspicious" level.
  It was checked: the variable is known at application time, so this is not leakage.
- Logistic regression on WoE. Every coefficient has the expected sign.
- Points scaled with PDO 20. Each applicant's score equals the sum of the points table.
- Calibration checked with a reliability diagram and Brier score (OOF: scorecard 0.178, XGBoost 0.169, no-skill 0.210).
  The scorecard over-predicts PD in the highest-risk bin.
- Cut-offs are chosen on out-of-fold scores and applied unchanged to the test set.

| Cut-off rule       | OOF approval / bad rate among approved | Test approval / bad rate among approved |
| ------------------ | -------------------------------------- | --------------------------------------- |
| Cost-optimal (5:1) | 34.8% / 9.4%                           | 34.0% / 10.3%                           |
| Approve ~50%       | 50.0% / 13.2%                          | 53.5% / 12.1%                           |
| Approve ~70%       | 70.0% / 20.5%                          | 70.5% / 19.1%                           |

Raising approval from about 35% to 50% increases expected cost per applicant by about 9% out-of-fold (0.548 to 0.598).
Whether that pays off depends on loan margins, which this dataset does not contain.

## Phase 3: Explainability, reason codes and fairness

**SHAP.** XGBoost TreeSHAP and scorecard contributions agree on the main drivers. `Checking account` ranks first in both.
The largest difference is `Credit amount`, which XGBoost relies on more.

**Reason codes.** Declined applicants get up to four reasons. Three selection methods are compared: points below the best bin,
points below the average applicant, and points below the average applicant near the cut-off. The last two follow the example
methods in the US Regulation B commentary (comment 9(b)(2)-5). The app uses the average-applicant method.

**Fairness audit** (out-of-fold, cost-optimal cut-off):

| Group           | DI ratio (95% CI) | Approval among creditworthy | Mean PD vs observed bad rate |
| --------------- | ----------------- | --------------------------- | ---------------------------- |
| Female vs male  | 0.86 (0.69-1.07)  | 44% vs 46%                  | 0.31 vs 0.36 (female)        |
| Under 25 vs 25+ | 0.38 (0.22-0.56)  | 22% vs 48%                  | 0.42 vs 0.38 (under 25)      |

- **Sex:** no clear disparity. The confidence interval includes 1 and approval among creditworthy applicants is similar.
- **Age:** a clear disparity. The model also over-predicts risk for applicants under 25.
- **Proxy check:** the model inputs predict sex with AUC 0.62, and `Age` alone does the same. Age is the main proxy for sex in this data.
- **Decision:** removing `Age` changes OOF AUC by -0.002 (std 0.005) and raises the under-25 DI ratio to 0.61. The final model excludes `Age`.
  The remaining gap reflects a genuinely higher bad rate in this group (38% vs 29%) and possible proxies.

## Lessons learned

- **Training-serving skew in SHAP.** The preprocessing pipeline outputs a sparse matrix, and XGBoost treats absent sparse
  entries as missing. Converting it to dense before computing SHAP turned those entries into real zeros, so SHAP explained a
  different model (PDs differed by up to 0.5 on a synthetic check). The symptom was that the strongest variable by IV never appeared
  as a top reason. Fixed by computing TreeSHAP on exactly the matrix the model was trained on, and checking additivity against
  the pipeline's predictions.
- **Model portability.** A pickled XGBoost model trained on Linux failed to load on Windows with `input stream corrupted`
  ([dmlc/xgboost#12459](https://github.com/dmlc/xgboost/issues/12459)). The fix is to save boosters with `save_model()`.
- **Pin library versions.** A scikit-learn version mismatch between training and serving broke unpickling of the `ColumnTransformer`.

## Run it

## Run it

```bash
# Notebooks
pip install -r requirements.txt
# Run notebooks/01-03 in order (they download the data with kagglehub);
# notebook 03 writes credit_scorecard.pkl - copy it into app/

# App only
pip install -r app/requirements.txt
cd app
python -m streamlit run app.py
```

Developed with Python 3.13 (Google Colab). The pickled model must be loaded with the same `scikit-learn` and `optbinning` versions it was saved with.

## Limitations

- 1,000 applicants from 1970s-80s Germany. The methods transfer; the numbers and the policy conclusions do not.
- The test set has about 60 bad customers, so test metrics carry wide intervals. Decisions rely on out-of-fold results.
- The 5:1 cost matrix ignores revenue from good customers. A real cut-off would come from profitability and risk appetite.
- Group-level fairness metrics rest on small groups (110 applicants under 25 out-of-fold).
- Regulation B is US law. It is used here as a reference for reason-code methods, not as legal advice. Section 1002.6(a)
  on the "effects test" was amended in 2026. Other jurisdictions apply their own rules.
- SHAP and scorecard points explain the models, not causal effects.

## Dataset Credit

- **Original data:** Hofmann, H. (1994). _Statlog (German Credit Data)_ [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5NC77. License: Creative Commons Attribution 4.0 International (CC BY 4.0), as shown on the
  [UCI dataset page](https://archive.ics.uci.edu/dataset/144).
- **Version used:** [German Credit Risk - With Target](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk) on Kaggle (user `kabure`),
  a reduced version with 9 of the original 20 attributes plus the `Risk` label. License shown on the Kaggle page:
  [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/).
