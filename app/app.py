import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Scorecard artifact from Phase 2 (optbinning Scorecard + scaling + cut-off)
from pathlib import Path
# Resolve the model path relative to this file, so the app works from any working directory
MODEL_PATH = Path(__file__).resolve().parent / "credit_scorecard.pkl"
artifact = joblib.load(MODEL_PATH)

scorecard = artifact["scorecard"]
cutoff = artifact["cost_cutoff"]
scaling = artifact["scaling"]
points_table = scorecard.table(style="detailed")
platt = None  # Platt scaling was evaluated in Phase 2 but not applied (no Brier improvement)
reason_ref = artifact.get("reason_reference")  # average points per variable over training applicants
reason_text = artifact.get("reason_text", {})
max_reasons = artifact.get("max_reasons", 4)


def calibrated_pd(p):
    # Platt scaling on the logit of the raw scorecard PD (does not change the score or ranking)
    if platt is None:
        return p
    p = min(max(p, 1e-6), 1 - 1e-6)
    z = platt["a"] + platt["b"] * np.log(p / (1 - p))
    return float(1 / (1 + np.exp(-z)))


def fmt_bin(b):
    # Categorical bins are stored as arrays of categories; numerical bins as interval strings
    if isinstance(b, str):
        return b
    try:
        return ", ".join(str(v) for v in list(b))
    except TypeError:
        return str(b)


def points_breakdown(row):
    # Points per variable for one applicant, compared with the best bin and the average applicant
    bp = scorecard.binning_process_
    out = []
    for var in bp.get_support(names=True):
        ob = bp.get_binned_variable(var)
        bin_idx = int(ob.transform(row[var].to_numpy(), metric="indices")[0])
        var_tab = points_table[points_table["Variable"] == var].reset_index(drop=True)
        pts = float(var_tab.loc[bin_idx, "Points"])
        max_pts = float(var_tab["Points"].max())
        avg_pts = float(reason_ref[str(var)]) if reason_ref else float("nan")
        out.append({"Variable": str(var), "Bin": fmt_bin(var_tab.loc[bin_idx, "Bin"]),
                    "Points": pts, "Average points": avg_pts, "Max points": max_pts,
                    "Below average": avg_pts - pts})
    return pd.DataFrame(out).sort_values("Below average", ascending=False).reset_index(drop=True)


st.title("Credit Scorecard")
st.write("Enter applicant information to get a credit score and probability of default (PD).")
st.caption(f"Scaling: {scaling['base_score']} points at {scaling['base_odds']}:1 good:bad odds, "
           f"PDO {scaling['pdo']}. Higher score = lower risk. "
           f"Approve if score >= {cutoff:.0f} (cost-optimal cut-off, 5:1 cost matrix, chosen out-of-fold).")

job = st.selectbox("Job (0 = unskilled non-resident ... 3 = highly skilled)", options=[0, 1, 2, 3], index=2)
housing = st.selectbox("Housing", options=["own", "rent", "free"])
saving_accounts = st.selectbox("Saving accounts",
                               options=["none", "little", "moderate", "quite rich", "rich"])
checking_account = st.selectbox("Checking account", options=["none", "little", "moderate", "rich"])
credit_amount = st.number_input("Credit amount (DM)", min_value=0, value=3000)
duration = st.number_input("Duration (months)", min_value=1, max_value=72, value=12)
purpose = st.selectbox("Purpose", options=["car", "radio/TV", "furniture/equipment", "business",
                                           "education", "repairs", "domestic appliances",
                                           "vacation/others"])

input_df = pd.DataFrame([{
    "Credit amount": credit_amount,
    "Duration": duration,
    "Job": str(job),
    "Housing": housing,
    "Saving accounts": saving_accounts,
    "Checking account": checking_account,
    "Purpose": purpose,
}])[artifact["features"]]

if st.button("Score applicant"):
    score = float(scorecard.score(input_df)[0])
    pd_hat = calibrated_pd(float(scorecard.predict_proba(input_df)[0, 1]))

    col1, col2 = st.columns(2)
    col1.metric("Credit score", f"{score:.0f}")
    col2.metric("Probability of default", f"{pd_hat:.1%}")

    if score >= cutoff:
        st.success(f"Decision: **APPROVE** (score {score:.0f} >= cut-off {cutoff:.0f})")
    else:
        st.error(f"Decision: **REJECT / refer** (score {score:.0f} < cut-off {cutoff:.0f})")

    breakdown = points_breakdown(input_df)

    if score < cutoff and reason_ref:
        # Reason codes: variables where the applicant scored furthest below the average applicant
        st.subheader("Principal reasons")
        below = breakdown[breakdown["Below average"] > 0].head(max_reasons)
        if below.empty:
            st.write("No variable is below the average applicant; the decision is driven by the overall score.")
        for i, (_, r) in enumerate(below.iterrows(), start=1):
            st.write(f"{i}. {reason_text.get(r['Variable'], r['Variable'])} "
                     f"({r['Bin']}: {r['Points']:.0f} points vs average {r['Average points']:.0f})")

    st.subheader("Points breakdown")
    st.caption("Sorted by points below the average applicant (positive = worse than average).")
    st.dataframe(breakdown.round(1), hide_index=True)
