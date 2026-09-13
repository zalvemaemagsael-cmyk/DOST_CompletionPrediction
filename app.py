"""
MSME Project Risk Dashboard
----------------------------
Single-page Streamlit app that answers one question on load:
which MSME projects are most likely to fail, and roughly why.

Data:  MSME_synthetic_panel_data.xlsx  (monthly panel, one row per project-month)
Model: MSME_CompletionModel_panel.pkl  (StandardScaler -> L1 Logistic Regression)
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "MSME_synthetic_panel_data.xlsx"
MODEL_PATH = APP_DIR / "MSME_CompletionModel_panel.pkl"

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="MSME Project Risk Dashboard",
    page_icon="\U0001F6A8",
    layout="wide",
)

TIER_COLORS = {
    "High risk": "#C0392B",
    "Medium risk": "#E08E0B",
    "Low risk": "#2E7D32",
}
TIER_BG = {
    "High risk": "#FDEDEC",
    "Medium risk": "#FEF5E7",
    "Low risk": "#EAF7EC",
}


# --------------------------------------------------------------------------
# Loading (cached so the app doesn't redo work on every rerun)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Synthetic_Panel")
    # Chronological order within each project, as instructed: there is no
    # separate row-counter column, so we sort explicitly.
    df = df.sort_values(["Project_ID", "Project_Year", "Month_of_Year"]).reset_index(drop=True)
    df["Month_Index"] = (df["Project_Year"] - 1) * 12 + df["Month_of_Year"]
    return df


@st.cache_data(show_spinner=False)
def score_projects(df: pd.DataFrame, _model) -> tuple[pd.DataFrame, list]:
    """Score the latest available record for every project and derive a
    plain-language 'why' from the model's own coefficients (no SHAP needed
    for a linear model)."""

    feat_cols = list(_model.feature_names_in_)

    # Latest record per project = most recent month observed for that project.
    latest = df.groupby("Project_ID", as_index=False).tail(1).copy()
    latest = latest.reset_index(drop=True)

    X = latest[feat_cols]
    proba = _model.predict_proba(X)  # column 1 = P(Completed)
    prob_complete = proba[:, 1]
    latest["Risk"] = 1 - prob_complete

    # Manually reconstruct the linear score so we can rank without the
    # probability saturating at 0/1, and so we can attribute risk to features.
    scaler = _model.named_steps["preprocessor"].transformers_[0][1]
    logreg = _model.named_steps["logreg"]
    means = scaler.mean_
    scales = scaler.scale_
    coefs = logreg.coef_[0]
    intercept = logreg.intercept_[0]

    Xv = X.to_numpy(dtype=float)
    Z = (Xv - means) / scales
    contrib = Z * coefs
    logit = intercept + contrib.sum(axis=1)
    latest["Logit"] = logit

    for i, col in enumerate(feat_cols):
        latest[f"z__{col}"] = Z[:, i]
        latest[f"contrib__{col}"] = contrib[:, i]

    # Rank by the raw log-odds (not the saturated probability) so the
    # ordering among near-100%-risk projects is still meaningful.
    latest = latest.sort_values("Logit", ascending=True).reset_index(drop=True)
    latest.insert(0, "Rank", np.arange(1, len(latest) + 1))

    def tier_of(r):
        if r >= 0.66:
            return "High risk"
        elif r >= 0.33:
            return "Medium risk"
        return "Low risk"

    latest["Tier"] = latest["Risk"].apply(tier_of)
    latest["Reason"] = latest.apply(lambda row: build_reason(row, feat_cols, means, scales), axis=1)

    return latest, feat_cols


# --------------------------------------------------------------------------
# Plain-English reason generation from linear-model contributions
# --------------------------------------------------------------------------
FEATURE_LABELS = {
    "Debt_Service_Coverage_Ratio": "debt-service coverage ratio",
    "Proponent_Equity_Share_Pct": "owner's equity stake",
    "Capacity_Utilization_Pre_Project_Pct": "pre-project capacity utilization",
}


def _phrase_for(col: str, value: float, mean: float) -> str:
    if col == "Debt_Service_Coverage_Ratio":
        if value < 1.0:
            return (
                f"Debt-service coverage ratio is critically low at {value:.2f}\u00d7 "
                f"\u2014 below the 1.0 break-even point needed to cover monthly debt payments."
            )
        return (
            f"Debt-service coverage ratio is soft at {value:.2f}\u00d7, "
            f"below the typical {mean:.2f}\u00d7."
        )
    if col == "Proponent_Equity_Share_Pct":
        return (
            f"Owner's equity stake is thin at {value:.1f}%, "
            f"versus a typical {mean:.1f}%."
        )
    if col == "Capacity_Utilization_Pre_Project_Pct":
        return (
            f"Pre-project capacity utilization was low at {value:.1f}%, "
            f"versus a typical {mean:.1f}%."
        )
    return f"{FEATURE_LABELS.get(col, col)} is below the typical range."


def build_reason(row: pd.Series, feat_cols: list, means: np.ndarray, scales: np.ndarray) -> str:
    if row["Risk"] < 0.5:
        return "No major red flags \u2014 fundamentals are in a healthy range."

    items = []
    for i, col in enumerate(feat_cols):
        items.append((col, row[f"contrib__{col}"], row[f"z__{col}"], row[col], means[i]))

    # Only features actively dragging the project toward "not completed"
    # (negative contribution) are candidate reasons.
    risk_drivers = sorted([it for it in items if it[1] < 0], key=lambda it: it[1])

    if not risk_drivers:
        return "Flagged by the model, though no single factor dominates \u2014 worth a closer look."

    phrases = [_phrase_for(risk_drivers[0][0], risk_drivers[0][3], risk_drivers[0][4])]

    # Add a second reason only if it is a meaningful secondary driver.
    if len(risk_drivers) > 1:
        strongest = abs(risk_drivers[0][1])
        second = risk_drivers[1]
        if strongest > 0 and abs(second[1]) >= 0.35 * strongest:
            phrases.append(_phrase_for(second[0], second[3], second[4]))

    return " Also, ".join(phrases)


def project_label(row: pd.Series) -> str:
    return str(row["Beneficiary_Name"])


# --------------------------------------------------------------------------
# Load everything
# --------------------------------------------------------------------------
model = load_model(MODEL_PATH)
panel_df = load_data(DATA_PATH)
scored, feat_cols = score_projects(panel_df, model)

# ==========================================================================
# 1. HEADLINE: the projects most likely to fail, and roughly why
# ==========================================================================
st.title("\U0001F6A8 MSME Risk Management Dashboard")
st.markdown(
    "###### A centralized view of MSME projects at risk of incompletion \u2014 "
    "evidence-based flags powered by Logistic Regression, built to support administrators' decision-making."
)
st.caption(
    "Projects below are ranked by predicted risk of incompletion, using each project's most recent monthly record."
)

top3 = scored.head(3)
cols = st.columns(3)
for col, (_, row) in zip(cols, top3.iterrows()):
    tier = row["Tier"]
    with col:
        st.markdown(
            f"""
            <div style="border:1px solid {TIER_COLORS[tier]}22; border-left:6px solid {TIER_COLORS[tier]};
                        background:{TIER_BG[tier]}; border-radius:8px; padding:14px 16px; height:100%;">
                <div style="font-size:0.8rem; color:#666; font-weight:600;">RANK #{row['Rank']}</div>
                <div style="font-size:1.05rem; font-weight:700; margin:2px 0 6px 0;">{project_label(row)}</div>
                <div style="font-size:1.8rem; font-weight:800; color:{TIER_COLORS[tier]};">{row['Risk']*100:.1f}% incompletion risk</div>
                <div style="display:inline-block; margin:4px 0 8px 0; padding:2px 10px; border-radius:12px;
                            background:{TIER_COLORS[tier]}; color:white; font-size:0.75rem; font-weight:700;">
                    {tier.upper()}
                </div>
                <div style="font-size:0.88rem; color:#333; margin-top:4px;">{row['Reason']}</div>
                <div style="font-size:0.75rem; color:#888; margin-top:8px;">
                    {row['Sector']} \u00b7 {row['Province']} \u00b7 {row['Size_of_Enterprise'].title()}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")

with st.expander("See full incompletion-risk ranking (top 15)", expanded=False):
    top15 = scored.head(15).copy()
    top15_display = pd.DataFrame(
        {
            "Rank": top15["Rank"],
            "Project": top15.apply(project_label, axis=1),
            "Sector": top15["Sector"],
            "Province": top15["Province"],
            "Size": top15["Size_of_Enterprise"].str.title(),
            "Incompletion Risk %": (top15["Risk"] * 100).round(1),
            "Tier": top15["Tier"],
            "Evidence (why flagged)": top15["Reason"],
        }
    )

    def tier_style(val):
        color = TIER_COLORS.get(val, "#333")
        return f"color: {color}; font-weight: 700;"

    st.dataframe(
        top15_display.style.map(tier_style, subset=["Tier"]),
        hide_index=True,
        width="stretch",
        column_config={
            "Incompletion Risk %": st.column_config.ProgressColumn(
                "Incompletion Risk %", min_value=0, max_value=100, format="%.1f%%"
            ),
        },
    )

st.divider()

# ==========================================================================
# 2. PORTFOLIO PULSE
# ==========================================================================
st.subheader("Portfolio pulse")
st.caption("A quick read on overall portfolio health, for context on the flags above.")

n_total = len(scored)
n_high = int((scored["Tier"] == "High risk").sum())
pct_high = n_high / n_total * 100
avg_risk = scored["Risk"].mean() * 100
avg_dscr = scored["Debt_Service_Coverage_Ratio"].mean()
n_below_1 = int((scored["Debt_Service_Coverage_Ratio"] < 1.0).sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Projects tracked", f"{n_total}")
m2.metric("At risk of incompletion", f"{n_high}", f"{pct_high:.0f}% of portfolio")
m3.metric("Avg. incompletion risk", f"{avg_risk:.1f}%")
m4.metric("Below break-even DSCR (< 1.0\u00d7)", f"{n_below_1}", f"of {n_total}")

st.divider()

# ==========================================================================
# 3. WHY: trend chart backing up the flags
# ==========================================================================
st.subheader("Evidence behind the flags")
st.caption("The trend data supporting each flag above, not just an asserted score.")

n_trend = min(5, (scored["Tier"] == "High risk").sum() or 5)
flagged_ids = scored.head(n_trend)["Project_ID"].tolist()
flagged_names = dict(zip(scored["Project_ID"], scored.apply(project_label, axis=1)))

trend_df = panel_df[panel_df["Project_ID"].isin(flagged_ids)].copy()
trend_df["Project"] = trend_df["Project_ID"].map(flagged_names)

portfolio_median = panel_df.groupby("Month_Index")["Debt_Service_Coverage_Ratio"].median().reset_index()

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=portfolio_median["Month_Index"],
        y=portfolio_median["Debt_Service_Coverage_Ratio"],
        name="Portfolio median",
        line=dict(color="#B0B0B0", width=2, dash="dot"),
        hovertemplate="Month %{x}<br>Portfolio median DSCR: %{y:.2f}<extra></extra>",
    )
)
for pid in flagged_ids:
    sub = trend_df[trend_df["Project_ID"] == pid].sort_values("Month_Index")
    fig.add_trace(
        go.Scatter(
            x=sub["Month_Index"],
            y=sub["Debt_Service_Coverage_Ratio"],
            name=flagged_names[pid],
            mode="lines",
            hovertemplate="Month %{x}<br>DSCR: %{y:.2f}<extra>" + flagged_names[pid] + "</extra>",
        )
    )
fig.add_hline(
    y=1.0,
    line_dash="dash",
    line_color="#C0392B",
    annotation_text="Break-even (1.0\u00d7)",
    annotation_position="bottom right",
)
fig.update_layout(
    height=420,
    xaxis_title="Project month (1 = project start)",
    yaxis_title="Debt-Service Coverage Ratio",
    legend_title_text="",
    margin=dict(t=10, b=10, l=10, r=10),
)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Dotted grey line = portfolio-wide median DSCR per project month, for context. "
    "Colored lines are the top flagged projects above."
)

st.divider()

# ==========================================================================
# 4. SEGMENT VIEW (only where it adds insight: where does risk concentrate?)
# ==========================================================================
st.subheader("Where incompletion risk concentrates")
st.caption("A segment-level cut to help administrators target oversight where it matters most.")

seg_choice = st.radio(
    "Group by", ["Sector", "Province", "Size_of_Enterprise"], horizontal=True, label_visibility="collapsed"
)

seg = (
    scored.groupby(seg_choice)
    .agg(Avg_Risk=("Risk", "mean"), Projects=("Project_ID", "count"), High_Risk=("Tier", lambda s: (s == "High risk").sum()))
    .reset_index()
)
seg["Avg_Risk_Pct"] = seg["Avg_Risk"] * 100
seg = seg.sort_values("Avg_Risk_Pct", ascending=True)
if seg_choice == "Size_of_Enterprise":
    seg[seg_choice] = seg[seg_choice].str.title()

fig2 = px.bar(
    seg,
    x="Avg_Risk_Pct",
    y=seg_choice,
    orientation="h",
    text=seg["Avg_Risk_Pct"].round(1).astype(str) + "%",
    hover_data={"Projects": True, "High_Risk": True, "Avg_Risk_Pct": ":.1f"},
    color="Avg_Risk_Pct",
    color_continuous_scale=["#2E7D32", "#E08E0B", "#C0392B"],
)
fig2.update_layout(
    height=320,
    xaxis_title="Average incompletion risk (%)",
    yaxis_title="",
    coloraxis_showscale=False,
    margin=dict(t=10, b=10, l=10, r=10),
)
fig2.update_traces(textposition="outside")
st.plotly_chart(fig2, width="stretch")

st.caption(
    "Powered by a Logistic Regression model (debt-service coverage ratio, owner's equity share, "
    "and pre-project capacity utilization) to give administrators an evidence-based, centralized "
    "view of incompletion risk across the MSME project portfolio. "
    "This dataset is synthetic and for demonstration purposes only."
)
