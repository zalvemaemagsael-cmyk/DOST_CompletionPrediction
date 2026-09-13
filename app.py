"""
ASENXO MSME Project Completion Risk Management Dashboard
----------------------------------------------------------
Monitor -> Predict -> Explain Risk -> Prioritize -> Intervene -> Monitor

Data:  MSME_synthetic_panel_data.xlsx  (monthly panel, one row per project-month)
Model: MSME_CompletionModel_panel.pkl  (StandardScaler -> L1 Logistic Regression)

IMPORTANT: dataset and model are SYNTHETIC / for demonstration only.
See the README sheet inside MSME_synthetic_panel_data.xlsx for full data-generation
notes and caveats (near-perfect separability by construction, illustrative
Province/Sector tags, etc.). Nothing in this app should be read as an official
DOST statistic or decision.
"""

from __future__ import annotations

import hashlib
import pickle
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================================================
# Paths & page config
# ==========================================================================
APP_DIR = Path(__file__).parent
DATA_PATH = APP_DIR / "MSME_synthetic_panel_data.xlsx"
MODEL_PATH = APP_DIR / "MSME_CompletionModel_panel.pkl"

st.set_page_config(
    page_title="ASENXO | MSME Project Completion Risk Dashboard",
    page_icon="\U0001F3DB\uFE0F",
    layout="wide",
)

# --------------------------------------------------------------------------
# Palette - DOST-style navy/blue government analytics look
# --------------------------------------------------------------------------
NAVY = "#0B3D66"
NAVY_DARK = "#082B49"
ACCENT = "#1F6FB2"
BG = "#F4F7FA"
CARD_BG = "#FFFFFF"
BORDER = "#E1E8EF"

TIER_ORDER = ["Critical", "High", "Medium", "Low"]
TIER_COLORS = {
    "Critical": "#B3261E",
    "High": "#D96C06",
    "Medium": "#C99A00",
    "Low": "#2E7D32",
}
TIER_BG = {
    "Critical": "#FBEAE9",
    "High": "#FDF1E4",
    "Medium": "#FDF6DC",
    "Low": "#EAF5EC",
}

MODEL_PREDICTORS = [
    "Debt_Service_Coverage_Ratio",
    "Proponent_Equity_Share_Pct",
    "Capacity_Utilization_Pre_Project_Pct",
]
FEATURE_LABELS = {
    "Debt_Service_Coverage_Ratio": "Debt-Service Coverage Ratio (DSCR)",
    "Proponent_Equity_Share_Pct": "Proponent Equity Share (%)",
    "Capacity_Utilization_Pre_Project_Pct": "Pre-Project Capacity Utilization (%)",
}

PROVINCE_COORDS = {
    "Iloilo": (10.7202, 122.5621),
    "Negros Occidental": (10.6407, 123.1568),
    "Capiz": (11.3889, 122.7550),
    "Aklan": (11.8166, 122.0942),
    "Antique": (10.7797, 121.9032),
    "Guimaras": (10.5931, 122.6325),
}

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {BG}; }}
    section[data-testid="stSidebar"] {{ background-color: {NAVY_DARK}; }}
    section[data-testid="stSidebar"] * {{ color: #EAF1F8 !important; }}
    section[data-testid="stSidebar"] .stSlider label, section[data-testid="stSidebar"] .stMultiSelect label {{
        color: #EAF1F8 !important;
    }}
    h1, h2, h3 {{ color: {NAVY}; }}
    div[data-testid="stMetric"] {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 12px 16px;
    }}
    .kpi-note {{ color:#667; font-size:0.78rem; }}
    .demo-banner {{
        background: #FFF4E5;
        border: 1px solid #F0C36D;
        border-left: 6px solid #B98900;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 0.86rem;
        color: #5A4300;
        margin-bottom: 10px;
    }}
    .section-caption {{ color:#5A6B7B; font-size:0.9rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# Loading (cached)
# ==========================================================================
@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Synthetic_Panel")
    df = df.sort_values(["Project_ID", "Project_Year", "Month_of_Year"]).reset_index(drop=True)
    df["Month_Index"] = (df["Project_Year"] - 1) * 12 + df["Month_of_Year"]
    return df


@st.cache_data(show_spinner=False)
def score_panel(df: pd.DataFrame, _model, low_c: float, med_c: float, high_c: float) -> pd.DataFrame:
    """Score EVERY project-month row (not just the latest) so we can build
    historical trend charts and early-warning signals (decline, persistence)
    directly from the model, rather than from the raw outcome label."""
    feat_cols = list(_model.feature_names_in_)
    X = df[feat_cols]
    proba = _model.predict_proba(X)  # column 1 = P(Completed)
    out = df.copy()
    out["Completion_Prob"] = proba[:, 1]
    out["Noncompletion_Prob"] = 1 - out["Completion_Prob"]
    out["Risk_Tier"] = out["Completion_Prob"].apply(lambda p: tier_of(p * 100, low_c, med_c, high_c))

    # Reconstruct the linear score (logit) so we can attribute risk to the
    # three predictors for a plain-language explanation (linear model -> no
    # SHAP needed).
    scaler = _model.named_steps["preprocessor"].transformers_[0][1]
    logreg = _model.named_steps["logreg"]
    means, scales, coefs = scaler.mean_, scaler.scale_, logreg.coef_[0]
    Xv = X.to_numpy(dtype=float)
    Z = (Xv - means) / scales
    contrib = Z * coefs
    for i, col in enumerate(feat_cols):
        out[f"z__{col}"] = Z[:, i]
        out[f"contrib__{col}"] = contrib[:, i]
    return out


def tier_of(prob_pct: float, low_c: float, med_c: float, high_c: float) -> str:
    """prob_pct = completion probability, as a percentage (0-100)."""
    if prob_pct >= low_c:
        return "Low"
    elif prob_pct >= med_c:
        return "Medium"
    elif prob_pct >= high_c:
        return "High"
    return "Critical"


def suggested_action(tier: str) -> str:
    return {
        "Critical": "Priority intervention - on-site assessment and financial restructuring review recommended.",
        "High": "Schedule a monitoring visit within 30 days; verify DSCR recovery and equity commitments.",
        "Medium": "Increase monitoring frequency; watch for continued decline in DSCR or utilization.",
        "Low": "Maintain standard monitoring cycle - no immediate action needed.",
    }[tier]


def _phrase_for(col: str, value: float, mean: float) -> str:
    label = FEATURE_LABELS.get(col, col)
    if col == "Debt_Service_Coverage_Ratio":
        if value < 1.0:
            return (
                f"DSCR is critically low at {value:.2f}\u00d7, below the 1.0\u00d7 break-even "
                f"point needed to cover monthly debt payments."
            )
        return f"DSCR is soft at {value:.2f}\u00d7, versus a portfolio-typical {mean:.2f}\u00d7."
    if col == "Proponent_Equity_Share_Pct":
        return f"Owner's equity stake is thin at {value:.1f}%, versus a typical {mean:.1f}%."
    if col == "Capacity_Utilization_Pre_Project_Pct":
        return f"Pre-project capacity utilization is low at {value:.1f}%, versus a typical {mean:.1f}%."
    return f"{label} is below the typical range."


def build_reason(row: pd.Series, feat_cols: list, means: np.ndarray, scales: np.ndarray) -> str:
    if row["Risk_Tier"] == "Low":
        return "No major red flags - all three predictors are in a healthy range."
    items = [(c, row[f"contrib__{c}"], row[c], means[i]) for i, c in enumerate(feat_cols)]
    drivers = sorted([it for it in items if it[1] < 0], key=lambda it: it[1])
    if not drivers:
        return "Flagged by the model, though no single predictor dominates - worth a closer look."
    phrases = [_phrase_for(drivers[0][0], drivers[0][2], drivers[0][3])]
    if len(drivers) > 1 and abs(drivers[0][1]) > 0:
        if abs(drivers[1][1]) >= 0.35 * abs(drivers[0][1]):
            phrases.append(_phrase_for(drivers[1][0], drivers[1][2], drivers[1][3]))
    return " Also, ".join(phrases)


def project_label(row: pd.Series) -> str:
    return str(row["Beneficiary_Name"])


# --------------------------------------------------------------------------
# Deterministic "simulated" intervention-workflow fields
# --------------------------------------------------------------------------
PERSONNEL_POOL = [
    "Engr. R. Villanueva - Provincial MSME Focal",
    "Ms. C. Dumaguit - Monitoring & Evaluation Officer",
    "Mr. J. Bautista - Project Officer",
    "Ms. L. Fernandez - Regional Extension Officer",
    "Engr. P. Aguirre - Technical Monitoring Officer",
    "Ms. K. Salazar - MSME Development Officer",
]
STATUS_POOL_BY_TIER = {
    "Critical": ["Ongoing", "Scheduled", "Not Yet Started"],
    "High": ["Scheduled", "Ongoing", "Not Yet Started"],
    "Medium": ["Not Yet Started", "Scheduled", "Ongoing"],
}
REMARKS_POOL_BY_TIER = {
    "Critical": [
        "Awaiting confirmation of on-site visit schedule.",
        "Coordinating with provincial office for urgent financial review.",
        "Owner requested extension; verification pending.",
    ],
    "High": [
        "Monitoring visit being scheduled with the beneficiary.",
        "DSCR being tracked monthly pending recovery.",
        "Awaiting updated financial statements from proponent.",
    ],
    "Medium": [
        "Routine check-in scheduled with provincial focal.",
        "No major issues reported yet; watching utilization trend.",
        "Equity infusion commitment being verified.",
    ],
}


def _seeded_rng(project_id: str, salt: str = "") -> random.Random:
    h = hashlib.sha256((project_id + salt).encode()).hexdigest()
    return random.Random(int(h[:12], 16))


def simulate_intervention_row(project_id: str, tier: str, today: date) -> dict:
    rng = _seeded_rng(project_id, "intervention")
    status_pool = STATUS_POOL_BY_TIER.get(tier, STATUS_POOL_BY_TIER["Medium"])
    remarks_pool = REMARKS_POOL_BY_TIER.get(tier, REMARKS_POOL_BY_TIER["Medium"])
    offset_days = rng.randint(7, 45)
    return {
        "Suggested Monitoring Action": suggested_action(tier),
        "Intervention Status": rng.choice(status_pool),
        "Assigned Personnel": rng.choice(PERSONNEL_POOL),
        "Follow-up Date": today + timedelta(days=offset_days),
        "Remarks": rng.choice(remarks_pool),
    }


# ==========================================================================
# Sidebar - filters, thresholds, disclaimer
# ==========================================================================
st.sidebar.markdown("## \U0001F3DB\uFE0F ASENXO")
st.sidebar.caption("MSME Project Completion Risk Dashboard")
st.sidebar.markdown("---")

st.sidebar.markdown("### Risk classification (proposed)")
st.sidebar.caption(
    "Configurable cut-offs on **completion probability**. These are a "
    "*proposed / system* classification for this demo - not an official "
    "DOST standard."
)
low_c = st.sidebar.slider("Low risk starts at (\u2265 %)", 50, 95, 75, 1)
med_c = st.sidebar.slider("Medium risk starts at (\u2265 %)", 25, low_c - 1, 50, 1)
high_c = st.sidebar.slider("High risk starts at (\u2265 %)", 0, med_c - 1, 25, 1)
st.sidebar.caption(f"Below {high_c}% completion probability = **Critical**.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")

model = load_model(MODEL_PATH)
raw_df = load_data(DATA_PATH)
scored_panel = score_panel(raw_df, model, low_c, med_c, high_c)

all_provinces = sorted(scored_panel["Province"].dropna().unique().tolist())
all_sectors = sorted(scored_panel["Sector"].dropna().unique().tolist())
all_sizes = sorted(scored_panel["Size_of_Enterprise"].dropna().unique().tolist())

f_province = st.sidebar.multiselect("Province", all_provinces, default=[])
f_sector = st.sidebar.multiselect("Sector", all_sectors, default=[])
f_size = st.sidebar.multiselect("Enterprise size", [s.title() for s in all_sizes], default=[])

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:0.75rem; opacity:0.85;'>"
    "Model: L1 Logistic Regression (3 predictors)<br>"
    "Data: synthetic monthly panel, 321 projects</div>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Latest-record-per-project view (current status), with filters applied
# --------------------------------------------------------------------------
latest_all = scored_panel.groupby("Project_ID", as_index=False).tail(1).reset_index(drop=True)

feat_cols = MODEL_PREDICTORS
scaler = model.named_steps["preprocessor"].transformers_[0][1]
means, scales = scaler.mean_, scaler.scale_
latest_all["Reason"] = latest_all.apply(lambda r: build_reason(r, feat_cols, means, scales), axis=1)
latest_all["Suggested_Action"] = latest_all["Risk_Tier"].map(suggested_action)
latest_all["Size_Title"] = latest_all["Size_of_Enterprise"].str.title()

mask = pd.Series(True, index=latest_all.index)
if f_province:
    mask &= latest_all["Province"].isin(f_province)
if f_sector:
    mask &= latest_all["Sector"].isin(f_sector)
if f_size:
    mask &= latest_all["Size_Title"].isin(f_size)
latest = latest_all[mask].copy()
latest = latest.sort_values("Completion_Prob", ascending=True).reset_index(drop=True)
latest.insert(0, "Rank", np.arange(1, len(latest) + 1))

panel_filtered = scored_panel[scored_panel["Project_ID"].isin(latest["Project_ID"])].copy()

# ==========================================================================
# Header + disclaimer
# ==========================================================================
st.title("MSME Project Completion Risk Management Dashboard")
st.markdown(
    "###### Which MSME projects are at risk of not completing, why, and which need intervention - "
    "an evidence-based view built to support administrators' oversight."
)
st.markdown(
    '<div class="demo-banner">\U0001F6A7 <b>DEMO MODE \u2014 Synthetic Data:</b> This dashboard uses '
    "synthetic data for demonstration. Predictions and province/sector patterns do not represent "
    "actual DOST statistics.</div>",
    unsafe_allow_html=True,
)

if len(latest) == 0:
    st.warning("No projects match the current filters. Adjust the filters in the sidebar.")
    st.stop()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "\U0001F4CA Risk Overview",
        "\u26A0\uFE0F At-Risk Projects",
        "\U0001F50E Project Risk Profile",
        "\U0001F4C8 Risk Analytics",
        "\U0001F6A8 Early Warning",
        "\U0001F6E0\uFE0F Intervention Monitoring",
    ]
)

# ==========================================================================
# TAB 1 - RISK OVERVIEW
# ==========================================================================
with tab1:
    st.subheader("Portfolio risk overview")
    st.markdown(
        '<div class="section-caption">Counts are by unique Project_ID (not monthly rows), '
        "using each project's most recent monthly record.</div>",
        unsafe_allow_html=True,
    )

    n_total = len(latest)
    avg_prob = latest["Completion_Prob"].mean() * 100
    n_at_risk = int((latest["Risk_Tier"] != "Low").sum())
    n_critical = int((latest["Risk_Tier"] == "Critical").sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total projects", f"{n_total}")
    m2.metric("Avg. completion probability", f"{avg_prob:.1f}%")
    m3.metric("At-risk projects", f"{n_at_risk}", f"{n_at_risk / n_total * 100:.0f}% of filtered set")
    m4.metric("Critical projects", f"{n_critical}", f"{n_critical / n_total * 100:.0f}% of filtered set")

    st.markdown("")
    c1, c2 = st.columns([1, 1.3])
    with c1:
        dist = latest["Risk_Tier"].value_counts().reindex(TIER_ORDER).fillna(0).astype(int)
        fig = go.Figure(
            go.Pie(
                labels=dist.index,
                values=dist.values,
                hole=0.55,
                marker=dict(colors=[TIER_COLORS[t] for t in dist.index]),
                sort=False,
            )
        )
        fig.update_layout(
            title="Risk distribution", height=340, margin=dict(t=50, b=10, l=10, r=10), showlegend=True
        )
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("##### Top 5 highest-risk projects")
        top5 = latest.head(5)
        for _, row in top5.iterrows():
            tier = row["Risk_Tier"]
            st.markdown(
                f"""
                <div style="border:1px solid {TIER_COLORS[tier]}33; border-left:6px solid {TIER_COLORS[tier]};
                            background:{TIER_BG[tier]}; border-radius:8px; padding:10px 14px; margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="font-size:0.72rem; color:#666; font-weight:700;">RANK #{row['Rank']}</span><br>
                            <span style="font-size:0.98rem; font-weight:700;">{project_label(row)}</span><br>
                            <span style="font-size:0.78rem; color:#555;">{row['Sector']} \u00b7 {row['Province']} \u00b7 {row['Size_Title']}</span>
                        </div>
                        <div style="text-align:right;">
                            <span style="font-size:1.3rem; font-weight:800; color:{TIER_COLORS[tier]};">{row['Completion_Prob']*100:.1f}%</span><br>
                            <span style="display:inline-block; padding:1px 9px; border-radius:10px; background:{TIER_COLORS[tier]};
                                        color:white; font-size:0.7rem; font-weight:700;">{tier.upper()}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ==========================================================================
# TAB 2 - AT-RISK PROJECTS
# ==========================================================================
with tab2:
    st.subheader("At-risk projects")
    st.markdown(
        '<div class="section-caption">All projects classified Medium, High, or Critical, '
        "ranked by lowest completion probability first. Search or filter to narrow the list.</div>",
        unsafe_allow_html=True,
    )

    at_risk = latest[latest["Risk_Tier"] != "Low"].copy()
    search = st.text_input("Search by beneficiary name or Project ID", "")
    if search:
        s = search.lower()
        at_risk = at_risk[
            at_risk["Beneficiary_Name"].str.lower().str.contains(s)
            | at_risk["Project_ID"].str.lower().str.contains(s)
        ]
    tier_filter = st.multiselect("Risk level", ["Critical", "High", "Medium"], default=[])
    if tier_filter:
        at_risk = at_risk[at_risk["Risk_Tier"].isin(tier_filter)]

    display = pd.DataFrame(
        {
            "Beneficiary": at_risk["Beneficiary_Name"],
            "Project ID": at_risk["Project_ID"],
            "Province": at_risk["Province"],
            "Sector": at_risk["Sector"],
            "Enterprise Size": at_risk["Size_Title"],
            "Project Cost (PhP)": at_risk["Project_Cost"],
            "Completion Probability %": (at_risk["Completion_Prob"] * 100).round(1),
            "Risk Level": at_risk["Risk_Tier"],
            "Suggested Action": at_risk["Suggested_Action"],
        }
    )

    def tier_style(val):
        return f"color:{TIER_COLORS.get(val, '#333')}; font-weight:700;"

    st.dataframe(
        display.style.map(tier_style, subset=["Risk Level"]).format({"Project Cost (PhP)": "{:,.0f}"}),
        hide_index=True,
        width="stretch",
        height=460,
        column_config={
            "Completion Probability %": st.column_config.ProgressColumn(
                "Completion Probability %", min_value=0, max_value=100, format="%.1f%%"
            ),
        },
    )
    st.caption(f"Showing {len(display)} of {len(latest)} filtered projects.")

# ==========================================================================
# TAB 3 - PROJECT RISK PROFILE
# ==========================================================================
with tab3:
    st.subheader("Project risk profile")
    st.markdown(
        '<div class="section-caption">Select a project to inspect its predictors, risk drivers, '
        "and historical trend.</div>",
        unsafe_allow_html=True,
    )

    options = latest.sort_values("Completion_Prob")["Project_ID"].tolist()
    label_map = dict(zip(latest["Project_ID"], latest.apply(lambda r: f"{project_label(r)}  ({r['Project_ID']})", axis=1)))
    sel = st.selectbox("Select project", options, format_func=lambda x: label_map.get(x, x))

    row = latest[latest["Project_ID"] == sel].iloc[0]
    tier = row["Risk_Tier"]

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"#### {project_label(row)}")
        st.caption(row["Project_ID"])
        info_tbl = pd.DataFrame(
            {
                "Field": ["Province", "Sector", "Enterprise Size", "Project Cost (PhP)", "Latest record (project month)"],
                "Value": [
                    row["Province"],
                    row["Sector"],
                    row["Size_Title"],
                    f"{row['Project_Cost']:,.0f}",
                    f"Year {row['Project_Year']}, Month {row['Month_of_Year']} (index {row['Month_Index']})",
                ],
            }
        )
        st.dataframe(info_tbl, hide_index=True, width="stretch")
    with c2:
        st.markdown(
            f"""
            <div style="border:1px solid {TIER_COLORS[tier]}33; border-left:6px solid {TIER_COLORS[tier]};
                        background:{TIER_BG[tier]}; border-radius:8px; padding:16px;">
                <div style="font-size:0.85rem; color:#555;">Completion probability</div>
                <div style="font-size:2rem; font-weight:800; color:{TIER_COLORS[tier]};">{row['Completion_Prob']*100:.1f}%</div>
                <div style="font-size:0.85rem; color:#555; margin-top:6px;">Non-completion probability: {row['Noncompletion_Prob']*100:.1f}%</div>
                <div style="display:inline-block; margin-top:8px; padding:3px 12px; border-radius:12px;
                            background:{TIER_COLORS[tier]}; color:white; font-size:0.78rem; font-weight:700;">
                    {tier.upper()} RISK
                </div>
                <div style="font-size:0.88rem; color:#333; margin-top:10px;"><b>Why flagged:</b> {row['Reason']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("##### Model predictors (latest record)")
    pred_cols = st.columns(3)
    for i, col in enumerate(MODEL_PREDICTORS):
        with pred_cols[i]:
            unit = "\u00d7" if col == "Debt_Service_Coverage_Ratio" else "%"
            st.metric(FEATURE_LABELS[col], f"{row[col]:.2f}{unit}" if unit == "\u00d7" else f"{row[col]:.1f}{unit}")

    hist = scored_panel[scored_panel["Project_ID"] == sel].sort_values("Month_Index")

    st.markdown("##### Historical completion-probability trend")
    fig_p = go.Figure()
    fig_p.add_trace(
        go.Scatter(
            x=hist["Month_Index"],
            y=hist["Completion_Prob"] * 100,
            mode="lines+markers",
            line=dict(color=ACCENT, width=2.5),
            name="Completion probability",
        )
    )
    for i, edge in enumerate([high_c, med_c, low_c]):
        fig_p.add_hline(y=edge, line_dash="dot", line_color="#999", opacity=0.6)
    fig_p.update_layout(
        height=320,
        xaxis_title="Project month (1 = project start)",
        yaxis_title="Completion probability (%)",
        margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_p, width="stretch")

    st.markdown("##### DSCR & capacity utilization trends")
    fig_d = go.Figure()
    fig_d.add_trace(
        go.Scatter(
            x=hist["Month_Index"], y=hist["Debt_Service_Coverage_Ratio"], name="DSCR (\u00d7)",
            line=dict(color=NAVY, width=2), yaxis="y1",
        )
    )
    fig_d.add_trace(
        go.Scatter(
            x=hist["Month_Index"], y=hist["Capacity_Utilization_Pre_Project_Pct"], name="Capacity utilization (%)",
            line=dict(color="#D96C06", width=2), yaxis="y2",
        )
    )
    fig_d.add_hline(y=1.0, line_dash="dash", line_color="#B3261E", annotation_text="DSCR break-even (1.0\u00d7)", yref="y1")
    fig_d.update_layout(
        height=340,
        xaxis_title="Project month (1 = project start)",
        yaxis=dict(title="DSCR (\u00d7)"),
        yaxis2=dict(title="Capacity utilization (%)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.12),
        margin=dict(t=30, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_d, width="stretch")
    st.caption(
        "Proponent equity share is largely time-invariant per project and is shown as a single "
        f"figure above ({row['Proponent_Equity_Share_Pct']:.1f}%) rather than a trend line."
    )

# ==========================================================================
# TAB 4 - RISK ANALYTICS
# ==========================================================================
with tab4:
    st.subheader("Where completion risk concentrates")
    st.markdown(
        '<div class="section-caption">Segment-level view to help target oversight. Province and '
        "Sector tags are illustrative/synthetic (see data notes) - treat patterns as demo-only.</div>",
        unsafe_allow_html=True,
    )

    seg_choice = st.radio(
        "Group by", ["Province", "Sector", "Enterprise Size", "Project Start Year"], horizontal=True
    )

    lat = latest.copy()
    lat["Project Start Year"] = raw_df.groupby("Project_ID")["Calendar_Year"].min().reindex(lat["Project_ID"]).values
    group_col_map = {
        "Province": "Province",
        "Sector": "Sector",
        "Enterprise Size": "Size_Title",
        "Project Start Year": "Project Start Year",
    }
    gcol = group_col_map[seg_choice]

    seg = (
        lat.groupby(gcol)
        .agg(
            Avg_Completion_Prob=("Completion_Prob", "mean"),
            Projects=("Project_ID", "count"),
            At_Risk=("Risk_Tier", lambda s: (s != "Low").sum()),
            Critical=("Risk_Tier", lambda s: (s == "Critical").sum()),
        )
        .reset_index()
    )
    seg["Avg_Completion_Pct"] = seg["Avg_Completion_Prob"] * 100
    seg = seg.sort_values("Avg_Completion_Pct", ascending=True)

    fig_seg = px.bar(
        seg,
        x="Avg_Completion_Pct",
        y=gcol,
        orientation="h",
        text=seg["Avg_Completion_Pct"].round(1).astype(str) + "%",
        hover_data={"Projects": True, "At_Risk": True, "Critical": True, "Avg_Completion_Pct": ":.1f"},
        color="Avg_Completion_Pct",
        color_continuous_scale=["#B3261E", "#D96C06", "#C99A00", "#2E7D32"],
        range_color=[0, 100],
    )
    fig_seg.update_layout(
        height=340,
        xaxis_title="Average completion probability (%)",
        yaxis_title="",
        coloraxis_showscale=False,
        margin=dict(t=10, b=10, l=10, r=10),
    )
    fig_seg.update_traces(textposition="outside")
    st.plotly_chart(fig_seg, width="stretch")

    st.markdown("##### Segment summary")
    seg_display = seg.rename(
        columns={
            gcol: seg_choice,
            "Avg_Completion_Pct": "Avg. Completion Probability %",
            "Projects": "Projects",
            "At_Risk": "At-Risk Projects",
            "Critical": "Critical Projects",
        }
    )[[seg_choice, "Projects", "At-Risk Projects", "Critical Projects", "Avg. Completion Probability %"]]
    st.dataframe(
        seg_display.sort_values("Avg. Completion Probability %").style.format(
            {"Avg. Completion Probability %": "{:.1f}%"}
        ),
        hide_index=True,
        width="stretch",
    )

    st.markdown("---")
    st.markdown("##### Geographic risk view (province-level, illustrative)")
    st.caption(
        "Provincial tags in this dataset are synthetic / illustrative (assigned by keyword match "
        "or seeded random draw) - not verified beneficiary addresses. Coordinates shown are "
        "real province centroids used only to place the illustrative aggregates on a map."
    )
    geo = (
        lat.groupby("Province")
        .agg(Avg_Completion_Pct=("Completion_Prob", lambda s: s.mean() * 100), Projects=("Project_ID", "count"),
             At_Risk=("Risk_Tier", lambda s: (s != "Low").sum()))
        .reset_index()
    )
    geo["lat"] = geo["Province"].map(lambda p: PROVINCE_COORDS.get(p, (None, None))[0])
    geo["lon"] = geo["Province"].map(lambda p: PROVINCE_COORDS.get(p, (None, None))[1])
    geo = geo.dropna(subset=["lat", "lon"])

    if len(geo) > 0:
        fig_map = go.Figure(
            go.Scattermapbox(
                lat=geo["lat"], lon=geo["lon"],
                mode="markers+text",
                marker=dict(
                    size=(geo["At_Risk"].clip(lower=1) * 4 + 10),
                    color=geo["Avg_Completion_Pct"],
                    colorscale=[[0, "#B3261E"], [0.5, "#C99A00"], [1, "#2E7D32"]],
                    cmin=0, cmax=100, showscale=True,
                    colorbar=dict(title="Avg %"),
                ),
                text=geo["Province"],
                textposition="top center",
                hovertext=[
                    f"{p}<br>Projects: {n}<br>At-risk: {a}<br>Avg completion: {v:.1f}%"
                    for p, n, a, v in zip(geo["Province"], geo["Projects"], geo["At_Risk"], geo["Avg_Completion_Pct"])
                ],
                hoverinfo="text",
            )
        )
        fig_map.update_layout(
            mapbox=dict(style="open-street-map", zoom=6.6, center=dict(lat=11.0, lon=122.6)),
            height=430,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_map, width="stretch")
    else:
        st.info("No mappable provinces in the current filtered set.")

# ==========================================================================
# TAB 5 - EARLY WARNING
# ==========================================================================
with tab5:
    st.subheader("Early warning flags")
    st.markdown(
        '<div class="section-caption">Automatic flags derived from the model\'s monthly scores: '
        "current High/Critical status, a declining probability trend, and persistent High/Critical "
        "classification over recent months.</div>",
        unsafe_allow_html=True,
    )

    TREND_WINDOW = 6
    PERSIST_WINDOW = 3
    DECLINE_SLOPE_THRESHOLD = -0.3  # percentage points per month

    def project_warnings(pid: str) -> dict:
        hist = panel_filtered[panel_filtered["Project_ID"] == pid].sort_values("Month_Index")
        cur_row = hist.iloc[-1]
        flags = []

        if cur_row["Risk_Tier"] in ("High", "Critical"):
            flags.append(f"Currently classified {cur_row['Risk_Tier']}")

        recent = hist.tail(TREND_WINDOW)
        if len(recent) >= 3:
            slope = np.polyfit(recent["Month_Index"], recent["Completion_Prob"] * 100, 1)[0]
            if slope <= DECLINE_SLOPE_THRESHOLD:
                flags.append(f"Declining completion probability (\u2248{slope:.2f} pts/month over last {len(recent)} months)")

        last_n = hist.tail(PERSIST_WINDOW)
        if len(last_n) == PERSIST_WINDOW and (last_n["Risk_Tier"].isin(["High", "Critical"])).all():
            flags.append(f"Persistent High/Critical risk for {PERSIST_WINDOW}+ consecutive months")

        return {"flags": flags, "n_flags": len(flags)}

    warn_rows = []
    for pid in latest["Project_ID"]:
        w = project_warnings(pid)
        if w["n_flags"] > 0:
            row = latest[latest["Project_ID"] == pid].iloc[0]
            warn_rows.append(
                {
                    "Beneficiary": row["Beneficiary_Name"],
                    "Project ID": pid,
                    "Risk Level": row["Risk_Tier"],
                    "Completion Probability %": round(row["Completion_Prob"] * 100, 1),
                    "Flags": " | ".join(w["flags"]),
                    "n_flags": w["n_flags"],
                }
            )

    warn_df = pd.DataFrame(warn_rows)
    if len(warn_df) == 0:
        st.success("No early-warning flags triggered for the current filtered set.")
    else:
        warn_df = warn_df.sort_values(["n_flags", "Completion Probability %"], ascending=[False, True]).drop(
            columns="n_flags"
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Projects with \u22651 flag", len(warn_df))
        c2.metric(
            "Declining-trend flags",
            int(warn_df["Flags"].str.contains("Declining").sum()),
        )
        c3.metric(
            "Persistent-risk flags",
            int(warn_df["Flags"].str.contains("Persistent").sum()),
        )
        st.dataframe(
            warn_df.style.map(tier_style, subset=["Risk Level"]),
            hide_index=True,
            width="stretch",
            height=420,
            column_config={
                "Completion Probability %": st.column_config.ProgressColumn(
                    "Completion Probability %", min_value=0, max_value=100, format="%.1f%%"
                ),
            },
        )
        st.caption(
            f"Thresholds used (demo defaults): decline \u2264 {DECLINE_SLOPE_THRESHOLD} pts/month over "
            f"the last {TREND_WINDOW} available months; persistence = {PERSIST_WINDOW} consecutive "
            "months at High/Critical."
        )

# ==========================================================================
# TAB 6 - INTERVENTION MONITORING
# ==========================================================================
with tab6:
    st.subheader("Intervention monitoring")
    st.markdown(
        '<div class="section-caption">System-suggested monitoring workflow for at-risk projects. '
        "Status, personnel, follow-up date, and remarks are <b>simulated for this demo</b> - not real "
        "assignments - and are editable below for illustration only (changes are session-only, not saved).</div>",
        unsafe_allow_html=True,
    )

    at_risk_ids = latest[latest["Risk_Tier"] != "Low"]["Project_ID"].tolist()
    today = date.today()

    if "intervention_state" not in st.session_state:
        st.session_state.intervention_state = {}

    rows = []
    for pid in at_risk_ids:
        row = latest[latest["Project_ID"] == pid].iloc[0]
        if pid not in st.session_state.intervention_state:
            st.session_state.intervention_state[pid] = simulate_intervention_row(pid, row["Risk_Tier"], today)
        sim = st.session_state.intervention_state[pid]
        rows.append(
            {
                "Project ID": pid,
                "Beneficiary": row["Beneficiary_Name"],
                "Province": row["Province"],
                "Risk Level": row["Risk_Tier"],
                "Suggested Monitoring Action": sim["Suggested Monitoring Action"],
                "Intervention Status": sim["Intervention Status"],
                "Assigned Personnel": sim["Assigned Personnel"],
                "Follow-up Date": sim["Follow-up Date"],
                "Remarks": sim["Remarks"],
            }
        )

    if len(rows) == 0:
        st.success("No at-risk projects in the current filtered set - nothing to monitor.")
    else:
        interv_df = pd.DataFrame(rows).sort_values(
            "Risk Level", key=lambda s: s.map({"Critical": 0, "High": 1, "Medium": 2})
        )
        edited = st.data_editor(
            interv_df,
            hide_index=True,
            width="stretch",
            height=460,
            disabled=["Project ID", "Beneficiary", "Province", "Risk Level", "Suggested Monitoring Action"],
            column_config={
                "Intervention Status": st.column_config.SelectboxColumn(
                    options=["Not Yet Started", "Scheduled", "Ongoing", "Completed", "Deferred"]
                ),
                "Follow-up Date": st.column_config.DateColumn(),
            },
            key="intervention_editor",
        )
        # persist edits back into session state
        for _, r in edited.iterrows():
            st.session_state.intervention_state[r["Project ID"]].update(
                {
                    "Suggested Monitoring Action": r["Suggested Monitoring Action"],
                    "Intervention Status": r["Intervention Status"],
                    "Assigned Personnel": r["Assigned Personnel"],
                    "Follow-up Date": r["Follow-up Date"],
                    "Remarks": r["Remarks"],
                }
            )
        st.caption(
            f"{len(interv_df)} at-risk project(s) under monitoring in the current filtered set. "
            "These recommendations are system suggestions, not official DOST decisions."
        )

# ==========================================================================
# Footer
# ==========================================================================
st.markdown("---")
st.caption(
    "Model: L1-regularized Logistic Regression (StandardScaler \u2192 LogisticRegression) using "
    "exactly three predictors - Debt-Service Coverage Ratio, Proponent Equity Share %, and "
    "Pre-Project Capacity Utilization %. Province, Sector, Enterprise Size, and Project Cost are "
    "reference/segmentation fields only, not model inputs. "
    "DEMO MODE \u2014 Synthetic Data: predictions and patterns above do not represent actual DOST statistics."
)
