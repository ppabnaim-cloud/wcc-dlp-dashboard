# app.py
# WCC DLP Real-Time Dashboard — Streamlit + Plotly
# (with automated statistical descriptions / narrative insights)

import re
from datetime import datetime, date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
import seaborn as sns  # optional; keep if you plan to add seaborn charts later
import streamlit as st
from matplotlib import pyplot as plt  # optional
from pathlib import Path
import base64
import streamlit.components.v1 as components

# ---- soft import scikit-learn (so app still runs if missing) ----
SKLEARN_OK = True
SKLEARN_ERR = ""
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LinearRegression
except Exception as e:
    SKLEARN_OK = False
    SKLEARN_ERR = str(e)

# --------------------------
# CONFIG
# --------------------------
DASHBOARD_TITLE_LONG = (
    "Study on AI-Driven Dashboard Model for Integrated Service Monitoring, Defect Liability Period Tracking (DLP) & Risk Management for New MOH Facility"
)
st.set_page_config(page_title=DASHBOARD_TITLE_LONG, layout="wide")

# KPI label styling
st.markdown(
    """
<style>
.kpi-label{
  text-align:center; font-weight:700; font-size:22px;
  margin: 0 0 -10px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

DEFAULT_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vTZOoHpBpGvk2VPjHh1s2FZwvvVwSRikO0DZ5r3sng-GyqFa5-mX8v98pkq9t6OIY23zTxIF_oBGJaa/pub?gid=1204103743&single=true&output=csv"
)

DLP_EXPIRY = date(2026, 11, 22)
MY_TZ = pytz.timezone("Asia/Kuala_Lumpur")

# Cache-buster to avoid stale Google "publish to web" caching
def cache_bust_url(url: str, bucket_seconds: int = 15) -> str:
    ts = int(datetime.now(MY_TZ).timestamp() // bucket_seconds)
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}cb={ts}"

# Column synonyms (robust mapper)
CANON = {
    "level": ["level", "aras", "floor"],
    "department": ["department", "unit", "dept"],
    "location": ["location", "lokasi", "room", "space", "area", "room_name", "nama_ruang"],
    "last_modified": ["last_modified", "last modified", "modified", "updated", "last update", "last_updated"],
    "operational_status": ["operational_status", "operation status", "status_operational", "status"],
    # Defect categories (totals)
    "mechanical": ["mechanical defect", "total mechanical defect", "mechanical", "jumlah laporan kerosakan mekanikal"],
    "electrical": ["electrical defect", "total electrical defect", "electrical", "jumlah laporan kerosakan elektrikal"],
    "public": ["public defect", "civil defect", "awam defect", "public", "civil", "awam", "jumlah laporan kerosakan awam"],
    "biomedical": ["biomedical defect", "biomedical", "jumlah laporan kerosakan biomedical"],
    "ict": ["ict defect", "ict", "jumlah laporan kerosakan ict"],
    # Pending categories
    "pending_mechanical": ["pending mechanical", "mechanical pending", "mechanical not rectified", "mechanical unrepaired", "mechanical belum dibaikpulih"],
    "pending_electrical": ["pending electrical", "electrical pending", "electrical not rectified", "electrical unrepaired", "electrical belum dibaikpulih"],
    "pending_public": ["pending public", "pending civil", "pending awam", "public pending", "civil pending", "awam pending"],
    "pending_biomedical": ["pending biomedical", "biomedical pending"],
    "pending_ict": ["pending ict", "ict pending"],
    # Severity / disruption
    "major_defect": ["major defect", "major_defect", "major"],
    "recurrent_defect": ["recurrent defect", "recurrent_defect", "recurrent"],
    "clinical_disruption_days": [
        "clinical service disruption due to major or recurrent defects (days)",
        "clinical service disruption (days)",
        "service disruption (days)",
        "gangguan servis klinikal (days)",
        "disruption_days",
    ],
    # Optional pre-computed total
    "total_defect": [
        "total defect", "total_defect",
        "jumlah laporan kerosakan", "jumlah_laporan_kerosakan",
        "total kerosakan", "total_kerosakan",
    ],
    # ➕ NEW COLUMNS
    "incidence_report": [
        "incidence report", "incident report", "incidence", "incident",
        "laporan insiden", "laporan kejadian"
    ],
    "financial_rm": [
        "financial implication (rm)", "financial implication",
        "financial impact (rm)", "financial impact",
        "kos (rm)", "cost (rm)", "anggaran kos (rm)",
        "kos_kewangan", "financial_implication_rm", "financial_rm"
    ],
    # BCMS/BIA related
    "svc_name": ["service", "service_name", "svc", "critical_service", "perkhidmatan_kritikal"],
    "risk_owner": ["risk owner", "service owner", "owner", "pemilik_risiko"],
    "rto_hours": ["rto (hours)", "rto_hours", "rto_hour", "rto_jam"],
    "mtpd_hours": ["mtpd (hours)", "mtpd_hours", "mtpd_hour", "mtpd_jam"],
    "bia_flag": ["bia complete", "bia done", "bia flag", "bia_fields_ok"],
    "bcp_last_test": ["bcp last test", "bcp_last_test", "tarikh ujian bcp"],
    "bcp_next_due": ["bcp next due", "bcp_next_due", "bcp next test", "tarikh bcp seterusnya"],
    "bcp_overdue": ["bcp overdue", "bcp_overdue", "bcp test overdue"],
    "workaround": ["workaround", "contingency plan", "pelan kontingensi"],
    "upstream_dep": ["upstream dependency", "upstream_dep", "dependency", "kebergantungan"],
    "risk_appetite": ["risk appetite", "risk_appetite", "ambang risiko"],
}

# --------------------------
# Utilities
# --------------------------
def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]+", "_", regex=True)
    )
    return df

def _norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w]+", "_", s)
    return s

def find_first_match(df_cols, candidates):
    cols = list(df_cols)
    ncols = {col: _norm(col) for col in cols}
    for cand in candidates:
        c = _norm(cand)
        # exact match
        for col, ncol in ncols.items():
            if ncol == c:
                return col
        # whole-word match
        pattern = rf"(^|_){re.escape(c)}(_|$)"
        for col, ncol in ncols.items():
            if re.search(pattern, ncol):
                return col
    return None

def map_columns(df: pd.DataFrame):
    return {key: find_first_match(df.columns, cands) for key, cands in CANON.items()}

@st.cache_data(ttl=60)
def load_data(csv_url: str) -> pd.DataFrame:
    df = pd.read_csv(csv_url)
    df = sanitize_columns(df)
    df = df.dropna(how="all")
    return df

def safe_num(x):
    """Robust numeric coercion (handles commas, blanks)."""
    try:
        if isinstance(x, pd.DataFrame):
            return x.apply(lambda s: pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce"))
        if isinstance(x, (pd.Series, pd.Index)):
            return pd.to_numeric(x.astype(str).str.replace(",", "", regex=False), errors="coerce")
        return pd.to_numeric(pd.Series(x), errors="coerce")
    except Exception:
        return pd.Series(np.nan, index=getattr(x, "index", None))

def safe_money(series: pd.Series) -> pd.Series:
    """Convert strings like 'RM 1,234.50' or '1,000' to float (RM)."""
    if series is None:
        return pd.Series(dtype="float64")
    s = series.astype(str).str.replace(",", "", regex=False)
    s = s.str.replace(r"[^\d\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

# Excel-style cell & letter helpers
# NOTE: lower TTL for fresher KPI cell reads
@st.cache_data(ttl=5)
def read_csv_cell(csv_url: str, row_1based: int, col_letter: str):
    def col_letter_to_index(col: str) -> int:
        col = col.strip().upper()
        n = 0
        for ch in col:
            n = n * 26 + (ord(ch) - 64)  # 'A'->1
        return n - 1  # 0-based
    try:
        raw = pd.read_csv(csv_url, header=None)  # raw grid; no header munging
        r = row_1based - 1
        c = col_letter_to_index(col_letter)
        val = raw.iloc[r, c]
        v = pd.to_numeric(str(val).replace(",", ""), errors="coerce")
        return None if pd.isna(v) else float(v)
    except Exception:
        return None

def get_col_by_letter(df_sanitized: pd.DataFrame, letter: str) -> str | None:
    """Return sanitized column name by sheet letter (A=1)."""
    idx = 0
    for ch in letter.strip().upper():
        idx = idx * 26 + (ord(ch) - 64)
    idx = idx - 1  # 0-based
    if 0 <= idx < df_sanitized.shape[1]:
        return df_sanitized.columns[idx]
    return None

# ---------- Narrative helpers ----------
def _fmt_top(df, name_col, val_col, fmt="{:.0f}", k=3):
    return ", ".join([f"{row[name_col]} ({fmt.format(row[val_col])})" for _, row in df.head(k).iterrows()])

def insight_defects_vs_lastmod(m: pd.DataFrame, loc_col: str, corr_value: float):
    if m is None or m.empty:
        return
    n = len(m); r = float(corr_value) if pd.notna(corr_value) else float("nan")
    top3_def = m.sort_values("total_defects_loc", ascending=False).head(3)
    top3_recent = m.sort_values("days_since_last_mod", ascending=True).head(3)
    p_text = ""
    try:
        from scipy import stats
        if np.isfinite(r) and n > 2 and abs(r) < 1:
            t_stat = r * np.sqrt((n - 2) / max(1e-9, 1 - r**2))
            p = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))
            p_text = f", p={p:.3f}"
    except Exception:
        pass
    st.markdown(
        f"**Auto-insights:** {n} locations. Mean defects/location "
        f"**{m['total_defects_loc'].mean():.1f}** (median {m['total_defects_loc'].median():.1f}). "
        f"Correlation between days since last modified and defects is **{r:.2f}**{p_text}. "
        f"Top defects: {_fmt_top(top3_def, loc_col, 'total_defects_loc')}. "
        f"Most recently modified: {_fmt_top(top3_recent, loc_col, 'days_since_last_mod', fmt='{:.0f} days')}."
    )

def insight_risk_register(grid: pd.DataFrame, reg: pd.DataFrame, hi: pd.DataFrame, hi_threshold: int, loc_col: str, dept_col: str):
    cell = grid.stack().astype(int) if isinstance(grid, pd.DataFrame) else pd.Series(dtype=int)
    if not cell.empty:
        (imp_hot, like_hot) = cell.idxmax(); hot_val = int(cell.max())
        hot_text = f"Heatmap hotspot: Impact **{imp_hot}** × Likelihood **{like_hot}** with **{hot_val}** items."
    else:
        hot_text = "Heatmap has no populated cells."
    id_cols = [c for c in [loc_col, dept_col] if c]
    top5 = reg.sort_values(["risk_max", "risk_mean"], ascending=False).head(5)
    st.markdown(
        f"**Auto-insights:** {hot_text} "
        f"High-risk groups (threshold {hi_threshold}) count: **{len(hi)}**."
    )
    st.caption("Top 5 groups by Max then Mean risk")
    st.dataframe(top5[id_cols + ["risk_max", "risk_mean", "impact_mean", "like_mean"]],
                 use_container_width=True, height=220)

def insight_prediction(loc_risk: pd.DataFrame, auc: float, acc: float, thr: float, loc_col: str):
    if loc_risk is None or loc_risk.empty:
        return
    nloc = len(loc_risk); mean_risk = float(loc_risk["risk_proba"].mean())
    top3 = loc_risk.sort_values("risk_proba", ascending=False).head(3)
    hi_n = int((loc_risk["risk_proba"] >= thr).sum())
    top_txt = ", ".join([f"{r[loc_col]} ({r['risk_proba']:.2f})" for _, r in top3.iterrows()])
    st.markdown(
        f"**Auto-insights:** Model AUC **{auc:.3f}**, accuracy **{acc:.3f}**. "
        f"Across **{nloc}** locations, mean predicted risk **{mean_risk:.2f}**. "
        f"Locations above alert threshold {thr:.2f}: **{hi_n}**. "
        f"Top risks: {top_txt}."
    )

def _pct(n, d):
    return 0.0 if not d or d == 0 else (n / d * 100.0)

def insight_grouped_stacked(g: pd.DataFrame, group_col: str, cat_col: str, val_col: str, label_group: str):
    if g is None or g.empty:
        return
    total = float(g[val_col].sum())
    by_group = g.groupby(group_col, dropna=False)[val_col].sum().sort_values(ascending=False)
    by_cat   = g.groupby(cat_col,   dropna=False)[val_col].sum().sort_values(ascending=False)
    top_cell = g.sort_values(val_col, ascending=False).head(3)
    top_groups_txt = ", ".join([f"{idx} ({int(v)})" for idx, v in by_group.head(3).items()])
    top_cats_txt   = ", ".join([f"{idx} ({int(v)})" for idx, v in by_cat.head(3).items()])
    cells_txt      = ", ".join([f"{r[group_col]}×{r[cat_col]} ({int(r[val_col])})" for _, r in top_cell.iterrows()])
    st.caption(
        f"**Auto-insights:** Total **{int(total):,}**. Top {label_group.lower()}s: {top_groups_txt}. "
        f"Top categories: {top_cats_txt}. Largest cells: {cells_txt}."
    )

def insight_pending_grouped(g: pd.DataFrame, group_col: str, type_col: str, val_col: str, label_group: str):
    if g is None or g.empty:
        return
    total = float(g[val_col].sum())
    by_group = g.groupby(group_col, dropna=False)[val_col].sum().sort_values(ascending=False)
    by_type  = g.groupby(type_col,   dropna=False)[val_col].sum().sort_values(ascending=False)
    st.caption(
        f"**Auto-insights:** Pending total **{int(total):,}**. "
        f"Top {label_group.lower()}: {', '.join([f'{k} ({int(v)})' for k, v in by_group.head(3).items()])}. "
        f"Most pending types: {', '.join([f'{k} ({int(v)})' for k, v in by_type.head(3).items()])}."
    )

def insight_pie_status(pie_df: pd.DataFrame, name_col="Status", count_col="Count"):
    if pie_df is None or pie_df.empty:
        return
    total = float(pie_df[count_col].sum())
    pct = pie_df.copy()
    pct["pct"] = pct[count_col].apply(lambda x: _pct(x, total))
    top3 = pct.sort_values("pct", ascending=False).head(3)
    top_txt = ", ".join([f"{r[name_col]} ({r['pct']:.1f}%)" for _, r in top3.iterrows()])
    st.caption(f"**Auto-insights:** Total **{int(total):,}** items. Status mix top: {top_txt}.")

def insight_severity_table(tbl: pd.DataFrame, loc_col: str):
    if tbl is None or tbl.empty:
        return
    cols = [c for c in tbl.columns if c != loc_col]
    lines = []
    for c in cols:
        top3 = tbl.sort_values(c, ascending=False).head(3)
        txt = ", ".join([f"{r[loc_col]} ({int(r[c]) if pd.notna(r[c]) else 0})" for _, r in top3.iterrows()])
        lines.append(f"{c}: {txt}")
    st.caption("**Auto-insights:** " + " | ".join(lines))

def insight_regression(xname, yname, r, slope=None, r2=None):
    pieces = []
    if pd.notna(r):
        pieces.append(f"Pearson r **{r:.2f}**")
    if slope is not None and r2 is not None:
        pieces.append(f"Regression slope **{slope:.3f}** (R² **{r2:.2f}**)")
    if pieces:
        st.caption("**Auto-insights:** " + " | ".join(pieces) + f" for {yname} vs {xname}.")

# ---- NEW BCMS insight helpers ----
def insight_bcms_risk(reg: pd.DataFrame, risk_appetite: int):
    """Narrative for BCMS risk register (Impact × Likelihood)."""
    if reg is None or reg.empty:
        return

    id_cols = [c for c in reg.columns if c not in ["risk_max", "risk_mean", "impact_mean", "like_mean"]]
    label = "group" if len(id_cols) > 1 else (id_cols[0] if id_cols else "group")

    total_groups = len(reg)
    mean_risk = float(reg["risk_mean"].mean())
    hi = reg[reg["risk_max"] >= risk_appetite]
    hi_n = len(hi)

    def _name(r):
        if not id_cols:
            return "N/A"
        return " / ".join(str(r[c]) for c in id_cols)

    top3 = hi.sort_values(["risk_max", "risk_mean"], ascending=False).head(3)
    top_txt = ", ".join([f"{_name(r)} ({int(r['risk_max'])})" for _, r in top3.iterrows()]) if len(top3) else "None"

    st.caption(
        f"**Auto-insights (BCMS Risk):** {total_groups} {label}s in register. "
        f"Mean risk score **{mean_risk:.1f}**. "
        f"High-risk (≥{risk_appetite}) {label}s: **{hi_n}**. "
        f"Top high-risk {label}s: {top_txt}."
    )

def insight_bcms_overview(n_services: int, bia_ok: int, risk_breach: int,
                          rto_breach: int, bcp_overdue: int):
    """Narrative for BCMS overview KPI row."""
    if n_services <= 0:
        return
    pct_bia = _pct(bia_ok, n_services)
    st.caption(
        f"**Auto-insights (BCMS Overview):** Tracking **{n_services}** critical services. "
        f"BIA completed for **{bia_ok}** services ({pct_bia:.1f}%). "
        f"Risk score above appetite in **{risk_breach}** services. "
        f"Current RTO breaches: **{rto_breach}**. "
        f"BCP tests overdue: **{bcp_overdue}**."
    )

def insight_heatmap_simple(grid: pd.DataFrame, row_label: str, col_label: str):
    """Generic narrative for heatmaps (Service/Dept × Risk band)."""
    if grid is None or grid.empty:
        return

    total = int(grid.to_numpy().sum())
    flat = grid.stack().reset_index(name="Count")
    top = flat.sort_values("Count", ascending=False).head(3)
    cells_txt = ", ".join(
        f"{r[row_label]} × {r[col_label]} ({int(r['Count'])})"
        for _, r in top.iterrows()
    )

    st.caption(
        f"**Auto-insights (Heatmap):** Total **{total:,}** risk cells across all bands. "
        f"Top hotspots by {row_label} × {col_label}: {cells_txt}."
    )

# --------------------------
# SIDEBAR
# --------------------------
st.sidebar.title(DASHBOARD_TITLE_LONG)
csv_url = st.sidebar.text_input("CSV URL", value=DEFAULT_CSV_URL)

secs = st.sidebar.slider("Auto-refresh every (seconds)", 10, 180, 60)
try:
    from streamlit_autorefresh import st_autorefresh as auto_refresh
    if st.sidebar.checkbox("Enable auto-refresh", value=True):
        auto_refresh(interval=secs * 1000, key="auto_refresh")
except Exception:
    st.sidebar.caption("Tip: `pip install streamlit-autorefresh` to enable timed refresh. Cache TTL is 60s by default.")

if st.sidebar.button("Force refresh data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.write("---")
st.sidebar.write("Filters appear after data loads.")
AUTO_INSIGHTS = st.sidebar.checkbox("Show auto-insights (narrative)", value=True)
USE_FINANCE_IN_IMPACT = st.sidebar.checkbox("Blend Financial (RM) into BCMS Impact", value=True)

if not SKLEARN_OK:
    st.sidebar.warning(
        "Risk model disabled: scikit-learn not available in this Python. "
        "Use Python 3.12 and `pip install scikit-learn` to enable it."
    )

# --- BCMS Impact settings (Finance blending) ---
st.sidebar.write("---")
st.sidebar.subheader("BCMS Impact settings")

USE_FINANCE_IN_IMPACT = st.sidebar.checkbox(
    "Blend Financial (RM) into BCMS Impact (use worst-of days vs RM)",
    value=True
)

FIN_METHOD = st.sidebar.selectbox(
    "Financial banding method",
    ["Quantiles (auto, 5 bins)", "Fixed thresholds (RM)"],
    index=0,
    help="Quantiles adapts to data distribution. Fixed thresholds are policy-based."
)

if FIN_METHOD == "Fixed thresholds (RM)":
    st.sidebar.caption("Set RM thresholds for bands 1→5:")
    T1 = st.sidebar.number_input("T1 (to band 2)", value=1_000.0, step=500.0, min_value=0.0, format="%.0f")
    T2 = st.sidebar.number_input("T2 (to band 3)", value=5_000.0, step=500.0, min_value=T1, format="%.0f")
    T3 = st.sidebar.number_input("T3 (to band 4)", value=20_000.0, step=1_000.0, min_value=T2, format="%.0f")
    T4 = st.sidebar.number_input("T4 (to band 5)", value=100_000.0, step=5_000.0, min_value=T3, format="%.0f")
else:
    T1 = T2 = T3 = T4 = None  # not used in quantile mode

def band_finance(fin_series: pd.Series, method: str, t1=None, t2=None, t3=None, t4=None) -> pd.Series:
    fin = safe_money(fin_series).fillna(0.0)
    labels = [1, 2, 3, 4, 5]
    if method == "Quantiles (auto, 5 bins)":
        try:
            fb = pd.qcut(fin, 5, labels=labels, duplicates="drop")
            fb = fb.astype("Int64").fillna(1)
        except Exception:
            fb = pd.cut(
                fin,
                bins=[0, 1_000, 5_000, 20_000, 100_000, float("inf")],
                labels=labels,
                right=False,
                include_lowest=True
            ).astype("Int64").fillna(1)
    else:
        bins = [0, t1, t2, t3, t4, float("inf")]
        fb = pd.cut(
            fin,
            bins=bins,
            labels=labels,
            right=False,
            include_lowest=True
        ).astype("Int64").fillna(1)
    return fb.astype(int)

# --- BCMS thresholds / alert settings ---
st.sidebar.write("---")
st.sidebar.subheader("BCMS Risk thresholds")

BCMS_RISK_APPETITE = st.sidebar.slider(
    "Risk appetite threshold (Impact × Likelihood)",
    min_value=5, max_value=25, value=15, step=1,
)

DISRUPT_ALERT_DAYS = st.sidebar.number_input(
    "Alert if disruption days ≥",
    min_value=0, value=1, step=1,
    help="Used for BCMS watchlist & disruption alerts."
)

# --------------------------
# LOAD
# --------------------------
csv_url_fresh = cache_bust_url(csv_url)

with st.spinner("Loading data…"):
    df = load_data(csv_url_fresh)

colmap = map_columns(df)

# Fallback mappings by absolute sheet letters per your instruction:
DEPT_LETTER = get_col_by_letter(df, "D")
INC_LETTER  = get_col_by_letter(df, "G")

# Canonical helpers (prefer sheet letters first)
LEVEL  = colmap["level"]
DEPT   = DEPT_LETTER or colmap["department"]
LOC    = colmap["location"]
LMOD   = colmap["last_modified"]
OPSTAT = colmap["operational_status"]
INC    = INC_LETTER  or colmap.get("incidence_report")
FIN    = colmap.get("financial_rm")

with st.sidebar.expander("Incidence/Dept columns (debug)"):
    st.write("DEPT (D):", DEPT)
    st.write("INC (G):", INC)

cat_cols = {
    "mechanical": colmap["mechanical"],
    "electrical": colmap["electrical"],
    "public": colmap["public"],
    "biomedical": colmap["biomedical"],
    "ict": colmap["ict"],
}
pending_cols = {
    "mechanical": colmap["pending_mechanical"],
    "electrical": colmap["pending_electrical"],
    "public": colmap["pending_public"],
    "biomedical": colmap["pending_biomedical"],
    "ict": colmap["pending_ict"],
}

# TOTAL_DEFECT
TOTAL_DEFECT = colmap["total_defect"]
if TOTAL_DEFECT is None:
    found = [c for c in cat_cols.values() if c is not None]
    if found:
        df["total_defect_calc"] = df[found].apply(safe_num).fillna(0).sum(axis=1)
        TOTAL_DEFECT = "total_defect_calc"

# PENDING_TOTAL
pending_found = [c for c in pending_cols.values() if c is not None]
if pending_found:
    df["pending_total_calc"] = df[pending_found].apply(safe_num).fillna(0).sum(axis=1)
    PENDING_TOTAL = "pending_total_calc"
else:
    PENDING_TOTAL = None

# --- Precompute totals once (KPI & Debug) ---
total_from_col = (
    safe_num(df[TOTAL_DEFECT]).fillna(0).sum()
    if TOTAL_DEFECT and TOTAL_DEFECT in df.columns else np.nan
)

found_cat_cols = [c for c in cat_cols.values() if c and c in df.columns]
_seen = set()
found_cat_cols = [c for c in found_cat_cols if not (c in _seen or _seen.add(c))]

total_from_cats = (
    df[found_cat_cols].apply(safe_num).fillna(0).sum(axis=1).sum()
    if found_cat_cols else np.nan
)

with st.sidebar.expander("Debug totals"):
    st.write("TOTAL_DEFECT mapped to:", TOTAL_DEFECT)
    st.write("Category cols used:", found_cat_cols)
    st.write("Sum(TOTAL_DEFECT col):", None if np.isnan(total_from_col) else float(total_from_col))
    st.write("Sum(categories):", None if np.isnan(total_from_cats) else float(total_from_cats))

# --------------------------
# HEADER & KPIs
# --------------------------
tz_now = datetime.now(MY_TZ)
st.markdown(
    f"""
<div style="text-align:center; padding-top:0.25rem; padding-bottom:0.5rem;">
  <h1 style="margin:0">{DASHBOARD_TITLE_LONG}</h1>
  <div style="color:#6c757d; margin-top:0.25rem;">
    Live at {tz_now.strftime('%Y-%m-%d %H:%M:%S %Z')}
  </div>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)

# KPI 1: Days to DLP
with c1:
    st.markdown("<div class='kpi-label'>Days to DLP Expiry (22 November 2026)</div>", unsafe_allow_html=True)
    days_left = (DLP_EXPIRY - tz_now.date()).days + 1  # inclusive
    kpi = go.Figure(go.Indicator(mode="number", value=days_left, number={"font": {"size": 88}}))
    kpi.update_layout(height=160, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(kpi, use_container_width=True, config={"displayModeBar": False})

# KPI 2: Total Defects strictly from CSV cell Y94
with c2:
    st.markdown("<div class='kpi-label'>Total Defects (sum)</div>", unsafe_allow_html=True)
    cell_val = read_csv_cell(csv_url_fresh, row_1based=94, col_letter="Y")
    if cell_val is None:
        if not np.isnan(total_from_col) and total_from_col > 0:
            cell_val = float(total_from_col)
        elif not np.isnan(total_from_cats):
            cell_val = float(total_from_cats)
        else:
            cell_val = 0.0
    kpi2 = go.Figure(go.Indicator(mode="number", value=int(round(cell_val)), number={"font": {"size": 88}}))
    kpi2.update_layout(height=160, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(kpi2, use_container_width=True, config={"displayModeBar": False})

# --------------------------
# FLOWCHART IMAGE/PDF
# --------------------------
st.markdown("## Flowcharts")

APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
PDF_EXTS = {".pdf"}
RECURSIVE = False  # toggle if you want to scan subfolders too

def find_files(roots, exts, recursive=False):
    found = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        it = root.rglob("*") if recursive else root.glob("*")
        for p in it:
            if p.is_file() and p.suffix.lower() in exts:
                found.append(p)
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)  # newest first
    seen = set(); out = []
    for p in found:
        r = p.resolve()
        if r in seen:
            continue
        seen.add(r)
        out.append(p)
    return out

def show_image_compat(path: str):
    try:
        st.image(path, use_container_width=True)
    except TypeError:
        st.image(path, use_column_width=True)

def first_img_with_keywords(files, keywords=("pts", "complaint", "aduan")):
    for p in files:
        if any(k in p.name.lower() for k in keywords):
            return p
    return None

def prefer_dlp_user_flow(files):
    for p in files:
        if p.name.lower() == "dlp_wcc_user_flowchart.jpg":
            return p
    pri = [p for p in files if "pts" not in p.stem.lower() and any(k in p.stem.lower() for k in ("dlp", "wcc", "user", "flow"))]
    return pri[0] if pri else (files[0] if files else None)

def prefer_pts_flow(files):
    hit = first_img_with_keywords(files, keywords=("pts", "complaint", "aduan"))
    return hit or (files[0] if files else None)

def prefer_pdf(files):
    pri = [p for p in files if any(k in p.name.lower() for k in ("flow", "chart", "dlp", "pts"))]
    return pri[0] if pri else (files[0] if files else None)

def show_pdf_inline(pdf_path: Path, height: int = 700):
    with pdf_path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    html = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}"></iframe>'
    components.html(html, height=height)

img_candidates = find_files([ASSETS_DIR, APP_DIR], IMG_EXTS, RECURSIVE)
pdf_candidates = find_files([ASSETS_DIR, APP_DIR], PDF_EXTS, RECURSIVE)
default_img_left = prefer_dlp_user_flow(img_candidates)
default_pdf_right = prefer_pdf(pdf_candidates)

st.sidebar.write("### Flowchart picker")
img_label_map = {(p.parent.name + "/" + p.name): p for p in img_candidates}
pdf_label_map = {(p.parent.name + "/" + p.name): p for p in pdf_candidates}
img_labels = ["(auto)"] + list(img_label_map.keys())
pdf_labels = ["(auto)"] + list(pdf_label_map.keys())

img_choice_left = st.sidebar.selectbox("Image (JPG/PNG) for LEFT panel", img_labels, index=0)
pdf_choice_right = st.sidebar.selectbox("PDF for RIGHT panel", pdf_labels, index=0)

img_path_left = img_label_map.get(img_choice_left, default_img_left)
pdf_path_right = pdf_label_map.get(pdf_choice_right, default_pdf_right)

c_left, c_right = st.columns(2)
with c_left:
    st.markdown("**DLP WCC User Flowchart**")
    if img_path_left and img_path_left.exists():
        show_image_compat(str(img_path_left))
        st.caption(f"Showing: {img_path_left.parent.name}/{img_path_left.name}")
    else:
        st.info("No image found in `assets/` or app folder. Drop `dlp_wcc_user_flowchart.jpg` into `assets/`.")

with c_right:
    st.markdown("**PTS Complaint Flow (within DLP)**")
    if pdf_path_right and pdf_path_right.exists():
        show_pdf_inline(pdf_path_right, height=700)
        st.caption(f"Showing: {pdf_path_right.parent.name}/{pdf_path_right.name}")
    else:
        pts_img = prefer_pts_flow(img_candidates)
        if pts_img and pts_img.exists():
            show_image_compat(str(pts_img))
            st.caption(f"Showing: {pts_img.parent.name}/{pts_img.name}")
        else:
            st.info("No PTS flow image/PDF found in `assets/` or app folder.")

# --------------------------
# FILTERS
# --------------------------
st.markdown("### Filters")
f1, f2, f3 = st.columns(3)
picked_levels = f1.multiselect(
    "Level",
    sorted(df[LEVEL].dropna().astype(str).unique()) if LEVEL else [],
    default=None if not LEVEL else sorted(df[LEVEL].dropna().astype(str).unique()),
)
picked_depts  = f2.multiselect(
    "Department",
    sorted(df[DEPT].dropna().astype(str).unique()) if DEPT else [],
    default=None if not DEPT else sorted(df[DEPT].dropna().astype(str).unique()),
)
picked_stats  = f3.multiselect(
    "Operational Status",
    sorted(df[OPSTAT].dropna().astype(str).unique()) if OPSTAT else [],
    default=None if not OPSTAT else sorted(df[OPSTAT].dropna().astype(str).unique()),
)

mask = pd.Series(True, index=df.index)
if LEVEL  and picked_levels:
    mask &= df[LEVEL].astype(str).isin(picked_levels)
if DEPT   and picked_depts:
    mask &= df[DEPT].astype(str).isin(picked_depts)
if OPSTAT and picked_stats:
    mask &= df[OPSTAT].astype(str).isin(picked_stats)
dff = df.loc[mask].copy()

# --------------------------
# 1) Number of defect complaints according to level 1 until level 10
# --------------------------
st.markdown("## 1) Number of Defect Complaints According to Level 1 until Level 10")

if TOTAL_DEFECT and LEVEL:
    g_lvl = (
        dff.groupby(LEVEL, dropna=False)[TOTAL_DEFECT]
           .apply(lambda s: safe_num(s).fillna(0).sum())
           .reset_index()
    )
    g_lvl["_level_num"] = pd.to_numeric(g_lvl[LEVEL], errors="coerce")

    fig_lvl = px.bar(g_lvl, x=LEVEL, y=TOTAL_DEFECT, title=None, text=TOTAL_DEFECT)
    fig_lvl.update_layout(xaxis_title="Level", yaxis_title="Total Defects")
    fig_lvl.update_traces(textposition="outside")

    possible = [1, 3, 5, 7, 9]
    have = [v for v in possible if v in g_lvl["_level_num"].dropna().unique().tolist()]
    if have:
        fig_lvl.update_xaxes(
            tickmode="array",
            tickvals=[g_lvl.loc[g_lvl["_level_num"] == v, LEVEL].iloc[0] for v in have]
        )

    st.plotly_chart(fig_lvl, use_container_width=True)

    if AUTO_INSIGHTS and len(g_lvl):
        total = float(g_lvl[TOTAL_DEFECT].sum())
        mean = float(g_lvl[TOTAL_DEFECT].mean())
        med  = float(g_lvl[TOTAL_DEFECT].median())
        top3 = g_lvl.sort_values(TOTAL_DEFECT, ascending=False).head(3)
        top_txt = ", ".join([f"{r[LEVEL]} ({int(r[TOTAL_DEFECT])})" for _, r in top3.iterrows()])
        st.caption(
            f"**Auto-insights:** Across {len(g_lvl)} levels, total defects **{int(total):,}** "
            f"(mean {mean:.1f}, median {med:.1f}). Top levels: {top_txt}."
        )
else:
    st.info("Need LEVEL + TOTAL_DEFECT to plot.")

st.write("---")

# --------------------------
# 2) Number of defect complaints according department
# --------------------------
st.markdown("## 2) Number of Defect Complaints According Department")

if TOTAL_DEFECT and DEPT:
    g_dept = (
        dff.groupby(DEPT, dropna=False)[TOTAL_DEFECT]
           .apply(lambda s: safe_num(s).fillna(0).sum())
           .reset_index()
    )

    fig_dept = px.bar(g_dept, x=DEPT, y=TOTAL_DEFECT, title=None, text=TOTAL_DEFECT)
    fig_dept.update_layout(xaxis_title="Department", yaxis_title="Total Defects")
    fig_dept.update_traces(textposition="outside")

    st.plotly_chart(fig_dept, use_container_width=True)

    if AUTO_INSIGHTS and len(g_dept):
        total = float(g_dept[TOTAL_DEFECT].sum())
        mean = float(g_dept[TOTAL_DEFECT].mean())
        med  = float(g_dept[TOTAL_DEFECT].median())
        top3 = g_dept.sort_values(TOTAL_DEFECT, ascending=False).head(3)
        top_txt = ", ".join([f"{r[DEPT]} ({int(r[TOTAL_DEFECT])})" for _, r in top3.iterrows()])
        pct_top1 = (top3.iloc[0][TOTAL_DEFECT] / total * 100) if total > 0 else 0
        st.caption(
            f"**Auto-insights:** Total defects by department **{int(total):,}** "
            f"(mean {mean:.1f}, median {med:.1f}). Top departments: {top_txt}. "
            f"Highest contributor accounts for **{pct_top1:.1f}%** of all defects."
        )
else:
    st.info("Need DEPARTMENT + TOTAL_DEFECT to plot.")

st.write("---")

st.write("---")

# --------------------------
# 3) Defect Categories - Pie Chart Distribution
# --------------------------
st.markdown("## 3) Defect Categories - Pie Chart Distribution")

# Define category colors (needed for pie chart)
category_colors = {
    'Mechanical': '#FF6B6B',    # Red
    'Electrical': '#FFA500',    # Orange
    'Public': '#4ECDC4',        # Teal
    'Biomedical': '#95E1D3',    # Light Green
    'Ict': '#A569BD'            # Purple
}

# Get available category columns
found_cats = {k: v for k, v in cat_cols.items() if v is not None}

if found_cats:
    category_totals = []
    
    for cat_name, col_name in found_cats.items():
        if col_name in dff.columns:
            total = safe_num(dff[col_name]).fillna(0).sum()
            if total > 0:  # Only include categories with defects
                category_totals.append({
                    'Category': cat_name.title(),
                    'Total Defects': float(total)
                })
    
    if category_totals:
        cat_df = pd.DataFrame(category_totals)
        
        # Create pie chart
        fig_pie = px.pie(
            cat_df,
            names='Category',
            values='Total Defects',
            title="Defect Category Distribution (%)",
            color='Category',
            color_discrete_map=category_colors,
            hole=0.4  # Donut chart style
        )
        
        fig_pie.update_traces(
            textposition='outside',
            textinfo='label+percent+value',
            texttemplate='%{label}<br>%{value:,.0f}<br>(%{percent})'
        )
        
        fig_pie.update_layout(height=500)
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        if AUTO_INSIGHTS:
            total = cat_df['Total Defects'].sum()
            largest = cat_df.sort_values('Total Defects', ascending=False).iloc[0]
            largest_pct = (largest['Total Defects'] / total * 100)
            
            st.caption(
                f"**Pie chart insight:** **{largest['Category']}** dominates with **{largest_pct:.1f}%** "
                f"of all defects. This visualization shows the relative proportion of each category."
            )
    else:
        st.info("No defects found in any category.")

else:
    st.info("Need category columns for pie chart.")

st.write("---")

# --------------------------
# 4) Defect Category According to Level 1 until Level 10
# --------------------------
st.markdown("## 4) Defect Category According to Level 1 until Level 10")
found_cats = {k: v for k, v in cat_cols.items() if v is not None}
if LEVEL and found_cats:
    melt_cols = list(found_cats.values())
    tmp = dff[[LEVEL] + melt_cols].copy().melt(id_vars=[LEVEL], var_name="CategoryCol", value_name="Count")
    tmp["Count"] = safe_num(tmp["Count"]).fillna(0)
    inv = {v: k.title() for k, v in found_cats.items()}
    tmp["Category"] = tmp["CategoryCol"].map(inv).fillna(tmp["CategoryCol"])
    g = tmp.groupby([LEVEL, "Category"], dropna=False)["Count"].sum().reset_index()
    st.plotly_chart(
        px.bar(g, x=LEVEL, y="Count", color="Category", barmode="group", title="Defect Categories by Level"),
        use_container_width=True,
    )
    if AUTO_INSIGHTS:
        insight_grouped_stacked(g, LEVEL, "Category", "Count", label_group="Level")
else:
    st.info("Need LEVEL + at least one category column.")


# --------------------------
# 5) Defect Category According to Department
# --------------------------

st.markdown("## 5) Defect Category According to Department")
if DEPT and found_cats:
    melt_cols = list(found_cats.values())
    tmp = dff[[DEPT] + melt_cols].copy().melt(id_vars=[DEPT], var_name="CategoryCol", value_name="Count")
    tmp["Count"] = safe_num(tmp["Count"]).fillna(0)
    inv = {v: k.title() for k, v in found_cats.items()}
    tmp["Category"] = tmp["CategoryCol"].map(inv).fillna(tmp["CategoryCol"])
    g = tmp.groupby([DEPT, "Category"], dropna=False)["Count"].sum().reset_index()
    st.plotly_chart(
        px.bar(g, x=DEPT, y="Count", color="Category", barmode="group", title="Defect Categories by Department"),
        use_container_width=True,
    )
    if AUTO_INSIGHTS:
        insight_grouped_stacked(g, DEPT, "Category", "Count", label_group="Department")
else:
    st.info("Need DEPARTMENT + at least one category column.")

# --------------------------
# 6) Pending Defects (Category Breakdown (from cells)
# --------------------------
st.markdown("## 6) Pending Defects by Category")

pending_mech = read_csv_cell(csv_url_fresh, row_1based=94, col_letter="L") or 0
pending_elec = read_csv_cell(csv_url_fresh, row_1based=94, col_letter="O") or 0
pending_pub  = read_csv_cell(csv_url_fresh, row_1based=94, col_letter="R") or 0
pending_bio  = read_csv_cell(csv_url_fresh, row_1based=94, col_letter="U") or 0
pending_ict  = read_csv_cell(csv_url_fresh, row_1based=94, col_letter="X") or 0

pending_data = pd.DataFrame({
    "PendingType": ["Pending Mechanical", "Pending Electrical", "Pending Public", "Pending Biomedical", "Pending ICT"],
    "Pending Count": [pending_mech, pending_elec, pending_pub, pending_bio, pending_ict]
})

fig_pending = px.bar(
    pending_data,
    x="PendingType",
    y="Pending Count",
    text="Pending Count",
    title="Pending Defects (from Row 94 Cells)",
    color="PendingType"
)
fig_pending.update_traces(textposition="outside")
fig_pending.update_layout(xaxis_title="Pending Type", yaxis_title="Pending Count")
st.plotly_chart(fig_pending, use_container_width=True)

total_pending = sum(pending_data["Pending Count"])
top3 = pending_data.sort_values("Pending Count", ascending=False).head(3)
top_txt = ", ".join([f"{r['PendingType']} ({int(r['Pending Count'])})" for _, r in top3.iterrows()])
st.caption(
    f"**Auto-insights:** Total pending defects **{int(total_pending):,}**. "
    f"Top categories: {top_txt}."
)

# --------------------------
# 7A) Operational Status Pie
# --------------------------
st.markdown("## 6A) Operational Status — Percentage")
if OPSTAT:
    counts = dff[OPSTAT].astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown"}).value_counts(dropna=False)
    pie_df = counts.rename_axis("Status").reset_index(name="Count")
    
    # Define custom color mapping
    color_map = {
        "Pending": "#FF0000",      # Red
        "pending": "#FF0000",      # Red (lowercase)
        "Rotation": "#0000FF",   # Blue
        "rotation": "#0000FF",   # Blue (lowercase)
        "Active": "#00FF00",  # Green
        "Active": "#00FF00",  # Green (lowercase)
    }
    
    # Create pie chart with custom colors
    fig = px.pie(
        pie_df, 
        names="Status", 
        values="Count", 
        title="Operational Status (%)", 
        hole=0.35,
        color="Status",
        color_discrete_map=color_map
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if AUTO_INSIGHTS:
        insight_pie_status(pie_df, name_col="Status", count_col="Count")
else:
    st.info("Need OPERATIONAL STATUS column.")
    
# --------------------------
# 7B) Defects by Location & Operational Status
# --------------------------
st.markdown("## 6B) Defects by Location & Operational Status (Rotation vs Pending)")

if LOC and OPSTAT and TOTAL_DEFECT:
    # Filter for Rotation and Pending only
    status_filter = dff[OPSTAT].astype(str).str.lower().str.strip().isin(['rotation', 'pending'])
    filtered = dff[status_filter].copy()
    
    if not filtered.empty:
        # Prepare data
        filtered['_total_def'] = safe_num(filtered[TOTAL_DEFECT]).fillna(0)
        filtered['_status_clean'] = filtered[OPSTAT].astype(str).str.strip()
        
        # Group by location and status
        g = (
            filtered.groupby([LOC, '_status_clean'], dropna=False)['_total_def']
            .sum()
            .reset_index()
        )
        
        # Create stacked bar chart
        fig = px.bar(
            g, 
            x=LOC, 
            y='_total_def', 
            color='_status_clean',
            title="Defects by Location (Rotation vs Pending)",
            labels={'_total_def': 'Total Defects', '_status_clean': 'Status'},
            text='_total_def',
            color_discrete_map={
                'Pending': '#FF0000',
                'pending': '#FF0000',
                'Rotation': '#0000FF',
                'rotation': '#0000FF'
            }
        )
        fig.update_traces(textposition='inside')
        fig.update_layout(xaxis_title="Location", yaxis_title="Total Defects", barmode='stack')
        st.plotly_chart(fig, use_container_width=True)
        
        if AUTO_INSIGHTS:
            total_rot = g[g['_status_clean'].str.lower() == 'rotation']['_total_def'].sum()
            total_pend = g[g['_status_clean'].str.lower() == 'pending']['_total_def'].sum()
            n_rot = len(g[g['_status_clean'].str.lower() == 'rotation'])
            n_pend = len(g[g['_status_clean'].str.lower() == 'pending'])
            
            st.caption(
                f"**Auto-insights:** **{n_rot}** rotation locations with **{int(total_rot):,}** defects. "
                f"**{n_pend}** pending locations with **{int(total_pend):,}** defects. "
                f"Total: **{int(total_rot + total_pend):,}** defects across rotation/pending locations."
            )
    else:
        st.info("No locations with 'Rotation' or 'Pending' status found.")
else:
    st.info("Need LOCATION, OPERATIONAL STATUS, and TOTAL_DEFECT columns.")

st.write("---")

# --------------------------
# 6C) Rotation & Pending Locations - Detailed Table
# --------------------------
st.markdown("## 6C) Rotation & Pending Locations - Detailed View")

if LOC and OPSTAT:
    # Filter for Rotation and Pending
    status_filter = dff[OPSTAT].astype(str).str.lower().str.strip().isin(['rotation', 'pending'])
    filtered = dff[status_filter].copy()
    
    if not filtered.empty:
        # Prepare columns to display
        display_cols = [LOC, OPSTAT]
        if DEPT: display_cols.append(DEPT)
        if LEVEL: display_cols.append(LEVEL)
        if TOTAL_DEFECT: display_cols.append(TOTAL_DEFECT)
        if PENDING_TOTAL: display_cols.append(PENDING_TOTAL)
        
        # Add disruption column if it exists (get it from colmap)
        disrupt_col = colmap.get("clinical_disruption_days")
        if disrupt_col and disrupt_col in filtered.columns:
            display_cols.append(disrupt_col)
        
        # Get unique columns
        display_cols = list(dict.fromkeys([c for c in display_cols if c in filtered.columns]))
        
        view = filtered[display_cols].copy()
        
        # Clean status column
        view[OPSTAT] = view[OPSTAT].astype(str).str.strip()
        
        # Sort by status (Pending first) then by total defects
        if TOTAL_DEFECT in view.columns:
            view['_sort_def'] = safe_num(view[TOTAL_DEFECT]).fillna(0)
            view = view.sort_values([OPSTAT, '_sort_def'], ascending=[True, False])
            view = view.drop('_sort_def', axis=1)
        
        st.dataframe(view, use_container_width=True, height=400)
        
        if AUTO_INSIGHTS:
            n_rot = (filtered[OPSTAT].astype(str).str.lower().str.strip() == 'rotation').sum()
            n_pend = (filtered[OPSTAT].astype(str).str.lower().str.strip() == 'pending').sum()
            st.caption(
                f"**Auto-insights:** Showing **{len(filtered)}** locations "
                f"(**{n_rot}** rotation, **{n_pend}** pending)."
            )
    else:
        st.info("No locations with 'Rotation' or 'Pending' status found.")
else:
    st.info("Need LOCATION and OPERATIONAL STATUS columns.")

st.write("---")

# --------------------------
# 7) Service Disruption by Location — Days
# --------------------------
st.markdown("## 7) Service Disruption by Location — Days")

DISRUPT_COL = colmap.get("clinical_disruption_days")

if LOC and DISRUPT_COL and (DISRUPT_COL in dff.columns):
    tmp = dff[[LOC, DISRUPT_COL]].copy()
    tmp["_days"] = pd.to_numeric(
        tmp[DISRUPT_COL].astype(str)
            .str.replace(",", "", regex=False)
            .str.extract(r"(-?\d+\.?\d*)", expand=False),
        errors="coerce"
    ).fillna(0)

    g = (
        tmp.groupby(LOC, dropna=False)["_days"]
           .sum()
           .reset_index()
           .sort_values("_days", ascending=False)
    )
    g = g[g["_days"] > 0]

    if g.empty:
        st.info("No service disruptions recorded (> 0 days).")
    else:
        def _wrap_label(s: str, width: int = 28) -> str:
            words = str(s).split()
            lines, line = [], []
            for w in words:
                if len(" ".join(line + [w])) <= width:
                    line.append(w)
                else:
                    lines.append(" ".join(line)); line = [w]
            if line: lines.append(" ".join(line))
            return "<br>".join(lines)

        PER_ROW = 3
        for i in range(0, len(g), PER_ROW):
            cols = st.columns(PER_ROW)
            for (j, (_, row)) in enumerate(g.iloc[i:i+PER_ROW].iterrows()):
                with cols[j]:
                    fig = go.Figure(go.Indicator(
                        mode="number",
                        value=float(row["_days"]),
                        number={"font": {"size": 76}},
                        title={"text": f"<b>{_wrap_label(row[LOC])}</b>", "font": {"size": 16}},
                    ))
                    fig.update_layout(height=200, margin=dict(l=0, r=0, t=60, b=0))
                    fig.add_annotation(
                        x=0.5, y=0.12, xref="paper", yref="paper",
                        text="<span style='color:#6c757d;'>Days of Disruption</span>",
                        showarrow=False, align="center", xanchor="center", yanchor="bottom",
                        font=dict(size=12)
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        total_days = int(g["_days"].sum())
        top3 = ", ".join([f"{r[LOC]} ({int(r['_days'])})" for _, r in g.head(3).iterrows()])
        st.caption(
            f"**Auto-insights:** {len(g)} locations affected. "
            f"Cumulative disruption **{total_days} days**. Top: {top3}."
        )
else:
    st.info("Need LOCATION and a numeric 'Clinical Disruption (Days)' column.")

# --------------------------
# 8) Clinical Incidence Report Related to Defects by Department — SUM of column G
# --------------------------
st.markdown("## 8) Clinical Incidence Report Related to Defect by Department (Σ of column G)")

if DEPT and INC and (DEPT in dff.columns) and (INC in dff.columns):
    inc_num = safe_num(dff[INC]).fillna(0)
    g = (
        dff.assign(_inc=inc_num)
           .groupby(DEPT, dropna=False)["_inc"]
           .sum()
           .reset_index()
           .sort_values("_inc", ascending=False)
    )

    with st.expander("Debug: incidence sum by department (top 10)"):
        st.dataframe(g.head(10), use_container_width=True)

    fig = px.bar(
        g, x=DEPT, y="_inc", text="_inc",
        title="Incidence Reports by Department (sum of column G)"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="Department", yaxis_title="Incidence Report (sum)")
    st.plotly_chart(fig, use_container_width=True)

    if AUTO_INSIGHTS:
        total = int(g["_inc"].sum())
        top3 = ", ".join([f"{r[DEPT]} ({int(r['_inc'])})" for _, r in g.head(3).iterrows()])
        st.caption(f"**Auto-insights:** Total **{total}**. Top departments: {top3}.")
else:
    st.info("Need Department (column D) and Incidence Report (column G).")

# --------------------------
# 9) BCMS Risk Scoring (Impact × Likelihood) & BIA
# --------------------------
st.markdown("## 9) BCMS Risk Scoring (Impact × Likelihood) & BIA")

# --- METHODOLOGY EXPLANATION (New) ---
with st.expander("ℹ️  Click to see Risk Scoring Methodology", expanded=True):
    st.markdown("""
    **Risk Score = Impact × Likelihood** (Score ranges from 1 to 25)
    
    | Band | Impact (Severity) <br> *Measured by Disruption Days* | Likelihood (Frequency) <br> *Measured by Defect Count* |
    | :---: | :--- | :--- |
    | **1** | **Minimal** (0 days) | **Rare** (0 - 1 defects) |
    | **2** | **Low** (1 - 2 days) | **Occasional** (2 - 5 defects) |
    | **3** | **Moderate** (3 - 6 days) | **Frequent** (6 - 15 defects) |
    | **4** | **High** (7 - 13 days) | **Very Frequent** (16 - 40 defects) |
    | **5** | **Critical** (14+ days) | **Constant** (> 40 defects) |
    """)

IMPACT_DAYS_COL = colmap.get("clinical_disruption_days")
SVC_COL         = colmap.get("svc_name")
OWNER_COL       = colmap.get("risk_owner")
RTO_COL         = colmap.get("rto_hours")
MTPD_COL        = colmap.get("mtpd_hours")
BIA_FLAG_COL    = colmap.get("bia_flag")
BCP_LAST_COL    = colmap.get("bcp_last_test")
BCP_NEXT_COL    = colmap.get("bcp_next_due")
BCP_OVER_COL    = colmap.get("bcp_overdue")
WORKAROUND_COL  = colmap.get("workaround")
UPDEP_COL       = colmap.get("upstream_dep")
RISK_APP_COL    = colmap.get("risk_appetite")

# Service label
if SVC_COL and SVC_COL in dff.columns:
    dff["_svc"] = dff[SVC_COL].astype(str)
elif LOC and LOC in dff.columns:
    dff["_svc"] = dff[LOC].astype(str)
elif DEPT and DEPT in dff.columns:
    dff["_svc"] = dff[DEPT].astype(str)
else:
    dff["_svc"] = "N/A"

# Owner label
if OWNER_COL and OWNER_COL in dff.columns:
    dff["_owner"] = dff[OWNER_COL].astype(str)
elif DEPT and DEPT in dff.columns:
    dff["_owner"] = dff[DEPT].astype(str)
else:
    dff["_owner"] = ""

# Disruption days Extraction
if IMPACT_DAYS_COL and IMPACT_DAYS_COL in dff.columns:
    dff["_disrupt_days"] = pd.to_numeric(
        dff[IMPACT_DAYS_COL].astype(str)
            .str.replace(",", "", regex=False)
            .str.extract(r"(-?\d+\.?\d*)", expand=False),
        errors="coerce"
    ).fillna(0)
else:
    dff["_disrupt_days"] = 0

# --- UPDATED: Impact Banding Logic (Fixed Thresholds) ---
# 0 days = Band 1
# 1-2 days = Band 2
# 3-6 days = Band 3
# 7-13 days = Band 4
# 14+ days = Band 5
def band_days_fixed(series: pd.Series) -> pd.Series:
    # Use right=False: [0, 1) -> 0; [1, 3) -> 1,2; [3, 7) -> 3..6; [7, 14) -> 7..13; [14, inf) -> 14+
    bins = [-float("inf"), 1, 3, 7, 14, float("inf")]
    labels = [1, 2, 3, 4, 5]
    return pd.cut(series, bins=bins, labels=labels, right=False).astype(int)

impact_days_band = band_days_fixed(dff["_disrupt_days"])

# Financial impact blend (if enabled)
impact_fin_band = (
    band_finance(dff[FIN], FIN_METHOD, T1, T2, T3, T4)
    if (USE_FINANCE_IN_IMPACT and FIN and FIN in dff.columns)
    else pd.Series(1, index=dff.index, dtype=int)
)

IMPACT_BAND = np.maximum(impact_days_band, impact_fin_band) if USE_FINANCE_IN_IMPACT else impact_days_band

# --- UPDATED: Likelihood Banding Logic (Fixed Thresholds) ---
# 0-1 defects = Band 1
# 2-5 defects = Band 2
# 6-15 defects = Band 3
# 16-40 defects = Band 4
# 40+ (>40) defects = Band 5

# 1. Get Defect Counts
if TOTAL_DEFECT and TOTAL_DEFECT in dff.columns:
    tmp_freq = safe_num(dff[TOTAL_DEFECT]).fillna(0)
else:
    melt_cols = [c for c in cat_cols.values() if c and c in dff.columns]
    tmp_freq = dff[melt_cols].apply(safe_num).fillna(0).sum(axis=1) if melt_cols else pd.Series(0, index=dff.index)

# 2. Group Frequency (if mapped by location/dept)
group_key = [c for c in [LOC, DEPT] if c]
if group_key:
    # Group by location/dept to get the total defects for that ENTITY, not just that row
    freq_by_grp = dff.groupby(group_key, dropna=False).apply(
        lambda g: safe_num(g[TOTAL_DEFECT]).fillna(0).sum() if TOTAL_DEFECT in g.columns else 0
    )
    # Map back to original dataframe size
    freq_map = dff[group_key].merge(
        freq_by_grp.rename("grp_freq"),
        left_on=group_key,
        right_index=True,
        how="left"
    )["grp_freq"].fillna(0).values
    freq_series = pd.Series(freq_map, index=dff.index)
else:
    freq_series = tmp_freq

# 3. Apply Fixed Bands
def band_freq_fixed(series: pd.Series) -> pd.Series:
    # Use right=True: (-0.1, 1] -> 0,1; (1, 5] -> 2..5; (5, 15] -> 6..15; (15, 40] -> 16..40; (40, inf) -> >40
    bins = [-0.1, 1, 5, 15, 40, float("inf")]
    labels = [1, 2, 3, 4, 5]
    return pd.cut(series, bins=bins, labels=labels, right=True).astype(int)

LIKELIHOOD_BAND = band_freq_fixed(freq_series)

# --- RISK CALCULATION ---
RISK_SCORE = (IMPACT_BAND * LIKELIHOOD_BAND).astype(int)
dff["_impact_band"] = IMPACT_BAND
dff["_likelihood_band"] = LIKELIHOOD_BAND
dff["_risk_score"] = RISK_SCORE

# BIA flag logic
if BIA_FLAG_COL and BIA_FLAG_COL in dff.columns:
    s_bia = dff[BIA_FLAG_COL].astype(str).str.lower().str.strip()
    dff["_bia_flag"] = s_bia.isin(["1", "yes", "y", "true", "completed", "done"]).astype(int)
else:
    bia_cols = [c for c in [RTO_COL, MTPD_COL, OWNER_COL, WORKAROUND_COL] if c and c in dff.columns]
    dff["_bia_flag"] = (
        dff[bia_cols].notna().any(axis=1).astype(int) if bia_cols else pd.Series(0, index=dff.index, dtype=int)
    )

# RTO / MTPD gaps and breaches
if RTO_COL and RTO_COL in dff.columns:
    dff["_rto_h"] = safe_num(dff[RTO_COL]).fillna(0)
else:
    dff["_rto_h"] = 0

if MTPD_COL and MTPD_COL in dff.columns:
    dff["_mtpd_h"] = safe_num(dff[MTPD_COL]).fillna(0)
else:
    dff["_mtpd_h"] = 0

disrupt_hours = dff["_disrupt_days"] * 24.0
dff["_gap_rto_h"] = disrupt_hours - dff["_rto_h"]
dff["_gap_mtpd_h"] = disrupt_hours - dff["_mtpd_h"]
dff["_rto_breach"] = (dff["_rto_h"] > 0) & (disrupt_hours > dff["_rto_h"])
dff["_mtpd_breach"] = (dff["_mtpd_h"] > 0) & (disrupt_hours > dff["_mtpd_h"])

# Risk appetite breach flag
dff["_risk_breach"] = dff["_risk_score"] >= BCMS_RISK_APPETITE

# BCP overdue
if BCP_OVER_COL and BCP_OVER_COL in dff.columns:
    s_over = dff[BCP_OVER_COL].astype(str).str.lower().str.strip()
    dff["_bcp_overdue"] = s_over.isin(["1", "yes", "y", "true"]).astype(bool)
else:
    if BCP_NEXT_COL and BCP_NEXT_COL in dff.columns:
        try:
            next_dates = pd.to_datetime(dff[BCP_NEXT_COL], errors="coerce")
            dff["_bcp_overdue"] = next_dates < pd.to_datetime(tz_now.date())
        except Exception:
            dff["_bcp_overdue"] = False
    else:
        dff["_bcp_overdue"] = False

# BCMS risk register (per service/location/department)
id_cols = [c for c in [SVC_COL, LOC, DEPT] if c]
if id_cols:
    reg = (
        dff[id_cols + ["_impact_band", "_likelihood_band", "_risk_score", "_disrupt_days"]]
        .groupby(id_cols, dropna=False)
        .agg(
            risk_max=("_risk_score", "max"),
            risk_mean=("_risk_score", "mean"),
            impact_mean=("_impact_band", "max"), # Take Max impact for conservative risk rating
            like_mean=("_likelihood_band", "max"), # Take Max likelihood
            total_disruption=("_disrupt_days", "sum")
        )
        .reset_index()
        .sort_values(["risk_max", "risk_mean"], ascending=False)
    )
    
    # Highlight high risk rows
    def highlight_risk(row):
        val = row["risk_max"]
        if val >= 20:
            return ['background-color: #ffcccb'] * len(row) # Red
        elif val >= 15:
            return ['background-color: #ffe5b4'] * len(row) # Orange
        elif val >= 10:
            return ['background-color: #ffffe0'] * len(row) # Yellow
        return [''] * len(row)

    st.dataframe(reg.style.apply(highlight_risk, axis=1), use_container_width=True, height=400)
    
    if AUTO_INSIGHTS:
        insight_bcms_risk(reg, BCMS_RISK_APPETITE)
        st.caption(f"**Methodology Note:** Bands are calculated based on Max Disruption Days (Impact) and Total Defect Count (Likelihood) per location.")
else:
    st.info("BCMS risk register needs at least one of: Service, Location, or Department columns.")

st.write("---")

# --------------------------
# 10) Chi-Square Test: Defect Category Independence by Department
# --------------------------

st.markdown("## 10) Chi-Square Test: Defect Category Independence by Department")
if DEPT and found_cats:
    # Prepare data
    cat_dept = []
    for cat_name, col_name in found_cats.items():
        if col_name in dff.columns:
            counts = dff.groupby(DEPT)[col_name].apply(lambda x: safe_num(x).sum())
            for dept, count in counts.items():
                cat_dept.extend([dept] * int(count))  # Replicate rows
    
    # Create contingency table
    contingency = pd.crosstab(
        index=pd.Series(cat_dept, name='Department'),
        columns=pd.Series([cat for cat in found_cats.keys() for _ in range(int(dff[found_cats[cat]].sum()))], name='Category')
    )
    
    # Perform chi-square test
    from scipy.stats import chi2_contingency
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    # Calculate Cramér's V
    n = contingency.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
    
    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Chi-Square Statistic", f"{chi2:.2f}")
    with col2:
        st.metric("p-value", f"{p_value:.4f}")
    with col3:
        st.metric("Cramér's V (Effect Size)", f"{cramers_v:.3f}")
    
    # Generate Auto Insights
    if p_value < 0.05:
        st.success(f"✅ **SIGNIFICANT ASSOCIATION FOUND** (p < 0.05)")
        
        # Effect size interpretation
        if cramers_v < 0.1:
            effect = "negligible"
        elif cramers_v < 0.3:
            effect = "small"
        elif cramers_v < 0.5:
            effect = "medium"
        else:
            effect = "large"
        
        st.caption(
            f"Defect categories are NOT independent of departments. "
            f"Certain departments show distinct defect patterns. "
            f"Effect size Cramér's V = {cramers_v:.3f} ({effect} effect)."
        )
        
        # AUTO INSIGHTS SECTION
        st.markdown("### 🔍 Auto-Generated Insights")
        
        # Calculate standardized residuals for insight generation
        observed = contingency.values
        standardized_residuals = (observed - expected) / np.sqrt(expected)
        
        # Find significant patterns (|residual| > 2 indicates significance)
        insights = []
        
        for i, dept in enumerate(contingency.index):
            for j, cat in enumerate(contingency.columns):
                residual = standardized_residuals[i, j]
                obs_count = observed[i, j]
                exp_count = expected[i, j]
                
                if abs(residual) > 2:  # Significant deviation
                    if residual > 2:
                        direction = "**significantly higher**"
                        emoji = "⬆️"
                        color = "red"
                    else:
                        direction = "**significantly lower**"
                        emoji = "⬇️"
                        color = "blue"
                    
                    pct_diff = ((obs_count - exp_count) / exp_count) * 100
                    insights.append({
                        'dept': dept,
                        'cat': cat,
                        'direction': direction,
                        'emoji': emoji,
                        'obs': obs_count,
                        'exp': exp_count,
                        'pct': pct_diff,
                        'residual': abs(residual),
                        'color': color
                    })
        
        # Sort by residual strength
        insights.sort(key=lambda x: x['residual'], reverse=True)
        
        if insights:
            st.markdown("**Key Patterns Detected:**")
            
            for idx, insight in enumerate(insights[:5], 1):  # Show top 5 insights
                st.markdown(
                    f"{insight['emoji']} **{insight['dept']}** has {insight['direction']} "
                    f"defects in **{insight['cat']}**: "
                    f"{int(insight['obs'])} observed vs {insight['exp']:.1f} expected "
                    f"({insight['pct']:+.1f}%)"
                )
            
            # Summary recommendation
            st.markdown("---")
            st.markdown("**💡 Recommendation:**")
            
            # Find department with most deviations
            dept_deviation_counts = {}
            for insight in insights:
                dept = insight['dept']
                dept_deviation_counts[dept] = dept_deviation_counts.get(dept, 0) + 1
            
            if dept_deviation_counts:
                focus_dept = max(dept_deviation_counts, key=dept_deviation_counts.get)
                st.info(
                    f"🎯 Focus quality improvement efforts on **{focus_dept}** which shows "
                    f"the most distinctive defect patterns ({dept_deviation_counts[focus_dept]} categories deviate from expected). "
                    f"Investigate root causes specific to this department's processes or conditions."
                )
        else:
            st.info("While statistically significant, all department-category combinations are within normal variation ranges (no extreme outliers).")
    
    else:
        st.info("ℹ️ **No significant association detected** (p ≥ 0.05)")
        st.caption(
            "Defect categories appear to be distributed independently across departments. "
            "This suggests defects are not strongly influenced by department-specific factors."
        )
        
        st.markdown("### 🔍 Auto-Generated Insights")
        st.markdown(
            "**Interpretation:** Defect patterns are relatively uniform across all departments. "
            "This indicates:\n"
            "- Quality issues are likely systemic rather than department-specific\n"
            "- Common root causes may affect all departments equally\n"
            "- Department-level interventions may not be the most effective approach"
        )
    
    # Heatmap of observed vs expected frequencies
    fig = px.imshow(
        contingency,
        labels=dict(x="Defect Category", y="Department", color="Count"),
        title="Observed Defect Distribution (Heatmap)",
        text_auto=True,
        color_continuous_scale="YlOrRd"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Optional: Show standardized residuals heatmap for deeper analysis
    with st.expander("📊 Advanced: Standardized Residuals (Click to Expand)"):
        st.caption("Values > 2 or < -2 indicate significant deviations from expected frequencies")
        
        residuals_df = pd.DataFrame(
            standardized_residuals,
            index=contingency.index,
            columns=contingency.columns
        )
        
        fig_residuals = px.imshow(
            residuals_df,
            labels=dict(x="Defect Category", y="Department", color="Std. Residual"),
            title="Standardized Residuals (Deviation from Expected)",
            text_auto='.2f',
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            zmin=-4,
            zmax=4
        )
        st.plotly_chart(fig_residuals, use_container_width=True)
        
        st.caption(
            "🔴 Red (positive): More defects than expected | "
            "🔵 Blue (negative): Fewer defects than expected | "
            "⚪ White (near 0): As expected"
        )

# --------------------------
# 11) AI Predictive Modeling (Linear Regression & Gradient Boosting)
# --------------------------
st.markdown("## 11) AI Predictive Modeling")

if SKLEARN_OK:
    st.info("Models are trained on the full dataset (ignoring current sidebar filters) to ensure sufficient data size.")

    # --- 1. PREPARATION: Clean Data for ML ---
    feature_map = {
        "Mechanical": colmap["mechanical"],
        "Electrical": colmap["electrical"],
        "Public": colmap["public"],
        "Biomedical": colmap["biomedical"],
        "ICT": colmap["ict"]
    }
    
    valid_feats = {k: v for k, v in feature_map.items() if v and v in df.columns}
    
    if len(valid_feats) < 2:
        st.warning("Not enough defect category columns found to run AI models.")
    else:
        # Prepare X (Features)
        X = df[list(valid_feats.values())].apply(safe_num).fillna(0)
        
        # Clean Disruption Days for Target (y)
        disrupt_col = colmap.get("clinical_disruption_days")
        if disrupt_col and disrupt_col in df.columns:
            y_days = pd.to_numeric(
                df[disrupt_col].astype(str).str.replace(",", "").str.extract(r"(-?\d+\.?\d*)", expand=False),
                errors='coerce'
            ).fillna(0)
        else:
            y_days = None

        tab_reg, tab_class = st.tabs(["📉 Linear Regression (Predict Disruption)", "🌳 Gradient Boosting (Classify High Risk)"])

        # --- A) LINEAR REGRESSION ---
        with tab_reg:
            st.markdown("### Predict Clinical Disruption Days")
            st.caption("This model attempts to predict how many days of disruption occur based on the number of defects in each category.")
            
            if y_days is not None and y_days.sum() > 0:
                # Train
                lr = LinearRegression()
                lr.fit(X, y_days)
                y_pred = lr.predict(X)
                r2 = lr.score(X, y_days)

                # Coefficients
                coef_df = pd.DataFrame({
                    "Defect Category": list(valid_feats.keys()),
                    "Impact (Coefficient)": lr.coef_
                }).sort_values("Impact (Coefficient)", ascending=False)

                # --- Auto-Insight Generator (Regression) ---
                top_driver = coef_df.iloc[0]
                driver_name = top_driver["Defect Category"]
                driver_val = top_driver["Impact (Coefficient)"]
                
                if r2 < 0.2:
                    fit_desc = "weak correlation (other factors likely drive disruption)"
                elif r2 < 0.5:
                    fit_desc = "moderate correlation"
                else:
                    fit_desc = "strong predictive power"

                insight_text_reg = (
                    f"**Auto-insight:** The model shows a **{fit_desc}** (R²={r2:.3f}). "
                    f"The biggest driver of clinical disruption is **{driver_name}** defects: "
                    f"each additional reported {driver_name} defect adds roughly **{driver_val:.2f} days** of disruption "
                    "on average, keeping other factors constant."
                )

                c1, c2 = st.columns([1, 2])
                with c1:
                    st.metric("Model R² Score", f"{r2:.3f}", help="1.0 is perfect prediction.")
                    st.dataframe(coef_df, hide_index=True, use_container_width=True)
                with c2:
                    viz_df = pd.DataFrame({"Actual Days": y_days, "Predicted Days": y_pred})
                    fig_lr = px.scatter(
                        viz_df, x="Actual Days", y="Predicted Days", 
                        title="Actual vs. Predicted Disruption Days"
                    )
                    min_val = min(y_days.min(), y_pred.min())
                    max_val = max(y_days.max(), y_pred.max())
                    fig_lr.add_shape(
                        type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                        line=dict(color="Red", dash="dash"),
                    )
                    st.plotly_chart(fig_lr, use_container_width=True)
                
                # Show Insight below graph
                st.info(insight_text_reg)

            else:
                st.warning("Cannot run Regression: 'Clinical Disruption Days' column missing or empty.")

        # --- B) GRADIENT BOOSTING CLASSIFIER ---
        with tab_class:
            st.markdown("### Classify 'High Risk' Locations")
            st.caption("This model classifies a location as 'High Risk' if its total defects are above the median.")
            
            total_def_col = TOTAL_DEFECT if TOTAL_DEFECT in df.columns else None
            
            if total_def_col:
                totals = safe_num(df[total_def_col]).fillna(0)
                median_val = totals.median()
                y_class = (totals > median_val).astype(int)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
                
                gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
                gb.fit(X_train, y_train)
                y_pred_class = gb.predict(X_test)
                acc = accuracy_score(y_test, y_pred_class)
                
                importances = gb.feature_importances_
                feat_imp_df = pd.DataFrame({
                    "Feature": list(valid_feats.keys()),
                    "Importance": importances
                }).sort_values("Importance", ascending=False)
                
                # --- Auto-Insight Generator (Classification) ---
                top_feat = feat_imp_df.iloc[0]
                top_feat_name = top_feat["Feature"]
                top_feat_score = top_feat["Importance"]
                
                if acc > 0.85:
                    acc_desc = "highly accurate"
                elif acc > 0.70:
                    acc_desc = "moderately accurate"
                else:
                    acc_desc = "low accuracy"

                insight_text_gb = (
                    f"**Auto-insight:** The model is **{acc_desc}** ({acc:.1%}) at identifying high-risk locations. "
                    f"**{top_feat_name}** defects are the single most important indicator "
                    f"(Importance: {top_feat_score:.2f}), meaning an increase in {top_feat_name} issues "
                    "is the strongest warning sign that a location will become a hotspot."
                )

                c1, c2 = st.columns([1, 2])
                with c1:
                    st.metric("Model Accuracy", f"{acc:.1%}")
                    st.write(f"**Threshold (Median):** > {int(median_val)} defects")
                    st.dataframe(feat_imp_df, hide_index=True, use_container_width=True)
                with c2:
                    fig_imp = px.bar(
                        feat_imp_df, x="Importance", y="Feature", orientation='h',
                        title="Feature Importance (Gradient Boosting)",
                        color="Importance", color_continuous_scale="Viridis"
                    )
                    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_imp, use_container_width=True)
                
                # Show Insight below graph
                st.success(insight_text_gb)

            else:
                st.warning("Cannot run Classification: Total Defect column missing.")

else:
    st.error("Scikit-learn not installed. These models cannot run.")

# --------------------------
# 12) Statistical Analysis: Operational Status vs Clinical Disruption
# --------------------------
st.markdown("## 12) Statistical Analysis: Operational Status vs Clinical Disruption")

if OPSTAT and DISRUPT_COL and (OPSTAT in dff.columns) and (DISRUPT_COL in dff.columns):
    st.markdown("""
    **Objective:** Compare clinical service disruption days across three operational status groups:
    - Group 1: Active
    - Group 2: Rotation
    - Group 3: Pending
    """)
    
    # Prepare data
    analysis_df = dff[[OPSTAT, DISRUPT_COL]].copy()
    analysis_df['Status_Clean'] = analysis_df[OPSTAT].astype(str).str.lower().str.strip()
    analysis_df['Disruption_Days'] = pd.to_numeric(
        analysis_df[DISRUPT_COL].astype(str)
            .str.replace(",", "", regex=False)
            .str.extract(r"(-?\d+\.?\d*)", expand=False),
        errors="coerce"
    ).fillna(0)
    
    # Filter for the three groups
    analysis_df = analysis_df[analysis_df['Status_Clean'].isin(['active', 'rotation', 'pending'])].copy()
    
    if len(analysis_df) > 0:
        # Summary statistics by group
        st.markdown("### 12.1) Descriptive Statistics")
        
        summary = analysis_df.groupby('Status_Clean')['Disruption_Days'].agg([
            ('Count', 'count'),
            ('Mean', 'mean'),
            ('Median', 'median'),
            ('Std Dev', 'std'),
            ('Min', 'min'),
            ('Max', 'max'),
            ('Q1', lambda x: x.quantile(0.25)),
            ('Q3', lambda x: x.quantile(0.75))
        ]).round(2)
        
        st.dataframe(summary, use_container_width=True)
        
        if AUTO_INSIGHTS:
            highest_mean = summary['Mean'].idxmax()
            highest_mean_val = summary.loc[highest_mean, 'Mean']
            lowest_mean = summary['Mean'].idxmin()
            lowest_mean_val = summary.loc[lowest_mean, 'Mean']
            highest_var = summary['Std Dev'].idxmax()
            highest_var_val = summary.loc[highest_var, 'Std Dev']
            
            st.caption(
                f"**Auto-insights (Descriptive):** **{highest_mean.title()}** status has highest mean disruption "
                f"(**{highest_mean_val:.1f} days**), while **{lowest_mean.title()}** has lowest "
                f"(**{lowest_mean_val:.1f} days**). **{highest_var.title()}** shows highest variability "
                f"(SD={highest_var_val:.1f}), indicating inconsistent disruption patterns."
            )
        
        # Visualization: Box plots
        st.markdown("### 12.2) Distribution Comparison (Box Plots)")
        fig_box = px.box(
            analysis_df, 
            x='Status_Clean', 
            y='Disruption_Days',
            color='Status_Clean',
            title="Distribution of Clinical Disruption Days by Operational Status",
            labels={'Status_Clean': 'Operational Status', 'Disruption_Days': 'Disruption Days'},
            color_discrete_map={
                'active': '#00FF00',
                'rotation': '#0000FF',
                'pending': '#FF0000'
            }
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
        
        if AUTO_INSIGHTS:
            # Calculate IQR and outliers for each group
            outlier_info = []
            outlier_details = {}
            for status in ['active', 'rotation', 'pending']:
                group_data = analysis_df[analysis_df['Status_Clean'] == status]['Disruption_Days']
                if len(group_data) > 0:
                    q1, q3 = group_data.quantile(0.25), group_data.quantile(0.75)
                    iqr = q3 - q1
                    outliers = group_data[(group_data < q1 - 1.5*iqr) | (group_data > q3 + 1.5*iqr)]
                    outlier_details[status] = {
                        'count': len(outliers),
                        'max': outliers.max() if len(outliers) > 0 else 0,
                        'median': group_data.median(),
                        'q1': q1,
                        'q3': q3
                    }
                    if len(outliers) > 0:
                        outlier_info.append(f"{status.title()} ({len(outliers)} outliers, max={outliers.max():.0f} days)")
            
            median_vals = analysis_df.groupby('Status_Clean')['Disruption_Days'].median().sort_values(ascending=False)
            highest_median = median_vals.index[0]
            highest_median_val = median_vals.iloc[0]
            lowest_median = median_vals.index[-1]
            lowest_median_val = median_vals.iloc[-1]
            
            # Count zero-disruption cases
            zero_counts = {status: (analysis_df[analysis_df['Status_Clean'] == status]['Disruption_Days'] == 0).sum() 
                          for status in ['active', 'rotation', 'pending']}
            
            outlier_text = "; ".join(outlier_info) if outlier_info else "No significant outliers detected"
            
            st.caption(
                f"**Auto-insights (Box Plot):** **{highest_median.title()}** has highest median disruption "
                f"(**{highest_median_val:.1f} days**) while **{lowest_median.title()}** has lowest "
                f"(**{lowest_median_val:.1f} days**). Box height shows interquartile range (middle 50% of data). "
                f"{outlier_text}. "
                f"Zero-disruption locations: Active={zero_counts.get('active', 0)}, "
                f"Rotation={zero_counts.get('rotation', 0)}, Pending={zero_counts.get('pending', 0)}. "
                f"Investigate outlier cases for systemic issues or exceptional circumstances."
            )
        
        # Violin plot for distribution shape
        st.markdown("### 12.3) Distribution Shape (Violin Plots)")
        fig_violin = px.violin(
            analysis_df,
            x='Status_Clean',
            y='Disruption_Days',
            color='Status_Clean',
            box=True,
            points='all',
            title="Distribution Shape of Disruption Days by Status",
            labels={'Status_Clean': 'Operational Status', 'Disruption_Days': 'Disruption Days'},
            color_discrete_map={
                'active': '#00FF00',
                'rotation': '#0000FF',
                'pending': '#FF0000'
            }
        )
        st.plotly_chart(fig_violin, use_container_width=True)
        
        if AUTO_INSIGHTS:
            # Analyze distribution shapes
            skew_info = []
            distribution_summary = {}
            
            try:
                from scipy.stats import skew as calc_skew
                
                for status in ['active', 'rotation', 'pending']:
                    group_data = analysis_df[analysis_df['Status_Clean'] == status]['Disruption_Days']
                    if len(group_data) > 2:
                        skewness = calc_skew(group_data)
                        mean_val = group_data.mean()
                        median_val = group_data.median()
                        std_val = group_data.std()
                        
                        distribution_summary[status] = {
                            'skewness': skewness,
                            'mean': mean_val,
                            'median': median_val,
                            'std': std_val,
                            'n': len(group_data)
                        }
                        
                        if abs(skewness) > 1:
                            direction = "right-skewed (many low values, few extremely high)" if skewness > 0 else "left-skewed (many high values, few extremely low)"
                            skew_info.append(f"**{status.title()}** is {direction} (skew={skewness:.2f})")
                        elif abs(skewness) > 0.5:
                            direction = "moderately right-skewed" if skewness > 0 else "moderately left-skewed"
                            skew_info.append(f"**{status.title()}** is {direction} (skew={skewness:.2f})")
                
                # Find most concentrated distribution
                if distribution_summary:
                    most_concentrated = min(distribution_summary.items(), key=lambda x: x[1]['std'])
                    most_variable = max(distribution_summary.items(), key=lambda x: x[1]['std'])
                    
                    skew_text = ". ".join(skew_info) if skew_info else "All groups show relatively symmetric distributions"
                    
                    # Identify modal regions (density peaks)
                    density_insight = []
                    for status, stats in distribution_summary.items():
                        if stats['median'] < 5:
                            density_insight.append(f"{status.title()} clusters near zero disruption")
                        elif stats['median'] > 10:
                            density_insight.append(f"{status.title()} clusters at high disruption levels")
                    
                    density_text = "; ".join(density_insight) if density_insight else "All groups show similar density patterns"
                    
                    st.caption(
                        f"**Auto-insights (Violin Plot):** Violin width shows density at each disruption level - "
                        f"wider = more locations. {skew_text}. "
                        f"**{most_concentrated[0].title()}** shows most consistent patterns (SD={most_concentrated[1]['std']:.1f}), "
                        f"while **{most_variable[0].title()}** is most variable (SD={most_variable[1]['std']:.1f}). "
                        f"{density_text}. "
                        f"The embedded box plot shows median and quartiles; individual points reveal all data values."
                    )
                else:
                    st.caption(
                        f"**Auto-insights (Violin Plot):** Violin width shows data density at each disruption level. "
                        f"Wider sections = more locations with that disruption duration. "
                        f"Individual points reveal actual data distribution and clustering patterns."
                    )
            except ImportError:
                skew_text = "Distribution analysis requires scipy"
                st.caption(
                    f"**Auto-insights (Violin Plot):** Violin width shows data density at each disruption level. "
                    f"Wider sections = more locations with that disruption duration. {skew_text}. "
                    f"Individual points reveal actual data distribution and clustering patterns."
                )
        
        # Statistical Tests
        st.markdown("### 12.4) Statistical Hypothesis Testing")
        
        # Prepare groups for testing
        active_data = analysis_df[analysis_df['Status_Clean'] == 'active']['Disruption_Days']
        rotation_data = analysis_df[analysis_df['Status_Clean'] == 'rotation']['Disruption_Days']
        pending_data = analysis_df[analysis_df['Status_Clean'] == 'pending']['Disruption_Days']
        
        groups = [g for g in [active_data, rotation_data, pending_data] if len(g) > 0]
        group_names = []
        if len(active_data) > 0: group_names.append('Active')
        if len(rotation_data) > 0: group_names.append('Rotation')
        if len(pending_data) > 0: group_names.append('Pending')
        
        if len(groups) >= 2:
            try:
                from scipy import stats
                
                # 1. Kruskal-Wallis H-test (non-parametric alternative to ANOVA)
                st.markdown("#### A) Kruskal-Wallis H-Test (Non-Parametric)")
                st.caption("Tests if there are statistically significant differences between the groups.")
                
                if len(groups) >= 2:
                    h_stat, p_value_kw = stats.kruskal(*groups)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("H-statistic", f"{h_stat:.4f}")
                    with col2:
                        st.metric("p-value", f"{p_value_kw:.4f}")
                    
                    if p_value_kw < 0.05:
                        st.success(f"✓ **Significant difference detected** (p < 0.05)")
                        st.caption("The operational status groups have significantly different disruption days.")
                    else:
                        st.info(f"No significant difference detected (p ≥ 0.05)")
                        st.caption("The disruption days do not significantly differ across operational status groups.")
                    
                    if AUTO_INSIGHTS:
                        interpretation = (
                            f"The Kruskal-Wallis test with H={h_stat:.2f} and p={p_value_kw:.4f} "
                            f"{'**confirms**' if p_value_kw < 0.05 else '**suggests**'} that operational status "
                            f"{'**does**' if p_value_kw < 0.05 else '**does not**'} significantly affect disruption duration. "
                        )
                        if p_value_kw < 0.05:
                            interpretation += "This indicates that management should prioritize certain status types for intervention."
                        else:
                            interpretation += "Disruption may be influenced more by other factors (location, department, defect type)."
                        st.caption(f"**Auto-insights (Kruskal-Wallis):** {interpretation}")
                
                # 2. Pairwise Mann-Whitney U tests (post-hoc)
                if len(groups) >= 2 and p_value_kw < 0.05:
                    st.markdown("#### B) Pairwise Mann-Whitney U Tests (Post-Hoc)")
                    st.caption("Identifies which specific pairs of groups differ significantly.")
                    
                    pairs = []
                    if len(active_data) > 0 and len(rotation_data) > 0:
                        u_stat, p_val = stats.mannwhitneyu(active_data, rotation_data, alternative='two-sided')
                        pairs.append(('Active vs Rotation', u_stat, p_val))
                    
                    if len(active_data) > 0 and len(pending_data) > 0:
                        u_stat, p_val = stats.mannwhitneyu(active_data, pending_data, alternative='two-sided')
                        pairs.append(('Active vs Pending', u_stat, p_val))
                    
                    if len(rotation_data) > 0 and len(pending_data) > 0:
                        u_stat, p_val = stats.mannwhitneyu(rotation_data, pending_data, alternative='two-sided')
                        pairs.append(('Rotation vs Pending', u_stat, p_val))
                    
                    # Bonferroni correction
                    alpha_corrected = 0.05 / len(pairs) if len(pairs) > 0 else 0.05
                    
                    pairwise_results = pd.DataFrame(pairs, columns=['Comparison', 'U-statistic', 'p-value'])
                    pairwise_results['Significant'] = pairwise_results['p-value'] < alpha_corrected
                    pairwise_results['Corrected α'] = alpha_corrected
                    
                    st.dataframe(pairwise_results, use_container_width=True)
                    st.caption(f"Using Bonferroni correction: α = 0.05 / {len(pairs)} = {alpha_corrected:.4f}")
                    
                    if AUTO_INSIGHTS:
                        sig_pairs = pairwise_results[pairwise_results['Significant']]
                        if len(sig_pairs) > 0:
                            sig_comparisons = ", ".join(sig_pairs['Comparison'].tolist())
                            st.caption(
                                f"**Auto-insights (Pairwise):** Significant differences found in: **{sig_comparisons}**. "
                                f"These pairs show statistically different disruption patterns and warrant separate "
                                f"management strategies. Non-significant pairs may be managed similarly."
                            )
                        else:
                            st.caption(
                                f"**Auto-insights (Pairwise):** Despite overall significance, no individual pairs "
                                f"meet the corrected threshold (α={alpha_corrected:.4f}). This suggests differences "
                                f"are spread across all groups rather than concentrated between specific pairs."
                            )
                
                # 3. Effect size (Eta-squared for Kruskal-Wallis)
                st.markdown("#### C) Effect Size")
                st.caption("Measures the magnitude of the difference (0.01=small, 0.06=medium, 0.14=large)")
                
                n_total = sum(len(g) for g in groups)
                eta_squared = (h_stat - len(groups) + 1) / (n_total - len(groups))
                
                st.metric("Eta-squared (η²)", f"{eta_squared:.4f}")
                
                if eta_squared < 0.01:
                    effect_interpretation = "Negligible effect"
                elif eta_squared < 0.06:
                    effect_interpretation = "Small effect"
                elif eta_squared < 0.14:
                    effect_interpretation = "Medium effect"
                else:
                    effect_interpretation = "Large effect"
                
                st.info(f"**Interpretation:** {effect_interpretation}")
                
                if AUTO_INSIGHTS:
                    practical_sig = (
                        f"**Auto-insights (Effect Size):** η²={eta_squared:.4f} indicates **{effect_interpretation.lower()}**. "
                    )
                    if eta_squared < 0.06:
                        practical_sig += (
                            "While statistically significant, the practical impact is small. "
                            "Other factors (location, defect type) may be more important for reducing disruption."
                        )
                    elif eta_squared < 0.14:
                        practical_sig += (
                            "Operational status has a moderate impact on disruption. "
                            "Consider this factor in planning and resource allocation."
                        )
                    else:
                        practical_sig += (
                            "Operational status is a MAJOR driver of disruption duration. "
                            "Management interventions targeting status transitions could significantly reduce disruption."
                        )
                    st.caption(practical_sig)
                
                # 4. Normality tests (for reference)
                st.markdown("#### D) Normality Tests (Shapiro-Wilk)")
                st.caption("Tests if data follows a normal distribution (informational only)")
                
                normality_results = []
                for group_data, name in zip([active_data, rotation_data, pending_data], 
                                           ['Active', 'Rotation', 'Pending']):
                    if len(group_data) >= 3:  # Shapiro-Wilk requires at least 3 samples
                        stat, p_val = stats.shapiro(group_data)
                        normality_results.append({
                            'Group': name,
                            'n': len(group_data),
                            'Statistic': f"{stat:.4f}",
                            'p-value': f"{p_val:.4f}",
                            'Normal?': 'Yes' if p_val > 0.05 else 'No'
                        })
                
                if normality_results:
                    st.dataframe(pd.DataFrame(normality_results), use_container_width=True)
                    st.caption("If p > 0.05, data is approximately normally distributed.")
                    
                    if AUTO_INSIGHTS:
                        non_normal = [r['Group'] for r in normality_results if r['Normal?'] == 'No']
                        if len(non_normal) > 0:
                            st.caption(
                                f"**Auto-insights (Normality):** Groups **{', '.join(non_normal)}** show "
                                f"non-normal distribution (skewed or heavy-tailed). This confirms that "
                                f"non-parametric tests (Kruskal-Wallis, Mann-Whitney) are more appropriate "
                                f"than traditional ANOVA/t-tests for this dataset."
                            )
                        else:
                            st.caption(
                                f"**Auto-insights (Normality):** All groups approximate normal distribution. "
                                f"Both parametric (ANOVA) and non-parametric tests would be valid, though "
                                f"non-parametric methods are more robust to outliers common in disruption data."
                            )
                
            except ImportError:
                st.warning("Install scipy for statistical tests: `pip install scipy`")
            except Exception as e:
                st.error(f"Error performing statistical tests: {e}")
        else:
            st.info("Need at least 2 groups with data to perform statistical comparison.")
        
        # Auto-insights summary
        if AUTO_INSIGHTS:
            total_locations = len(analysis_df)
            mean_active = active_data.mean() if len(active_data) > 0 else 0
            mean_rotation = rotation_data.mean() if len(rotation_data) > 0 else 0
            mean_pending = pending_data.mean() if len(pending_data) > 0 else 0
            
            st.markdown("### 12.5) Key Findings Summary")
            st.caption(
                f"**Auto-insights:** Analyzed **{total_locations}** locations. "
                f"Mean disruption days: Active={mean_active:.1f}, Rotation={mean_rotation:.1f}, Pending={mean_pending:.1f}. "
                f"{'Statistical tests show significant differences between groups.' if 'p_value_kw' in locals() and p_value_kw < 0.05 else 'No significant statistical differences detected between groups.'}"
            )
    else:
        st.info("No data found for Active, Rotation, or Pending status groups.")
else:
    st.info("Need OPERATIONAL STATUS and CLINICAL DISRUPTION columns for statistical analysis.")

# --------------------------
# 13) STATISTICAL ANALYSIS: Active vs (Rotation + Pending)
# --------------------------
st.markdown("## 13) Statistical Analysis: Active vs (Rotation + Pending)")

if OPSTAT and TOTAL_DEFECT:
    # Prepare comparison groups
    dff_stat = dff.copy()
    dff_stat['_status_clean'] = dff_stat[OPSTAT].astype(str).str.strip().str.lower()
    dff_stat['_total_def'] = safe_num(dff_stat[TOTAL_DEFECT]).fillna(0)
    
    # Create binary grouping
    dff_stat['_group'] = dff_stat['_status_clean'].apply(
        lambda x: 'Active' if x == 'active' else ('Rotation+Pending' if x in ['rotation', 'pending'] else 'Other')
    )
    
    # Filter to relevant groups
    analysis_df = dff_stat[dff_stat['_group'].isin(['Active', 'Rotation+Pending'])].copy()
    
    if len(analysis_df) > 0:
        active_defects = analysis_df[analysis_df['_group'] == 'Active']['_total_def']
        rot_pend_defects = analysis_df[analysis_df['_group'] == 'Rotation+Pending']['_total_def']
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Active Locations", len(active_defects))
            st.metric("Mean Defects (Active)", f"{active_defects.mean():.2f}")
            st.metric("Median Defects (Active)", f"{active_defects.median():.2f}")
        
        with col2:
            st.metric("Rotation+Pending Locations", len(rot_pend_defects))
            st.metric("Mean Defects (Rot+Pend)", f"{rot_pend_defects.mean():.2f}")
            st.metric("Median Defects (Rot+Pend)", f"{rot_pend_defects.median():.2f}")
        
        with col3:
            diff_mean = active_defects.mean() - rot_pend_defects.mean()
            diff_median = active_defects.median() - rot_pend_defects.median()
            st.metric("Difference in Mean", f"{diff_mean:.2f}", delta=None)
            st.metric("Difference in Median", f"{diff_median:.2f}", delta=None)
        
        # Statistical tests
        st.markdown("### Statistical Tests")
        
        try:
            from scipy import stats
            
            # 1. Mann-Whitney U Test (non-parametric, good for non-normal distributions)
            u_stat, p_mann = stats.mannwhitneyu(active_defects, rot_pend_defects, alternative='two-sided')
            
            # 2. Independent t-test (parametric)
            t_stat, p_ttest = stats.ttest_ind(active_defects, rot_pend_defects, equal_var=False)
            
            # 3. Levene's test for equality of variances
            lev_stat, p_levene = stats.levene(active_defects, rot_pend_defects)
            
            # 4. Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(active_defects)-1)*active_defects.std()**2 + 
                                  (len(rot_pend_defects)-1)*rot_pend_defects.std()**2) / 
                                 (len(active_defects) + len(rot_pend_defects) - 2))
            cohens_d = (active_defects.mean() - rot_pend_defects.mean()) / pooled_std if pooled_std > 0 else 0
            
            # Display results
            test_results = pd.DataFrame({
                'Test': [
                    'Mann-Whitney U Test',
                    'Independent t-test',
                    "Levene's Test (variance)",
                    "Cohen's d (effect size)"
                ],
                'Statistic': [
                    f"U = {u_stat:.2f}",
                    f"t = {t_stat:.2f}",
                    f"W = {lev_stat:.2f}",
                    f"d = {cohens_d:.3f}"
                ],
                'p-value': [
                    f"{p_mann:.4f}",
                    f"{p_ttest:.4f}",
                    f"{p_levene:.4f}",
                    "N/A"
                ],
                'Interpretation': [
                    "Significant difference" if p_mann < 0.05 else "No significant difference",
                    "Significant difference" if p_ttest < 0.05 else "No significant difference",
                    "Unequal variances" if p_levene < 0.05 else "Equal variances",
                    "Large effect" if abs(cohens_d) > 0.8 else ("Medium effect" if abs(cohens_d) > 0.5 else "Small effect")
                ]
            })
            
            st.dataframe(test_results, use_container_width=True)
            
            # Interpretation
            st.markdown("### Interpretation")
            
            if p_mann < 0.05:
                direction = "higher" if active_defects.median() > rot_pend_defects.median() else "lower"
                st.success(
                    f"✅ **Significant difference found** (Mann-Whitney U test, p={p_mann:.4f}). "
                    f"Active locations have **{direction}** defect counts than Rotation+Pending locations."
                )
            else:
                st.info(
                    f"ℹ️ **No significant difference** detected (Mann-Whitney U test, p={p_mann:.4f}). "
                    "The defect counts between Active and Rotation+Pending groups are statistically similar."
                )
            
            if abs(cohens_d) > 0.5:
                st.info(f"Effect size (Cohen's d = {cohens_d:.3f}) indicates a meaningful practical difference.")
            
        except Exception as e:
            st.warning(f"Could not perform statistical tests: {e}")
        
        # Box plot comparison
        st.markdown("### Distribution Comparison")
        
        fig_box = px.box(
            analysis_df,
            x='_group',
            y='_total_def',
            title="Defect Distribution: Active vs Rotation+Pending",
            labels={'_group': 'Group', '_total_def': 'Total Defects'},
            color='_group',
            color_discrete_map={'Active': '#00FF00', 'Rotation+Pending': '#FF6B6B'}
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Auto-insights for Box Plot
        if AUTO_INSIGHTS:
            # Calculate IQR and outliers for both groups
            q1_active = active_defects.quantile(0.25)
            q3_active = active_defects.quantile(0.75)
            iqr_active = q3_active - q1_active
            outliers_active = ((active_defects < (q1_active - 1.5 * iqr_active)) | 
                             (active_defects > (q3_active + 1.5 * iqr_active))).sum()
            
            q1_rot = rot_pend_defects.quantile(0.25)
            q3_rot = rot_pend_defects.quantile(0.75)
            iqr_rot = q3_rot - q1_rot
            outliers_rot = ((rot_pend_defects < (q1_rot - 1.5 * iqr_rot)) | 
                          (rot_pend_defects > (q3_rot + 1.5 * iqr_rot))).sum()
            
            spread_active = active_defects.std()
            spread_rot = rot_pend_defects.std()
            
            st.caption(
                f"**Box Plot Insights:** Active group median = **{active_defects.median():.1f}** "
                f"(IQR: {q1_active:.1f} to {q3_active:.1f}, {outliers_active} outliers). "
                f"Rotation+Pending median = **{rot_pend_defects.median():.1f}** "
                f"(IQR: {q1_rot:.1f} to {q3_rot:.1f}, {outliers_rot} outliers). "
                f"Active group has {'wider' if spread_active > spread_rot else 'narrower'} spread "
                f"(SD: {spread_active:.1f} vs {spread_rot:.1f})."
            )
        
        # Violin plot for detailed distribution
        fig_violin = px.violin(
            analysis_df,
            x='_group',
            y='_total_def',
            title="Defect Distribution (Violin Plot)",
            labels={'_group': 'Group', '_total_def': 'Total Defects'},
            color='_group',
            color_discrete_map={'Active': '#00FF00', 'Rotation+Pending': '#FF6B6B'},
            box=True,
            points='all'
        )
        st.plotly_chart(fig_violin, use_container_width=True)
        
        # Auto-insights for Violin Plot
        if AUTO_INSIGHTS:
            # Analyze distribution shape
            from scipy.stats import skew, kurtosis
            
            skew_active = skew(active_defects)
            skew_rot = skew(rot_pend_defects)
            kurt_active = kurtosis(active_defects)
            kurt_rot = kurtosis(rot_pend_defects)
            
            # Determine skewness interpretation
            def skew_text(s):
                if s > 1:
                    return "highly right-skewed (long tail toward high values)"
                elif s > 0.5:
                    return "moderately right-skewed"
                elif s < -1:
                    return "highly left-skewed (long tail toward low values)"
                elif s < -0.5:
                    return "moderately left-skewed"
                else:
                    return "roughly symmetric"
            
            # Determine concentration
            def kurt_text(k):
                if k > 3:
                    return "heavy-tailed (many extreme values)"
                elif k > 0:
                    return "moderately peaked"
                elif k < -1:
                    return "flat distribution (uniform spread)"
                else:
                    return "normal-like distribution"
            
            # Count high vs low defect locations
            median_overall = analysis_df['_total_def'].median()
            high_active = (active_defects > median_overall).sum()
            high_rot = (rot_pend_defects > median_overall).sum()
            
            pct_high_active = (high_active / len(active_defects) * 100) if len(active_defects) > 0 else 0
            pct_high_rot = (high_rot / len(rot_pend_defects) * 100) if len(rot_pend_defects) > 0 else 0
            
            st.caption(
                f"**Violin Plot Insights:** Active group distribution is **{skew_text(skew_active)}** "
                f"(skewness = {skew_active:.2f}) and **{kurt_text(kurt_active)}** (kurtosis = {kurt_active:.2f}). "
                f"Rotation+Pending is **{skew_text(skew_rot)}** (skewness = {skew_rot:.2f}) and "
                f"**{kurt_text(kurt_rot)}** (kurtosis = {kurt_rot:.2f}). "
                f"**{pct_high_active:.1f}%** of Active locations have above-median defects vs "
                f"**{pct_high_rot:.1f}%** of Rotation+Pending locations. "
                f"Violin width shows density: Active group concentrates around "
                f"**{active_defects.mode().iloc[0] if len(active_defects.mode()) > 0 else active_defects.median():.0f}** defects, "
                f"while Rotation+Pending concentrates around "
                f"**{rot_pend_defects.mode().iloc[0] if len(rot_pend_defects.mode()) > 0 else rot_pend_defects.median():.0f}** defects."
            )
        
    else:
        st.info("No data available for Active or Rotation+Pending groups.")
else:
    st.info("Need OPERATIONAL STATUS and TOTAL_DEFECT columns for statistical analysis.")

st.write("---")

# --------------------------
# 14) Data Master Severity & Disruption by Location
# --------------------------
st.markdown("## 14) Data Master Severity & Disruption by Location")
sev_cols, sev_labels = [], []
if colmap.get("major_defect"):
    sev_cols.append(colmap["major_defect"]); sev_labels.append("Major Defect")
if colmap.get("recurrent_defect"):
    sev_cols.append(colmap["recurrent_defect"]); sev_labels.append("Recurrent Defect")
if colmap.get("clinical_disruption_days"):
    sev_cols.append(colmap["clinical_disruption_days"]); sev_labels.append("Clinical Disruption (Days)")

if LOC and sev_cols:
    real_cols = [c for c in sev_cols if c in dff.columns]
    if real_cols:
        tmp = dff[[LOC] + real_cols].copy().loc[:, ~dff[[LOC] + real_cols].columns.duplicated()].copy()
        for c in real_cols:
            s = tmp[c]; cname = str(c).lower()
            if any(k in cname for k in ["day", "days", "disruption"]):
                tmp[c] = pd.to_numeric(
                    s.astype(str).str.replace(",", "", regex=False).str.extract(r"(-?\d+\.?\d*)", expand=False),
                    errors="coerce",
                )
            else:
                s2 = s.astype(str).str.lower().str.strip()
                tmp[c] = np.where(
                    s2.isin(["yes", "y", "true", "1"]),
                    1,
                    np.where(s2.isin(["no", "n", "false", "0"]), 0, safe_num(s)),
                )
        agg, final_labels = {}, []
        for c, label in zip(real_cols, sev_labels):
            if pd.api.types.is_numeric_dtype(tmp[c]):
                agg[c] = "sum"; final_labels.append(label)
            else:
                tmp[c] = tmp[c].astype(str).str.strip()
                agg[c] = lambda s: (s != "").sum()
                final_labels.append(label + " (count)")
        try:
            g = tmp.groupby(LOC, dropna=False).agg(agg).reset_index()
            if isinstance(g.columns, pd.MultiIndex):
                g.columns = ["_".join([str(x) for x in tup if str(x) not in ("", "None", "<lambda>")]).strip("_")
                             for tup in g.columns.values]
            non_loc_cols = [c for c in g.columns if c != LOC]
            g = g.rename(columns=dict(zip(non_loc_cols, final_labels[:len(non_loc_cols)])))
            st.dataframe(g, use_container_width=True, height=320)
            if AUTO_INSIGHTS:
                insight_severity_table(g, loc_col=LOC)
        except Exception as e:
            st.warning(f"Could not aggregate severity columns: {e}")
    else:
        st.info("Severity columns not found in filtered data.")
else:
    st.info("Need LOCATION and severity columns.")
# --------------------------
# 15) Data Master (All Rows & Columns)
# --------------------------
st.markdown("## 15) Data Master (All Rows & Columns)")
q = st.text_input("Quick search (matches anywhere in row; case-insensitive):", value="")
dm = dff.copy()
if q.strip():
    pat = re.compile(re.escape(q.strip()), re.IGNORECASE)
    dm = dm[dm.apply(lambda row: row.astype(str).str.contains(pat).any(), axis=1)]
st.dataframe(dm, use_container_width=True, height=420)
st.download_button(
    "Download filtered data (CSV)",
    data=dm.to_csv(index=False).encode("utf-8"),
    file_name="wcc_dlp_filtered.csv",
    mime="text/csv",
)
    
# --------------------------
# FOOTER
# --------------------------
st.write("---")
st.caption("If a section shows an info message, a required column wasn’t found. Rename in your Sheet, or extend the CANON mapper above.")
