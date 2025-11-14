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
    "WCC HTPN Dashboard: Operational Data, Defect Tracking, Future Development - Risk Register, "
    "Heatmap Analysis, and Predictive Risk Modeling"
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
# 1) Total Defect by Level
# --------------------------
st.markdown("## 1) Total Defect by Level")

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
# 2) Total Defect by Department
# --------------------------
st.markdown("## 2) Total Defect by Department")

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

# --------------------------
# 3) Category breakdowns
# --------------------------
st.markdown("## 3) Category Breakdown by Level (Mechanical/Electrical/Public/Biomedical/ICT)")
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

st.markdown("## 4) Category Breakdown by Department")
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
# 5) Pending Defects by Level (from cells)
# --------------------------
st.markdown("## 5) Pending Defects by Level")

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
# 6) Operational Status Pie
# --------------------------
st.markdown("## 6) Operational Status — Percentage")
if OPSTAT:
    counts = dff[OPSTAT].astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown"}).value_counts(dropna=False)
    pie_df = counts.rename_axis("Status").reset_index(name="Count")
    st.plotly_chart(
        px.pie(pie_df, names="Status", values="Count", title="Operational Status (%)", hole=0.35),
        use_container_width=True,
    )
    if AUTO_INSIGHTS:
        insight_pie_status(pie_df, name_col="Status", count_col="Count")
else:
    st.info("Need OPERATIONAL STATUS column.")

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

# Disruption days
if IMPACT_DAYS_COL and IMPACT_DAYS_COL in dff.columns:
    dff["_disrupt_days"] = pd.to_numeric(
        dff[IMPACT_DAYS_COL].astype(str)
            .str.replace(",", "", regex=False)
            .str.extract(r"(-?\d+\.?\d*)", expand=False),
        errors="coerce"
    ).fillna(0)
else:
    dff["_disrupt_days"] = 0

def band_days(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(
        series.astype(str)
              .str.replace(",", "", regex=False)
              .str.extract(r"(-?\d+\.?\d*)", expand=False),
        errors="coerce"
    ).fillna(0)
    try:
        return pd.qcut(s, 5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype("Int64").fillna(1).astype(int)
    except Exception:
        return pd.cut(
            s,
            bins=[-0.1, 0, 1, 3, 7, float("inf")],
            labels=[1, 2, 3, 4, 5],
            include_lowest=True
        ).astype(int)

impact_days_band = band_days(dff["_disrupt_days"])

impact_fin_band = (
    band_finance(dff[FIN], FIN_METHOD, T1, T2, T3, T4)
    if (USE_FINANCE_IN_IMPACT and FIN and FIN in dff.columns)
    else pd.Series(1, index=dff.index, dtype=int)
)

IMPACT_BAND = np.maximum(impact_days_band, impact_fin_band) if USE_FINANCE_IN_IMPACT else impact_days_band

# Likelihood band (frequency proxy)
if TOTAL_DEFECT and TOTAL_DEFECT in dff.columns:
    tmp_freq = safe_num(dff[TOTAL_DEFECT]).fillna(0)
else:
    melt_cols = [c for c in cat_cols.values() if c and c in dff.columns]
    tmp_freq = dff[melt_cols].apply(safe_num).fillna(0).sum(axis=1) if melt_cols else pd.Series(0, index=dff.index)

group_key = [c for c in [LOC, DEPT] if c]
if group_key:
    freq_by_grp = dff.groupby(group_key, dropna=False).apply(
        lambda g: safe_num(g[TOTAL_DEFECT]).fillna(0).sum() if TOTAL_DEFECT in g.columns else 0
    )
    freq_map = dff[group_key].merge(
        freq_by_grp.rename("grp_freq"),
        left_on=group_key,
        right_index=True,
        how="left"
    )["grp_freq"].fillna(0).values
    freq_series = pd.Series(freq_map, index=dff.index)
else:
    freq_series = tmp_freq

def band_freq(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    try:
        return pd.qcut(s, 5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype("Int64").fillna(1).astype(int)
    except Exception:
        return pd.cut(
            s,
            bins=[-0.1, 1, 5, 15, 40, float("inf")],
            labels=[1, 2, 3, 4, 5],
            include_lowest=True
        ).astype(int)

LIKELIHOOD_BAND = band_freq(freq_series)

RISK_SCORE = (IMPACT_BAND * LIKELIHOOD_BAND).astype(int)
dff["_impact_band"] = IMPACT_BAND
dff["_likelihood_band"] = LIKELIHOOD_BAND
dff["_risk_score"] = RISK_SCORE

# BIA flag
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
        dff[id_cols + ["_impact_band", "_likelihood_band", "_risk_score"]]
        .groupby(id_cols, dropna=False)
        .agg(
            risk_max=("_risk_score", "max"),
            risk_mean=("_risk_score", "mean"),
            impact_mean=("_impact_band", "mean"),
            like_mean=("_likelihood_band", "mean"),
        )
        .reset_index()
        .sort_values(["risk_max", "risk_mean"], ascending=False)
    )
    st.dataframe(reg.head(30), use_container_width=True, height=320)
    if AUTO_INSIGHTS:
        insight_bcms_risk(reg, BCMS_RISK_APPETITE)
else:
    st.info("BCMS risk register needs at least one of: Service, Location, or Department columns.")

# --------------------------
# 10) BCMS Overview & Watchlist
# --------------------------
st.markdown("## 10) BCMS Overview & Watchlist")

svc_unique = sorted(dff["_svc"].astype(str).str.strip().replace("", np.nan).dropna().unique().tolist())
n_services = len(svc_unique)

if "_bia_flag" in dff.columns and n_services > 0:
    bia_per_svc = dff.groupby("_svc")["_bia_flag"].max()
    bia_services = int((bia_per_svc > 0).sum())
else:
    bia_services = 0

risk_breach_n = int(dff["_risk_breach"].sum())
rto_breach_n = int(dff["_rto_breach"].sum())
bcp_overdue_n = int(dff["_bcp_overdue"].sum())

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("**Services with BIA fields**")
    st.markdown(f"<h2>{bia_services}</h2>", unsafe_allow_html=True)
with c2:
    st.markdown("**Risk score ≥ appetite**")
    st.markdown(f"<h2>{risk_breach_n}</h2>", unsafe_allow_html=True)
with c3:
    st.markdown("**RTO breaches (current data)**")
    st.markdown(f"<h2>{rto_breach_n}</h2>", unsafe_allow_html=True)
with c4:
    st.markdown("**BCP tests overdue**")
    st.markdown(f"<h2>{bcp_overdue_n}</h2>", unsafe_allow_html=True)

if AUTO_INSIGHTS and n_services > 0:
    insight_bcms_overview(n_services, bia_services, risk_breach_n, rto_breach_n, bcp_overdue_n)

# Watchlist table
dff["_bcp_last_test"] = dff[BCP_LAST_COL].astype(str) if BCP_LAST_COL and BCP_LAST_COL in dff.columns else ""
dff["_bcp_next_due"]  = dff[BCP_NEXT_COL].astype(str) if BCP_NEXT_COL and BCP_NEXT_COL in dff.columns else ""
dff["_workaround"]    = dff[WORKAROUND_COL].astype(str) if WORKAROUND_COL and WORKAROUND_COL in dff.columns else ""
dff["_updep"]         = dff[UPDEP_COL].astype(str) if UPDEP_COL and UPDEP_COL in dff.columns else ""

flag_cols = ["_risk_breach", "_rto_breach", "_mtpd_breach"]

watch_cols = [
    "_svc", "_owner", "_disrupt_days", "_rto_h", "_mtpd_h",
    "_gap_rto_h", "_gap_mtpd_h",
    "_risk_score", "_impact_band", "_likelihood_band",
    "_bcp_last_test", "_bcp_next_due", "_bcp_overdue",
    "_workaround", "_updep", LEVEL, DEPT, LOC,
] + flag_cols

for c in watch_cols:
    if c not in dff.columns:
        dff[c] = dff.get(c, "")

watch = dff[
    (dff["_risk_breach"]) |
    (dff["_rto_breach"]) |
    (dff["_mtpd_breach"]) |
    (dff["_disrupt_days"] >= DISRUPT_ALERT_DAYS)
][watch_cols].copy()

for c in flag_cols:
    watch[c] = watch[c].fillna(False).astype(bool)

if not watch.empty:
    watch = watch.sort_values(
        ["_risk_breach", "_rto_breach", "_mtpd_breach", "_disrupt_days"],
        ascending=[False, False, False, False]
    )

    display_watch = watch.rename(columns={
        "_svc": "Service",
        "_owner": "Risk Owner",
        "_disrupt_days": "Disruption (days)",
        "_rto_h": "RTO (h)",
        "_mtpd_h": "MTPD (h)",
        "_gap_rto_h": "Gap vs RTO (h)",
        "_gap_mtpd_h": "Gap vs MTPD (h)",
        "_risk_score": "Risk (I×L)",
        "_impact_band": "Impact band",
        "_likelihood_band": "Likelihood band",
        "_bcp_last_test": "BCP last test",
        "_bcp_next_due": "BCP next due",
        "_bcp_overdue": "BCP overdue?",
        "_workaround": "Workaround",
        "_updep": "Upstream dependency",
    })

    st.dataframe(display_watch, use_container_width=True, height=360)

    st.download_button(
        "Download BCMS Watchlist (CSV)",
        data=display_watch.to_csv(index=False).encode("utf-8"),
        file_name="bcms_watchlist.csv",
        mime="text/csv",
    )
else:
    st.info("No services currently breaching risk appetite, RTO/MTPD, or disruption alerts.")

# --------------------------
# 11) Heatmap — Service × Risk Band
# --------------------------
st.markdown("## 11) Heatmap — Service × Risk Band")

if "_risk_score" in dff.columns:
    dff["_risk_band_str"] = pd.cut(
        dff["_risk_score"],
        bins=[0, 5, 10, 15, 20, 25],
        labels=["1–5", "6–10", "11–15", "16–20", "21–25"],
        include_lowest=True,
    ).astype(str)

    if "_svc" in dff.columns:
        heat_svc = (
            dff.groupby(["_svc", "_risk_band_str"])
               .size()
               .unstack(fill_value=0)
               .sort_index()
        )
        heat_svc.index.name = "Service"
        heat_svc.columns.name = "Risk band"

        if not heat_svc.empty:
            fig_hsvc = px.imshow(
                heat_svc,
                aspect="auto",
                labels={"x": "Risk band (I×L)", "y": "Service", "color": "Count"},
                title="Service × Risk band"
            )
            st.plotly_chart(fig_hsvc, use_container_width=True)

            if AUTO_INSIGHTS:
                insight_heatmap_simple(heat_svc, "Service", "Risk band")
        else:
            st.info("No BCMS risk scores to plot for services.")
    else:
        st.info("Service column not found (for heatmap).")
else:
    st.info("Risk scores not available for heatmap.")

# --------------------------
# 11b) Heatmap — Department × Risk Band
# --------------------------
st.markdown("## 11b) Heatmap — Department × Risk Band")

if "_risk_score" in dff.columns and DEPT and DEPT in dff.columns:
    heat_dept = (
        dff.groupby([DEPT, "_risk_band_str"])
           .size()
           .unstack(fill_value=0)
           .sort_index()
    )
    heat_dept.index.name = "Department"
    heat_dept.columns.name = "Risk band"

    if not heat_dept.empty:
        fig_hdept = px.imshow(
            heat_dept,
            aspect="auto",
            labels={"x": "Risk band (I×L)", "y": "Department", "color": "Count"},
            title="Department × Risk band"
        )
        st.plotly_chart(fig_hdept, use_container_width=True)

        if AUTO_INSIGHTS:
            insight_heatmap_simple(heat_dept, "Department", "Risk band")
    else:
        st.info("No BCMS risk scores to plot for departments.")
else:
    st.info("Need Department column and BCMS risk scores for department heatmap.")

# --------------------------
# 12) Regression — Disruption vs Total Defects
# --------------------------
st.markdown("## 12) Regression — Disruption vs Total Defects")
if TOTAL_DEFECT and IMPACT_DAYS_COL and (TOTAL_DEFECT in dff.columns) and (IMPACT_DAYS_COL in dff.columns):
    x = safe_num(dff[TOTAL_DEFECT]).fillna(0)
    y = pd.to_numeric(
        dff[IMPACT_DAYS_COL].astype(str).str.replace(",", "", regex=False).str.extract(r"(-?\d+\.?\d*)", expand=False),
        errors="coerce"
    ).fillna(0)

    try:
        from numpy.linalg import lstsq
        Xmat = np.column_stack([np.ones_like(x), x])
        beta, *_ = lstsq(Xmat, y.values, rcond=None)
        y_hat = Xmat @ beta
        ss_res = np.sum((y.values - y_hat)**2)
        ss_tot = np.sum((y.values - y.values.mean())**2)
        r2 = 1 - (ss_res/ss_tot) if ss_tot > 0 else np.nan
        slope = float(beta[1])
        r = np.corrcoef(x.values, y.values)[0, 1] if len(x) > 1 else np.nan
    except Exception:
        slope, r2, r = None, None, np.nan

    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data"))
    if slope is not None:
        fig_sc.add_trace(go.Scatter(x=x, y=y_hat, mode="lines", name="Fit"))
    fig_sc.update_layout(
        xaxis_title="Total defects",
        yaxis_title="Disruption (days)",
        height=380,
        margin=dict(l=0, r=0, t=10, b=0)
    )
    st.plotly_chart(fig_sc, use_container_width=True)
    insight_regression("Total defects", "Disruption (days)", r, slope=slope, r2=r2)
else:
    st.info("Need TOTAL_DEFECT and disruption 'days' columns for regression.")

# --------------------------
# 13) Data Master Severity & Disruption by Location
# --------------------------
st.markdown("## 13) Data Master Severity & Disruption by Location")
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
# 14) Data Master (All Rows & Columns)
# --------------------------
st.markdown("## 14) Data Master (All Rows & Columns)")
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
