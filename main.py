# app/main.py
"""
Smart Data Processing & Analytics Assistant — Professional Dashboard Style (Option C)
Features:
 - Upload CSV/XLSX
 - Clean / Preprocess
 - Data Quality Score (DQS)
 - 8 smart chart suggestions + interactive builder
 - Dataset-aware Q&A (aggregation / top-k / simple forecast)
 - Voice output (browser TTS fallback, optional gTTS server-side)
 - Professional dashboard UI (cards, layout, sidebar)
"""
import os
import io
import re
import json
import html
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import streamlit.components.v1 as components
import streamlit as st
import time

import streamlit as st
import time
import time
import streamlit as st

# ================= SESSION INIT =================
if "page" not in st.session_state:
    st.session_state.page = "splash"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# ================= SPLASH SCREEN =================
def splash_screen():
    st.markdown(
        """
        <style>
        .splash-container{
            height:100vh;
            display:flex;
            flex-direction:column;
            justify-content:center;
            align-items:center;
            animation:fadeIn 1.2s ease-in-out;
        }
        @keyframes fadeIn{
            from{opacity:0; transform:translateY(40px);}
            to{opacity:1; transform:translateY(0);}
        }
        .title{
            font-size:48px;
            font-weight:700;
            text-align:center;
        }
        .subtitle{
            font-size:22px;
            margin-top:14px;
            color:#555;
            animation:fadeSub 2s ease-in-out;
        }
        @keyframes fadeSub{
            from{opacity:0;}
            to{opacity:1;}
        }
        </style>

        <div class="splash-container">
            <div class="title">
                📊 Smart Data Processing & Analytics Assistant
            </div>
            <div class="subtitle">
                Professional Analytics • AI Insights • Smart Charts
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    time.sleep(2.5)
    st.session_state.page = "login"
    st.rerun()


# ================= LOGIN SCREEN =================
def login_screen():
    st.markdown(
        """
        <style>
        .login-box{
            max-width:360px;
            margin:auto;
            margin-top:120px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    st.markdown("## 🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "sara" and password == "sara":
            st.session_state.logged_in = True
            st.session_state.page = "dashboard"
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.markdown("</div>", unsafe_allow_html=True)
# ================= ROUTER =================
if st.session_state.page == "splash":
    splash_screen()

elif st.session_state.page == "login":
    login_screen()

elif st.session_state.page == "dashboard":
    if not st.session_state.logged_in:
        st.session_state.page = "login"
        st.rerun()
        # ❌ BLOCK DASHBOARD UNTIL LOGIN
if st.session_state.page != "dashboard":
    st.stop()
try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

# Page config
st.set_page_config(page_title="Smart Analytics Assistant — Dashboard",
                   page_icon="📊",
                   layout="wide")

# Ensure folders
os.makedirs("data", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ---------------- CSS / Styling (professional) ----------------
st.markdown("""
<style>
:root{
  --card-bg: #ffffff;
  --muted: #6b7280;
  --accent: #0ea5a4;
  --card-radius: 12px;
  --shadow: 0 6px 18px rgba(14,21,47,0.06);
}
body {
  background-color: #f7fafc;
}
.header-row {
  display:flex; align-items:center; justify-content:space-between;
}
.app-title {font-size:28px; font-weight:700; color:#0f172a;}
.app-sub {color:var(--muted); margin-top:4px;}
.kpi {
  background:var(--card-bg); padding:18px; border-radius:12px; box-shadow: var(--shadow);
}
.section-card {background:var(--card-bg); padding:18px; border-radius:12px; box-shadow: var(--shadow); margin-bottom:14px;}
.small-muted {color:var(--muted); font-size:13px;}
.chart-card {background:var(--card-bg); padding:12px; border-radius:12px; box-shadow: var(--shadow);}
</style>
""", unsafe_allow_html=True)

# ---------- Top header ----------
col1, col2 = st.columns([3,1])
with col1:
    st.markdown('<div class="header-row"><div><div class="app-title">Smart Analytics Assistant</div><div class="app-sub">Professional Dashboard — Upload → Clean → Inspect → Ask → Export</div></div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div style="text-align:right"><img src="https://img.icons8.com/fluency/48/000000/data-configuration.png"/></div>', unsafe_allow_html=True)

st.markdown("")

# ---------- Sidebar controls ----------
with st.sidebar:
    st.markdown("## Upload & Controls")
    uploaded_file = st.file_uploader("Upload CSV / XLSX", type=["csv","xlsx"])
    st.markdown("---")
    auto_clean = st.checkbox("Auto-clean on upload", value=True)
    show_recommend = st.checkbox("Show chart recommendations", value=True)
    use_browser_tts = st.checkbox("Prefer browser TTS (fallback)", value=True)
    st.markdown("---")
  

# ---------------- Session state defaults ----------------
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None
if "last_answer_text" not in st.session_state:
    st.session_state.last_answer_text = ""

    

# ---------------- Helper functions (stable) ----------------
def standardize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_").lower() for c in df.columns]
    return df

def simple_clean(df):
    df = standardize_columns(df)
    df = df.drop_duplicates().reset_index(drop=True)
    for col in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                if df[col].mode().empty:
                    df[col].fillna("", inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        except Exception:
            df[col].fillna("", inplace=True)
    return df

def compute_quality_score(df_local):
    n = len(df_local)
    total_cells = df_local.size
    non_null_cells = df_local.count().sum()
    completeness = (non_null_cells / total_cells) * 100 if total_cells > 0 else 0
    dup_count = df_local.duplicated().sum()
    uniqueness = (1 - (dup_count / n)) * 100 if n > 0 else 100
    consistency_scores = []
    for col in df_local.columns:
        ser = df_local[col]
        if pd.api.types.is_numeric_dtype(ser):
            consistency_scores.append(100)
        else:
            parsed = pd.to_datetime(ser, errors='coerce')
            non_na = parsed.notna().sum()
            consistency_scores.append((non_na / len(ser)) * 100 if len(ser) > 0 else 100)
    consistency = np.mean(consistency_scores) if consistency_scores else 100
    validity_scores = []
    numeric_cols = df_local.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        ser = df_local[col].dropna()
        if len(ser) < 2:
            validity_scores.append(100)
            continue
        mean = ser.mean(); std = ser.std()
        low, high = mean - 4*std, mean + 4*std
        within = ser[(ser >= low) & (ser <= high)].count()
        validity_scores.append((within / len(ser)) * 100)
    validity = np.mean(validity_scores) if validity_scores else 100
    weights = {"completeness":0.35, "uniqueness":0.15, "consistency":0.25, "validity":0.25}
    overall = (completeness*weights["completeness"] +
               uniqueness*weights["uniqueness"] +
               consistency*weights["consistency"] +
               validity*weights["validity"])
    breakdown = {
        "completeness": round(completeness,1),
        "uniqueness": round(uniqueness,1),
        "consistency": round(consistency,1),
        "validity": round(validity,1),
        "overall": round(overall,1)
    }
    return round(overall,1), breakdown

# column detection helpers
def is_id_like(colname):
    if not isinstance(colname, str): return False
    col = colname.lower()
    patterns = ['id','row','index','sr','serial','no','order_no','orderid']
    return any(p in col for p in patterns)

BUSINESS_KEYWORDS = ['sales','profit','amount','amt','revenue','price','total','quantity','qty','count','value']

def choose_preferred(cols, df_local):
    if not cols:
        return None
    lower = [c.lower() for c in cols]
    for kw in BUSINESS_KEYWORDS:
        for i,c in enumerate(lower):
            if kw in c:
                return cols[i]
    numeric_candidates = [c for c in cols if not is_id_like(c)]
    if numeric_candidates:
        var_series = {}
        for c in numeric_candidates:
            try:
                var_series[c] = df_local[c].var() if pd.api.types.is_numeric_dtype(df_local[c]) else 0
            except Exception:
                var_series[c] = 0
        try:
            return max(var_series, key=var_series.get)
        except Exception:
            return numeric_candidates[0]
    return cols[0]

def find_best_columns(df_local):
    numeric_all = df_local.select_dtypes(include=['number']).columns.tolist()
    numeric = [c for c in numeric_all if not is_id_like(c)]
    categorical_all = df_local.select_dtypes(include=['object','category']).columns.tolist()
    categorical = [c for c in categorical_all if not is_id_like(c)]
    date_cols = []
    for c in df_local.columns:
        try:
            parsed = pd.to_datetime(df_local[c], errors='coerce')
            if parsed.notna().sum() > len(df_local) * 0.3:
                date_cols.append(c)
        except Exception:
            pass
    main_num = choose_preferred(numeric, df_local)
    nums = [c for c in numeric if c != main_num]
    second_num = nums[0] if nums else None
    cat_priority = ['category','segment','region','state','city','country','sub_category','sub-category','department']
    main_cat = None
    for k in cat_priority:
        for c in categorical:
            if k in c.lower():
                main_cat = c
                break
        if main_cat:
            break
    if not main_cat and categorical:
        filtered = {c: df_local[c].nunique() for c in categorical}
        filtered_ok = {k:v for k,v in filtered.items() if v <= 50}
        main_cat = (max(filtered_ok, key=filtered_ok.get) if filtered_ok else max(filtered, key=filtered.get))
    return {"date": date_cols, "numeric": numeric, "categorical": categorical,
            "main_num": main_num, "second_num": second_num, "main_cat": main_cat}

def recommend_mappings(info):
    recm = {}
    date_pref = None
    for d in info.get('date', []):
        if any(k in d.lower() for k in ['order','date','time','year','month','timestamp']):
            date_pref = d; break
    if not date_pref and info.get('date'):
        date_pref = info['date'][0] if info['date'] else None
    if date_pref and info['main_num']:
        recm['Line'] = (date_pref, info['main_num'])
        recm['Area'] = (date_pref, info['main_num'])
    if info['main_cat'] and info['main_num']:
        recm['Bar'] = (info['main_cat'], info['main_num'])
        recm['Box'] = (info['main_cat'], info['main_num'])
    if info['main_num'] and info['second_num']:
        recm['Scatter'] = (info['main_num'], info['second_num'])
    if info['main_num']:
        recm['Histogram'] = (info['main_num'],)
    if info['main_cat']:
        recm['Pie'] = (info['main_cat'],)
    if len(info['numeric']) >= 2:
        recm['Heatmap'] = (info['numeric'][0], info['numeric'][1])
    return recm

def splash_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.image("logo.png", width=140)
    st.markdown(
        "<h1 style='text-align:center;'>Smart Analytics Assistant</h1>",
        unsafe_allow_html=True
    )
    st.caption("Upload • Clean • Analyze • Ask")

    if st.button("🚀 Get Started"):
        st.session_state.page = "login"
        st.experimental_rerun()


def login_page():
    st.subheader("🔐 Login")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user and pwd:
            st.session_state.logged_in = True
            st.session_state.page = "instructions"
            st.experimental_rerun()
        else:
            st.error("Please enter login details")


def instructions_page():
    st.subheader("📘 Instructions")

    st.markdown("""
    **How to use this application**

    1️⃣ Upload CSV / Excel dataset  
    2️⃣ Clean missing & invalid values  
    3️⃣ Check Data Quality Score (DQS)  
    4️⃣ Explore auto & manual charts  
    5️⃣ Ask dataset-aware questions  
    """)

    if st.button("➡ Start Analysis"):
        st.session_state.page = "upload"
        st.experimental_rerun()

# ---------------- Upload & initial actions ----------------
uploaded = False
if uploaded_file:
    st.session_state.uploaded_name = uploaded_file.name
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state.raw_df = df
        uploaded = True
        if auto_clean:
            st.session_state.cleaned_df = simple_clean(df)
            st.success("Uploaded and auto-cleaned.")
        else:
            st.success("Uploaded. Click Clean Data to preprocess.")
    except Exception as e:
        st.error("Upload error: " + str(e))
        st.session_state.raw_df = None

# ---------- Main layout: left (analysis) and right (details) ----------
left, right = st.columns([3,1])

with left:
    # ---------- Data preview + cleaning ----------
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("1) Data Preview & Cleaning")
    if st.session_state.raw_df is None:
        st.info("Upload a CSV/XLSX file from the left sidebar to start.")
    else:
        st.write(f"**File:** {st.session_state.uploaded_name} — rows: {len(st.session_state.raw_df)}, cols: {len(st.session_state.raw_df.columns)}")
        st.dataframe(st.session_state.raw_df.head(6), height=200)
        if st.button("Clean Data"):
            try:
                st.session_state.cleaned_df = simple_clean(st.session_state.raw_df)
                base = st.session_state.uploaded_name.split('.')[0] if st.session_state.uploaded_name else "cleaned"
                path = f"data/cleaned_{base}.csv"
                try:
                    st.session_state.cleaned_df.to_csv(path, index=False)
                    st.info(f"Cleaned saved to {path}")
                except Exception:
                    st.info("Cleaned saved in memory.")
                st.success("Cleaning finished.")
            except Exception as e:
                st.error("Cleaning failed: " + str(e))
    st.markdown('</div>', unsafe_allow_html=True)

    # If cleaned, show KPIs
    if st.session_state.cleaned_df is not None:
        cleaned_df = st.session_state.cleaned_df
        # DQS + KPIs
        overall_score, dqs = compute_quality_score(cleaned_df)
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("2) Data Quality & Key Metrics")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f'<div class="kpi"><div style="font-size:12px;color:var(--muted)">DQS (0-100)</div><div style="font-size:20px;font-weight:700">{overall_score}</div></div>', unsafe_allow_html=True)
        with k2:
            rows = len(cleaned_df); cols = len(cleaned_df.columns)
            st.markdown(f'<div class="kpi"><div style="font-size:12px;color:var(--muted)">Rows</div><div style="font-size:20px;font-weight:700">{rows}</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="kpi"><div style="font-size:12px;color:var(--muted)">Columns</div><div style="font-size:20px;font-weight:700">{cols}</div></div>', unsafe_allow_html=True)
        with k4:
            nunique_total = sum(cleaned_df.nunique())
            st.markdown(f'<div class="kpi"><div style="font-size:12px;color:var(--muted)">Unique values (sum)</div><div style="font-size:20px;font-weight:700">{nunique_total}</div></div>', unsafe_allow_html=True)

        st.markdown("**DQS Breakdown**")
        st.write(f"- Completeness: {dqs['completeness']}%  •  Uniqueness: {dqs['uniqueness']}%  •  Consistency: {dqs['consistency']}%  •  Validity: {dqs['validity']}%")
        if dqs['completeness'] < 90: st.info("Recommendation: fill missing values or remove empty rows.")
        if dqs['uniqueness'] < 95: st.info("Recommendation: remove duplicates / check keys.")
        if dqs['consistency'] < 90: st.info("Recommendation: standardize dates & formats.")
        if dqs['validity'] < 90: st.info("Recommendation: check outliers & incorrect numeric entries.")
        st.markdown('</div>', unsafe_allow_html=True)

        # ---------- Chart recommendations ----------
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("3) Smart Chart Recommendations")
        col_info = find_best_columns(cleaned_df)
        mappings = recommend_mappings(col_info)

        if show_recommend:
            st.markdown("**Auto-suggested charts (editable)**")
            chart_list = ["Line","Bar","Scatter","Histogram","Pie","Area","Box","Heatmap"]
            rows = []
            for ch in chart_list:
                mp = mappings.get(ch)
                mapping_text = ", ".join(mp) if (mp and isinstance(mp, (list,tuple))) else (mp or "No automatic mapping")
                reason = {
                    "Line":"Trend over time (use if date present)",
                    "Bar":"Compare groups",
                    "Scatter":"Show correlation",
                    "Histogram":"Distribution",
                    "Pie":"Proportion (low unique)",
                    "Area":"Cumulative trend",
                    "Box":"Distribution & outliers by group",
                    "Heatmap":"Density between numeric columns"
                }.get(ch,"")
                rows.append((ch, mapping_text, reason))
            st.dataframe(pd.DataFrame(rows, columns=["Chart","Suggested mapping","Why"]), height=220)
        st.markdown("Edit below and generate preview")
        st.markdown('</div>', unsafe_allow_html=True)

        # ---------- Chart Builder ----------
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.subheader("4) Build & Preview Chart")
        chart_list = ["Line","Bar","Scatter","Histogram","Pie","Area","Box","Heatmap"]
        chosen_chart = st.selectbox("Chart type", chart_list, index=0)
        num_opts = col_info['numeric'] or list(cleaned_df.select_dtypes(include=[np.number]).columns)
        cat_opts = col_info['categorical'] or list(cleaned_df.select_dtypes(include=['object','category']).columns)
        date_opts = col_info['date'] or []
        def safe_index(lst, val):
            try: return lst.index(val)
            except Exception: return 0

        x_sel = y_sel = None
        nbins = 30
        agg = "sum"
        heatmap_bins = 10

        if chosen_chart == "Line":
            x_choices = date_opts or num_opts or list(cleaned_df.columns)
            y_choices = num_opts or list(cleaned_df.columns)
            default = mappings.get("Line", (None,None))
            x_sel = st.selectbox("X (date)", x_choices, index=safe_index(x_choices, default[0]))
            y_sel = st.selectbox("Y (numeric)", y_choices, index=safe_index(y_choices, default[1]))
            st.write("Line groups by month for trend.")
        elif chosen_chart == "Bar":
            x_choices = cat_opts or list(cleaned_df.columns)
            y_choices = num_opts or list(cleaned_df.columns)
            default = mappings.get("Bar", (None,None))
            x_sel = st.selectbox("X (category)", x_choices, index=safe_index(x_choices, default[0]))
            y_sel = st.selectbox("Y (numeric)", y_choices, index=safe_index(y_choices, default[1]))
            agg = st.selectbox("Aggregation", ["sum","mean","count"], index=0)
        elif chosen_chart == "Scatter":
            x_choices = num_opts or list(cleaned_df.columns)
            y_choices = num_opts or list(cleaned_df.columns)
            default = mappings.get("Scatter", (None,None))
            x_sel = st.selectbox("X (numeric)", x_choices, index=safe_index(x_choices, default[0]))
            y_sel = st.selectbox("Y (numeric)", y_choices, index=safe_index(y_choices, default[1]))
        elif chosen_chart == "Histogram":
            x_choices = num_opts or list(cleaned_df.columns)
            x_sel = st.selectbox("Column (numeric)", x_choices, index=0)
            nbins = st.slider("Bins", 5, 100, 30)
        elif chosen_chart == "Pie":
            x_choices = cat_opts or list(cleaned_df.columns)
            x_sel = st.selectbox("Category column", x_choices, index=0)
        elif chosen_chart == "Area":
            x_choices = date_opts or num_opts or list(cleaned_df.columns)
            y_choices = num_opts or list(cleaned_df.columns)
            x_sel = st.selectbox("X (date)", x_choices, index=0)
            y_sel = st.selectbox("Y (numeric)", y_choices, index=0)
        elif chosen_chart == "Box":
            x_choices = cat_opts or list(cleaned_df.columns)
            y_choices = num_opts or list(cleaned_df.columns)
            x_sel = st.selectbox("X (category)", x_choices, index=0)
            y_sel = st.selectbox("Y (numeric)", y_choices, index=0)
        elif chosen_chart == "Heatmap":
            num_choices = col_info['numeric'] or list(cleaned_df.select_dtypes(include=[np.number]).columns)
            if len(num_choices) >= 2:
                x_sel = st.selectbox("X (numeric)", options=num_choices, index=0)
                y_sel = st.selectbox("Y (numeric)", options=num_choices, index=1 if len(num_choices)>1 else 0)
                heatmap_bins = st.slider("Bins per axis", 5, 50, 10)
            else:
                st.info("Need at least 2 numeric columns for heatmap.")

        if st.button("Generate Chart"):
            try:
                fig = None
                if chosen_chart == "Line" and x_sel and y_sel:
                    tmp = cleaned_df.copy()
                    tmp['_time'] = pd.to_datetime(tmp[x_sel], errors='coerce')
                    tmp = tmp.dropna(subset=['_time'])
                    if tmp.empty:
                        st.error("X column could not be parsed as dates.")
                    else:
                        tmpm = tmp.set_index('_time').resample('M')[y_sel].sum().reset_index()
                        fig = px.line(tmpm, x='_time', y=y_sel, title=f"{y_sel} over time ({x_sel})")
                elif chosen_chart == "Bar" and x_sel and y_sel:
                    if agg == "sum":
                        df_temp = cleaned_df.groupby(x_sel)[y_sel].sum().reset_index()
                    elif agg == "mean":
                        df_temp = cleaned_df.groupby(x_sel)[y_sel].mean().reset_index()
                    else:
                        df_temp = cleaned_df.groupby(x_sel)[y_sel].count().reset_index(name='count')
                        fig = px.bar(df_temp, x=x_sel, y='count', title=f"Count by {x_sel}")
                    if fig is None:
                        fig = px.bar(df_temp, x=x_sel, y=y_sel, title=f"{y_sel} by {x_sel}")
                elif chosen_chart == "Scatter" and x_sel and y_sel:
                    fig = px.scatter(cleaned_df, x=x_sel, y=y_sel, trendline="ols", title=f"{y_sel} vs {x_sel}")
                elif chosen_chart == "Histogram" and x_sel:
                    fig = px.histogram(cleaned_df, x=x_sel, nbins=nbins, title=f"Distribution of {x_sel}")
                elif chosen_chart == "Pie" and x_sel:
                    tmp = cleaned_df[x_sel].value_counts().reset_index()
                    tmp.columns = [x_sel, 'count']
                    fig = px.pie(tmp, names=x_sel, values='count', title=f"Proportion of {x_sel}")
                elif chosen_chart == "Area" and x_sel and y_sel:
                    tmp = cleaned_df.copy()
                    tmp['_time'] = pd.to_datetime(tmp[x_sel], errors='coerce')
                    tmp = tmp.dropna(subset=['_time'])
                    tmpm = tmp.set_index('_time').resample('M')[y_sel].sum().reset_index()
                    fig = px.area(tmpm, x='_time', y=y_sel, title=f"{y_sel} area over time")
                elif chosen_chart == "Box" and x_sel and y_sel:
                    fig = px.box(cleaned_df, x=x_sel, y=y_sel, title=f"Box plot of {y_sel} by {x_sel}")
                elif chosen_chart == "Heatmap" and x_sel and y_sel:
                    x_bins = pd.cut(cleaned_df[x_sel], bins=heatmap_bins)
                    y_bins = pd.cut(cleaned_df[y_sel], bins=heatmap_bins)
                    pivot = pd.crosstab(x_bins, y_bins)
                    fig = px.imshow(pivot.values, labels=dict(x=str(y_sel), y=str(x_sel)),
                                    x=[str(i) for i in pivot.columns], y=[str(i) for i in pivot.index],
                                    title=f"Heatmap: {x_sel} vs {y_sel}")
                else:
                    st.error("Cannot build chart with chosen columns.")
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error("Chart error: " + str(e))
        st.markdown('</div>', unsafe_allow_html=True)
# ============================================================
# 5) ASK – TRUE DATASET-AWARE SALES AI (FINAL STABLE VERSION)
# ============================================================
cleaned_df = st.session_state.get("cleaned_df", None)

cleaned_df = st.session_state.get("cleaned_df", None)

if cleaned_df is None or cleaned_df.empty:
    st.info("Please upload and clean a dataset to enable Ask AI.")
    st.stop()

if cleaned_df is not None and not cleaned_df.empty:

    import re
    import numpy as np
    import pandas as pd
    import streamlit as st

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("5) Ask (Dataset-aware AI assistant)")
    st.write(
        "Examples: "
        "'total sales', "
        "'total sales in india', "
        "'march sales', "
        "'2019 sales', "
        "'thursday sales', "
        "'bakery sales', "
        "'compare 2019 vs 2020 sales'"
    )

    question = st.text_input("Ask a question (English)")

    # -----------------------------
    # MONTH & DAY MAPS
    # -----------------------------
    MONTH_MAP = {
        "jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,
        "apr":4,"april":4,"may":5,"jun":6,"june":6,
        "jul":7,"july":7,"aug":8,"august":8,
        "sep":9,"september":9,"oct":10,"october":10,
        "nov":11,"november":11,"dec":12,"december":12
    }

    DAY_MAP = {
        "monday":0,"tuesday":1,"wednesday":2,
        "thursday":3,"friday":4,"saturday":5,"sunday":6
    }

    # -----------------------------
    # HELPER FUNCTIONS
    # -----------------------------
    def extract_years(text):
        return [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", text)]

    def extract_months(text):
        return [v for k,v in MONTH_MAP.items() if k in text]

    def extract_days(text):
        return [v for k,v in DAY_MAP.items() if k in text]

    def detect_sales_column(df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if any(k in col.lower() for k in ["sale","revenue","amount","price","profit"]):
                return col
        return numeric_cols[0] if len(numeric_cols) > 0 else None

    def detect_date_column(df):
        for col in df.columns:
            try:
                pd.to_datetime(df[col])
                return col
            except:
                continue
        return None

    def apply_value_filters(df, text):
        text_words = set(re.findall(r"\b\w+\b", text.lower()))

        for col in df.select_dtypes(include=["object","category"]).columns:
            values = df[col].dropna().astype(str).unique()
            values = sorted(values, key=lambda x: -len(x))  # longest first

            for v in values:
                v_words = set(re.findall(r"\b\w+\b", v.lower()))
                if v_words.issubset(text_words):
                    return df[df[col].astype(str).str.lower() == v.lower()], v

        return df, None

    # -----------------------------
    # MAIN LOGIC
    # -----------------------------
    if st.button("Get Answer"):

        q = question.lower().strip()
        df = cleaned_df.copy()

        date_col = detect_date_column(df)
        sales_col = detect_sales_column(df)

        if sales_col is None:
            st.error("No numeric sales column found in dataset.")
            st.stop()

        if date_col:
            df["_dt"] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=["_dt"])

        # value filtering (country / category / product etc.)
        df, matched_value = apply_value_filters(df, q)

        years = extract_years(q)
        months = extract_months(q)
        days = extract_days(q)

        if years:
            df = df[df["_dt"].dt.year.isin(years)]

        if months:
            df = df[df["_dt"].dt.month.isin(months)]

        if days:
            df = df[df["_dt"].dt.weekday.isin(days)]

        # ---------------- TOTAL ----------------
        if "compare" not in q and "vs" not in q:
            total = df[sales_col].sum()

            label = f"Total {sales_col}"
            if matched_value:
                label += f" in {matched_value}"

            st.success(f"{label} = {total:,.2f}")
            st.caption("Calculated dynamically from dataset based on your question.")
            st.stop()

        # ---------------- COMPARISON ----------------
        if ("compare" in q or "vs" in q) and len(years) >= 2:
            y1, y2 = years[:2]
            v1 = df[df["_dt"].dt.year == y1][sales_col].sum()
            v2 = df[df["_dt"].dt.year == y2][sales_col].sum()

            st.info(f"{sales_col} in {y1}: {v1:,.2f}")
            st.info(f"{sales_col} in {y2}: {v2:,.2f}")
            st.success(f"Difference: {v2 - v1:,.2f}")
            st.stop()

        st.warning("Question understood, but required values were not found in dataset.")

    st.markdown("</div>", unsafe_allow_html=True)