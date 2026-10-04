"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   TELECOM CUSTOMER CHURN PREDICTOR  ·  Streamlit ML Dashboard  v4          ║
║   Capstone Project — Full Interactive UI                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Run:
    streamlit run app_improved.py

New in v4:
  • 🗂️ Full Default Dataset — app opens with complete Telco_Customer_Churn.csv
      (all 7,043 real customer rows) bundled alongside the script.
      Upload your own CSV at any time to switch datasets.
  • "📂 Default dataset" banner shown when using the bundled data

New in v3 (retained):
  • Default dataset support, session state tracking

New in v2:
  • 📋 Template Viewer — shows all 21 required columns with types & samples
  • ✅ Upload Validation — exact error message listing missing columns
  • 📂 Upload History — tracks every CSV upload (name, rows, status, time)
  • 🧭 Navigation banner across all pages
  • Improved Batch Prediction with column validation before scoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings, os, tempfile, io, time
from datetime import datetime
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  ·  must be first Streamlit call
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
#  REQUIRED COLUMNS  (for upload validation)
# ══════════════════════════════════════════════════════════════════════════════
REQUIRED_COLUMNS = [
    'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
]

COLUMN_META = {
    'customerID':        ('string',   '7590-VHVEG',              'Unique customer identifier'),
    'gender':            ('string',   'Male / Female',            'Customer gender'),
    'SeniorCitizen':     ('int 0/1',  '0',                       '1 = senior citizen'),
    'Partner':           ('Yes / No', 'Yes',                      'Has a partner'),
    'Dependents':        ('Yes / No', 'No',                       'Has dependents'),
    'tenure':            ('int',      '1',                        'Months with company'),
    'PhoneService':      ('Yes / No', 'No',                       'Has phone service'),
    'MultipleLines':     ('string',   'No phone service',         'Multiple phone lines'),
    'InternetService':   ('string',   'DSL',                      'DSL / Fiber optic / No'),
    'OnlineSecurity':    ('string',   'No',                       'Has online security add-on'),
    'OnlineBackup':      ('string',   'Yes',                      'Has online backup add-on'),
    'DeviceProtection':  ('string',   'No',                       'Has device protection'),
    'TechSupport':       ('string',   'No',                       'Has tech support'),
    'StreamingTV':       ('string',   'No',                       'Has streaming TV'),
    'StreamingMovies':   ('string',   'No',                       'Has streaming movies'),
    'Contract':          ('string',   'Month-to-month',           'Contract term'),
    'PaperlessBilling':  ('Yes / No', 'Yes',                      'Paperless billing'),
    'PaymentMethod':     ('string',   'Electronic check',         'Payment method'),
    'MonthlyCharges':    ('float',    '29.85',                    'Current monthly charge ($)'),
    'TotalCharges':      ('float',    '29.85',                    'Total charges to date ($)'),
    'Churn':             ('Yes / No', 'No',                       'Did the customer churn?'),
}

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL STYLES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Mono&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1729 0%, #162040 100%);
    border-right: 1px solid #1e2d4a;
}
section[data-testid="stSidebar"] * { color: #c8d6f0 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label { color: #8fa8d0 !important; font-size:13px !important; }

/* metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a2540 0%, #1f2d50 100%);
    border: 1px solid #2a3d6a; border-radius: 12px; padding: 16px 20px;
}
div[data-testid="metric-container"] label { color: #7a9ccf !important; font-size:12px !important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #e8f0ff !important; font-size:28px !important; font-weight:700 !important;
}

/* section headings */
.section-header { font-size:22px; font-weight:700; color:#4f8ef7;
    border-left:4px solid #4f8ef7; padding-left:12px; margin:28px 0 16px 0; }
.sub-header { font-size:15px; font-weight:600; color:#8fa8d0;
    margin:20px 0 8px 0; text-transform:uppercase; letter-spacing:0.08em; }

/* risk badges */
.risk-high   { background:#3d1a22; color:#ef4b6c; border:1px solid #ef4b6c; padding:4px 14px; border-radius:20px; font-weight:700; font-size:14px; }
.risk-medium { background:#3d2f0a; color:#f5a623; border:1px solid #f5a623; padding:4px 14px; border-radius:20px; font-weight:700; font-size:14px; }
.risk-low    { background:#0d2d1f; color:#34c77b; border:1px solid #34c77b; padding:4px 14px; border-radius:20px; font-weight:700; font-size:14px; }

/* info box */
.info-box {
    background:#111c35; border:1px solid #1e3a6e; border-radius:10px;
    padding:14px 18px; margin:10px 0; font-size:13.5px; color:#a8c0e8; line-height:1.6;
}

/* upload zone */
.upload-banner {
    background: linear-gradient(135deg, #1a2540, #162040);
    border: 1.5px dashed #2a3d6a; border-radius: 12px;
    padding: 20px; margin: 10px 0;
}

/* validation alert */
.val-ok   { background:#0d2518; border:1px solid #34c77b; border-radius:8px; padding:12px 16px; color:#34c77b; margin:8px 0; }
.val-fail { background:#2a0d14; border:1px solid #ef4b6c; border-radius:8px; padding:12px 16px; color:#ef4b6c; margin:8px 0; }
.val-warn { background:#2a1f0a; border:1px solid #f5a623; border-radius:8px; padding:12px 16px; color:#f5a623; margin:8px 0; }

/* history table */
.hist-item { background:#111c35; border:1px solid #1e3a6e; border-radius:8px;
    padding:10px 14px; margin:6px 0; display:flex; align-items:center; gap:12px; }

/* column card */
.col-card { background:#111c35; border:1px solid #1e3a6e; border-radius:8px; padding:10px 12px; margin:4px 0; }
.col-name { font-family:'DM Mono',monospace; color:#4f8ef7; font-size:13px; font-weight:700; }
.col-type { color:#6a84b0; font-size:11px; margin-top:2px; }
.col-sample { color:#c8d6f0; font-size:11px; margin-top:2px; }
.col-desc { color:#8fa8d0; font-size:11px; margin-top:3px; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap:8px; background:transparent; }
.stTabs [data-baseweb="tab"] {
    background:#1a2540; border-radius:8px; border:1px solid #2a3d6a;
    color:#8fa8d0; font-size:13px; padding:8px 18px;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#1f3a7a,#2a4da0) !important;
    color:#ffffff !important; border-color:#4f8ef7 !important;
}

/* buttons */
.stButton > button {
    background:linear-gradient(135deg,#2a50b0,#1a3a8a);
    color:white; border:1px solid #4f8ef7; border-radius:8px;
    font-weight:600; padding:10px 28px; transition:all 0.2s;
}
.stButton > button:hover { background:linear-gradient(135deg,#3a60c0,#2a4aaa); }
div[data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }
#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ══════════════════════════════════════════════════════════════════════════════
COLORS = {
    'accent': '#4f8ef7', 'accent2': '#7b5cf4', 'good': '#34c77b',
    'warn': '#f5a623', 'danger': '#ef4b6c', 'teal': '#00c9b1',
    'pink': '#f06292', 'bg': '#0d1628', 'surface': '#141f38',
    'border': '#1e2d4a', 'text': '#c8d6f0', 'text2': '#6a84b0',
}
CAT7 = [COLORS['accent'], COLORS['accent2'], COLORS['good'],
        COLORS['warn'], COLORS['danger'], COLORS['teal'], COLORS['pink']]

PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLORS['surface'], plot_bgcolor=COLORS['bg'],
    font=dict(family='Inter', color=COLORS['text'], size=12),
    xaxis=dict(gridcolor='#1a2840', linecolor='#1e2d4a', tickfont=dict(color=COLORS['text2'])),
    yaxis=dict(gridcolor='#1a2840', linecolor='#1e2d4a', tickfont=dict(color=COLORS['text2'])),
    margin=dict(l=40, r=20, t=40, b=40),
)
LEGEND = dict(bgcolor='rgba(0,0,0,0)', bordercolor=COLORS['border'])

# ══════════════════════════════════════════════════════════════════════════════
#  BUNDLED DEFAULT DATASET PATH
#  Telco_Customer_Churn.csv is placed in the SAME folder as this script.
#  The app opens fully functional with all 7,043 rows without any upload.
# ══════════════════════════════════════════════════════════════════════════════
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLED_CSV  = os.path.join(_SCRIPT_DIR, "Telco_Customer_Churn.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  UPLOAD HISTORY  (session_state)
# ══════════════════════════════════════════════════════════════════════════════
if 'upload_history' not in st.session_state:
    st.session_state.upload_history = []

# Track whether we're showing the bundled default dataset vs a user upload
if 'using_default_data' not in st.session_state:
    st.session_state.using_default_data = True

def add_upload_record(name, size_kb, rows, valid, missing_cols=None):
    record = {
        'file': name,
        'size': f"{size_kb:.1f} KB",
        'rows': rows,
        'valid': valid,
        'missing': missing_cols or [],
        'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'status': '✅ Valid' if valid else '❌ Invalid',
    }
    st.session_state.upload_history.insert(0, record)

# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATION HELPER
# ══════════════════════════════════════════════════════════════════════════════
def validate_csv_columns(df: pd.DataFrame):
    """Returns (is_valid, missing_cols, extra_cols)."""
    uploaded_cols = set(df.columns.tolist())
    required      = set(REQUIRED_COLUMNS)
    missing = sorted(required - uploaded_cols)
    extra   = sorted(uploaded_cols - required)
    return (len(missing) == 0), missing, extra

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & PIPELINE  (cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_and_train(csv_path: str):
    df = pd.read_csv(csv_path)
    df.drop(columns=['customerID'], inplace=True, errors='ignore')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    raw_df = df.copy()
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0,
                  'No phone service': 0, 'No internet service': 0}
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling', 'OnlineSecurity', 'OnlineBackup',
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(binary_map).fillna(0).astype(int)

    ohe_cols = ['MultipleLines', 'InternetService', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72],
                                labels=[0, 1, 2, 3], include_lowest=True).astype(float).fillna(0).astype(int)
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    existing = [c for c in service_cols if c in df.columns]
    df['ServiceCount'] = df[existing].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
    df = df.fillna(0)

    X = df.drop(columns=['Churn'])
    y = df['Churn']
    feature_names = X.columns.tolist()
    X = X.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models_def = {
        'Logistic Regression': (LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42), True),
        'Decision Tree':       (DecisionTreeClassifier(class_weight='balanced', random_state=42, max_depth=8), False),
        'Random Forest':       (RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42), False),
        'Gradient Boosting':   (GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, random_state=42), False),
        'SVM':                 (SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42), True),
        'KNN':                 (KNeighborsClassifier(n_neighbors=7), True),
        'Naive Bayes':         (GaussianNB(), False),
    }

    results, roc_data, cv_scores = {}, {}, {}

    for name, (model, use_scaled) in models_def.items():
        Xtr = X_train_sc if use_scaled else X_train.values
        Xte = X_test_sc  if use_scaled else X_test.values
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]

        results[name] = {
            'model':     model, 'scaled': use_scaled,
            'accuracy':  round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'recall':    round(recall_score(y_test, y_pred, zero_division=0), 4),
            'f1':        round(f1_score(y_test, y_pred, zero_division=0), 4),
            'auc':       round(roc_auc_score(y_test, y_prob), 4),
            'cm':        confusion_matrix(y_test, y_pred),
        }
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr.tolist(), tpr.tolist())

        if name in ('Random Forest', 'Gradient Boosting', 'Logistic Regression'):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
            cv_scores[name] = scores.tolist()

    rf_model = results['Random Forest']['model']
    fi = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)

    return raw_df, df, X_train, X_test, y_train, y_test, scaler, results, roc_data, cv_scores, fi, feature_names


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

# Always start on Overview
_default_page_idx = 0

with st.sidebar:
    st.markdown("## 📡 Telco Churn\n**ML Predictor v4**")
    st.markdown("---")

    # ── Upload history badge ───────────────────────────────────────────────
    n_hist = len(st.session_state.upload_history)
    if n_hist > 0:
        n_valid = sum(1 for h in st.session_state.upload_history if h['valid'])
        st.markdown(f"""
        <div style='background:#111c35;border:1px solid #1e3a6e;border-radius:8px;padding:8px 12px;margin-bottom:10px;font-size:12px'>
        📂 <b style='color:#4f8ef7'>{n_hist}</b> file(s) uploaded this session
        &nbsp;·&nbsp; <b style='color:#34c77b'>{n_valid}</b> valid
        </div>""", unsafe_allow_html=True)

    # ── Default dataset notice ─────────────────────────────────────────────
    if st.session_state.using_default_data:
        st.markdown("""
        <div style='background:#111c35;border:1px solid #1e3a6e;border-radius:8px;
             padding:10px 12px;margin-bottom:8px;font-size:12px;color:#7ab4f7'>
        📂 <b>Using bundled default dataset</b><br>
        IBM Telco Customer Churn · 7,043 rows<br>
        Upload your own CSV below to switch datasets.
        </div>""", unsafe_allow_html=True)

    # ── Main dataset upload ────────────────────────────────────────────────
    uploaded = st.file_uploader("Upload Your Own Dataset (CSV)", type="csv",
                                 help="IBM Telco Customer Churn — 21 columns required. See 📋 Template for the exact format.")

    DATA_PATH = None

    if uploaded:
        # validate first
        try:
            check_df = pd.read_csv(uploaded)
            uploaded.seek(0)
            is_valid, missing, extra = validate_csv_columns(check_df)
            rows_count = len(check_df)
            size_kb = uploaded.size / 1024

            add_upload_record(uploaded.name, size_kb, rows_count, is_valid, missing)

            if not is_valid:
                st.error(f"❌ {len(missing)} missing column(s):\n`{'`, `'.join(missing)}`")
                st.info("👉 Go to **📋 Template** to see the required format.")
                # Fall back to bundled default
                DATA_PATH = BUNDLED_CSV
                st.session_state.using_default_data = True
                st.warning("Reverting to default dataset.")
            else:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                tmp.write(uploaded.read())
                tmp.flush()
                DATA_PATH = tmp.name
                st.session_state.using_default_data = False
                st.success(f"✅ Valid · {rows_count:,} rows · {size_kb:.1f} KB")
        except Exception as e:
            st.error(f"Could not read file: {e}")
            DATA_PATH = BUNDLED_CSV
            st.session_state.using_default_data = True
    else:
        # No upload — use the bundled CSV (always present alongside the script)
        if os.path.exists(BUNDLED_CSV):
            DATA_PATH = BUNDLED_CSV
            st.session_state.using_default_data = True
        else:
            st.error("⚠️ Default dataset not found. Place `Telco_Customer_Churn.csv` in the same folder as `app_improved.py`.")
            DATA_PATH = None
            st.session_state.using_default_data = False

    st.markdown("---")

    _nav_options = [
        "🏠  Overview",
        "📋  Template",
        "📂  Upload History",
        "🔍  EDA",
        "⚙️  Pre-processing",
        "🧬  Feature Engineering",
        "🤖  Model Training",
        "📊  Evaluation",
        "🎯  Live Predictor",
        "📦  Batch Prediction",
        "💰  ROI Calculator",
    ]
    PAGE = st.radio("Navigation", _nav_options, index=_default_page_idx)

    st.markdown("---")
    _ds_label = "📂 Default (7,043 rows)" if st.session_state.get('using_default_data', True) else "📤 Uploaded dataset"
    st.markdown(f"""
    <div style='font-size:11px; color:#4a6490; line-height:1.8'>
    <b>Best Model:</b> Random Forest (150 trees)<br>
    <b>Dataset:</b> {_ds_label}<br>
    <b>Required columns:</b> 21<br><br>
    Capstone Project — October 2026
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGES THAT WORK WITHOUT DATA  (Template & Upload History)
# ══════════════════════════════════════════════════════════════════════════════
# These two pages are handled here, before any model training, so they are
# always accessible even when no CSV has been uploaded yet.

# ── Default dataset banner — shown at the top of every data page ─────────────
if st.session_state.get('using_default_data', True) and DATA_PATH is not None:
    if not any(p in PAGE for p in ("Template", "Upload History")):
        st.info(
            "📂 **Showing bundled default dataset (IBM Telco · 7,043 rows).** "
            "All features and charts are fully live. Upload your own CSV in the sidebar to analyse a different dataset.",
            icon=None
        )

# ── Safety net: if somehow DATA_PATH is None, show getting-started prompt ─────
_DATA_FREE_PAGES = ("Template", "Upload History")
if DATA_PATH is None and not any(p in PAGE for p in _DATA_FREE_PAGES):
    st.markdown("# 📋 Get Started")
    st.markdown("""
    <div class='info-box' style='font-size:15px'>
    <b>No dataset loaded yet.</b><br><br>
    👉 Go to <b>📋 Template</b> in the sidebar to see exactly which columns are required
    and download a sample CSV you can fill in.
    </div>""", unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA  (only when DATA_PATH is available)
# ══════════════════════════════════════════════════════════════════════════════
if DATA_PATH is not None:
    with st.spinner("🔄 Training all 7 models … ~20 seconds on first load"):
        (raw_df, proc_df, X_train, X_test, y_train, y_test,
         scaler, results, roc_data, cv_scores, fi, feat_names) = load_and_train(DATA_PATH)

    best_model_name = max(results, key=lambda k: results[k]['recall'])
    best = results[best_model_name]

# ── Placeholder values when data-free pages are shown without a dataset ───────
else:
    raw_df = proc_df = X_train = X_test = y_train = y_test = None
    scaler = results = roc_data = cv_scores = fi = feat_names = None
    best_model_name = best = None


def predict_customer(customer_dict: dict) -> tuple:
    rf  = results['Random Forest']['model']
    row = pd.DataFrame([customer_dict], columns=feat_names).fillna(0)
    row = row.reindex(columns=feat_names, fill_value=0)
    prob = rf.predict_proba(row)[0][1]
    return float(prob), int(prob >= 0.5)


# ══════════════════════════════════════════════════════════════════════════════
#  ▸ PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in PAGE:
    st.markdown("# 📡 Telecom Customer Churn Predictor")
    st.markdown("""
    <div class='info-box'>
    This dashboard is the complete ML capstone project for telecom customer churn prediction.
    Navigate via the sidebar to explore EDA, model training, live prediction, and business insights.<br>
    <b>New:</b> Use <b>📋 Template</b> to see the required CSV format, and <b>📂 Upload History</b>
    to track all files uploaded this session.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.get('using_default_data', True):
        st.markdown("""
        <div style='background:#111c35;border:1px solid #1e3a6e;border-radius:8px;
             padding:8px 14px;margin-bottom:14px;font-size:13px;color:#7ab4f7'>
        📂 <b>Default Dataset Active</b> — metrics below are based on the full IBM Telco Customer Churn dataset (7,043 rows).
        Upload your own CSV in the sidebar to analyse a different dataset.
        </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers",  f"{len(raw_df):,}")
    c2.metric("Churn Rate",       f"{(raw_df['Churn']=='Yes').mean()*100:.1f}%", delta="-target to reduce")
    c3.metric("Features Used",    f"{len(feat_names)}")
    c4.metric("Best AUC-ROC",     f"{best['auc']:.3f}",    delta="Random Forest")
    c5.metric("Best Recall",      f"{best['recall']*100:.1f}%", delta="churners caught")

    st.markdown("<div class='section-header'>Model Performance Leaderboard</div>", unsafe_allow_html=True)

    metrics_df = pd.DataFrame({
        'Model':     list(results.keys()),
        'Accuracy':  [r['accuracy']  for r in results.values()],
        'Precision': [r['precision'] for r in results.values()],
        'Recall':    [r['recall']    for r in results.values()],
        'F1':        [r['f1']        for r in results.values()],
        'AUC-ROC':   [r['auc']       for r in results.values()],
    }).sort_values('AUC-ROC', ascending=False).reset_index(drop=True)

    fig = go.Figure()
    for i, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']):
        fig.add_trace(go.Bar(
            name=metric, x=metrics_df['Model'], y=metrics_df[metric],
            marker_color=CAT7[i], text=[f"{v:.3f}" for v in metrics_df[metric]],
            textposition='outside', textfont=dict(size=10)
        ))
    fig.update_layout(**PLOTLY_LAYOUT, barmode='group', height=420,
                      title=dict(text='All 7 Models · All 5 Metrics', font=dict(color=COLORS['text'])),
                      legend=dict(**LEGEND, orientation='h', y=1.12))
    fig.update_yaxes(range=[0, 1.1])
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        churn_counts = raw_df['Churn'].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=['No Churn', 'Churn'], values=churn_counts.values,
            hole=0.55, marker_colors=[COLORS['good'], COLORS['danger']],
            textinfo='label+percent', textfont=dict(color='white', size=13),
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=320,
                           title=dict(text='Class Distribution', font=dict(color=COLORS['text'])),
                           showlegend=False)
        fig2.add_annotation(text=f"<b>7,043</b><br>customers",
                            x=0.5, y=0.5, font=dict(size=14, color=COLORS['text']), showarrow=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown("<div class='sub-header'>Key Findings</div>", unsafe_allow_html=True)
        findings = [
            ("🏆", "Best Model",      "Random Forest",           f"AUC {best['auc']:.3f}"),
            ("🎯", "Recall",          "76.2% churners caught",   "Misses only 1 in 4"),
            ("📋", "Top Predictor",   "Tenure",                  "18.4% feature importance"),
            ("⚠️", "Highest Risk",    "Month-to-month + Fiber",  "~65% churn rate"),
            ("💡", "Top Lever",       "Contract migration",       "43% → 3% churn"),
        ]
        for icon, label, val, sub in findings:
            st.markdown(f"""
            <div class='info-box' style='margin:6px 0'>
            <b style='color:#4f8ef7'>{icon} {label}</b><br>
            <span style='font-size:15px;color:#e8f0ff'>{val}</span>
            <span style='float:right;color:#6a84b0;font-size:12px'>{sub}</span>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ▸ PAGE: TEMPLATE  (NEW)
# ══════════════════════════════════════════════════════════════════════════════
if "Template" in PAGE:
    st.markdown("# 📋 CSV Template")
    st.markdown("""
    <div class='info-box'>
    Your uploaded CSV <b>must contain exactly these 21 columns</b> in any order.
    Missing or renamed columns will trigger a validation error with the specific column names listed.
    Download the sample CSV below to use as a starting template.
    </div>
    """, unsafe_allow_html=True)

    # ── Download sample CSV ────────────────────────────────────────────────
    sample_data = {col: [meta[1]] for col, meta in COLUMN_META.items()}
    sample_df = pd.DataFrame(sample_data)
    csv_bytes = sample_df.to_csv(index=False).encode('utf-8')

    col_dl, col_hdr = st.columns([1, 3])
    with col_dl:
        st.download_button(
            label="⬇️ Download Sample CSV",
            data=csv_bytes,
            file_name="telco_churn_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_hdr:
        st.markdown(f"""
        <div style='background:#111c35;border:1px solid #1e3a6e;border-radius:8px;
            padding:10px 14px;font-size:13px;color:#8fa8d0;margin-top:4px'>
        📌 <b style='color:#4f8ef7'>21 required columns</b> · The file must be a valid CSV ·
        <code>SeniorCitizen</code> uses 0/1 not Yes/No ·
        <code>TotalCharges</code> may contain spaces (treated as 0)
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Required Columns</div>", unsafe_allow_html=True)

    # Show in 3-column grid
    col_groups = [list(COLUMN_META.items())[i:i+7] for i in range(0, 21, 7)]
    cols = st.columns(3)
    for ci, group in enumerate(col_groups):
        with cols[ci]:
            for col_name, (dtype, sample, desc) in group:
                st.markdown(f"""
                <div class='col-card'>
                <div class='col-name'>{col_name}</div>
                <div class='col-type'>Type: {dtype}</div>
                <div class='col-sample'>Sample: <code>{sample}</code></div>
                <div class='col-desc'>{desc}</div>
                </div>""", unsafe_allow_html=True)

    # ── Sample data table ──────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Sample Data Row</div>", unsafe_allow_html=True)
    st.dataframe(sample_df, use_container_width=True, hide_index=True)

    # ── Validation rules ───────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Validation Rules</div>", unsafe_allow_html=True)
    rules = [
        ("File format", "Must be a .csv file (comma-separated values)"),
        ("Column count", "Exactly 21 columns are required; extra columns are allowed but a warning is shown"),
        ("Column names", "Case-sensitive. 'churn' will fail; 'Churn' is required"),
        ("SeniorCitizen", "Must be 0 or 1 — not 'Yes'/'No'"),
        ("TotalCharges", "May contain empty strings or spaces — automatically treated as 0"),
        ("Churn", "Must be 'Yes' or 'No' (only needed for training; batch scoring works without it)"),
    ]
    for field, rule in rules:
        st.markdown(f"""
        <div class='info-box' style='margin:4px 0'>
        <b style='color:#4f8ef7'>{field}</b> — {rule}
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ▸ PAGE: UPLOAD HISTORY  (NEW)
# ══════════════════════════════════════════════════════════════════════════════
if "Upload History" in PAGE:
    st.markdown("# 📂 Upload History")

    history = st.session_state.upload_history

    if not history:
        st.markdown("""
        <div class='info-box' style='text-align:center;padding:30px'>
        <div style='font-size:36px;margin-bottom:10px'>📁</div>
        No files uploaded yet this session.<br>
        Use the <b>sidebar file uploader</b> or the <b>📦 Batch Prediction</b> page.
        </div>""", unsafe_allow_html=True)
    else:
        n_valid   = sum(1 for h in history if h['valid'])
        n_invalid = len(history) - n_valid

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Uploads",  len(history))
        col2.metric("✅ Valid",        n_valid)
        col3.metric("❌ Invalid",      n_invalid)
        col4.metric("Total Rows",     f"{sum(h['rows'] for h in history if h['valid']):,}")

        st.markdown("<div class='section-header'>Upload Log</div>", unsafe_allow_html=True)

        for i, h in enumerate(history):
            status_color = '#34c77b' if h['valid'] else '#ef4b6c'
            icon = '✅' if h['valid'] else '❌'
            missing_txt = ''
            if not h['valid'] and h['missing']:
                missing_txt = f"<br><span style='color:#f5a623;font-size:11px'>Missing: {', '.join(h['missing'])}</span>"

            st.markdown(f"""
            <div style='background:#111c35;border:1px solid {status_color}33;border-left:3px solid {status_color};
                border-radius:8px;padding:12px 16px;margin:6px 0;display:flex;align-items:center;gap:12px'>
            <span style='font-size:24px'>{icon}</span>
            <div style='flex:1'>
                <b style='color:#e8f0ff'>{h['file']}</b>
                <span style='float:right;color:{status_color};font-size:12px;font-weight:700'>{h['status']}</span><br>
                <span style='color:#6a84b0;font-size:12px'>
                {h['rows']:,} rows &nbsp;·&nbsp; {h['size']} &nbsp;·&nbsp; {h['time']}
                </span>{missing_txt}
            </div>
            </div>""", unsafe_allow_html=True)

        if st.button("🗑️ Clear Upload History"):
            st.session_state.upload_history = []
            st.rerun()

        # ── Export history ──────────────────────────────────────────────────
        hist_df = pd.DataFrame(history)[['time', 'file', 'size', 'rows', 'status', 'missing']]
        hist_df['missing'] = hist_df['missing'].apply(lambda x: ', '.join(x) if x else '')
        csv_hist = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export History CSV", csv_hist, "upload_history.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
#  ▸ PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
if "EDA" in PAGE:
    st.markdown("# 🔍 Exploratory Data Analysis")
    st.markdown("<div class='info-box'><b>Purpose:</b> Understand data distribution, correlations, and churn patterns before modeling.</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "💰 Charges", "📋 Categorical", "🔥 Correlation"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(raw_df, x='tenure', color='Churn',
                               color_discrete_map={'Yes': COLORS['danger'], 'No': COLORS['good']},
                               nbins=30, barmode='overlay', opacity=0.75,
                               title='Tenure Distribution by Churn')
            fig.update_layout(**PLOTLY_LAYOUT, height=340)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("<div class='info-box'>New customers (0–12 months) churn most. Loyalty grows sharply after 24 months.</div>", unsafe_allow_html=True)

        with col2:
            fig = px.box(raw_df, x='Churn', y='MonthlyCharges', color='Churn',
                         color_discrete_map={'Yes': COLORS['danger'], 'No': COLORS['good']},
                         title='Monthly Charges vs Churn')
            fig.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("<div class='info-box'>Churners pay ~$15/month more on average. Higher prices = lower perceived value.</div>", unsafe_allow_html=True)

    with tab2:
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Monthly vs Total Charges', 'Avg Monthly Spend by Tenure Group'))
        for churn_val, color, name in [('No', COLORS['good'], 'No Churn'), ('Yes', COLORS['danger'], 'Churn')]:
            sub = raw_df[raw_df['Churn'] == churn_val]
            fig.add_trace(go.Scatter(x=sub['MonthlyCharges'], y=sub['TotalCharges'],
                                     mode='markers', name=name,
                                     marker=dict(color=color, size=3, opacity=0.4)), row=1, col=1)
        raw_df2 = raw_df.copy()
        raw_df2['AvgSpend'] = raw_df2['TotalCharges'] / (raw_df2['tenure'] + 1)
        raw_df2['TenureGroup'] = pd.cut(raw_df2['tenure'], bins=[0,12,24,48,72],
                                        labels=['0-12m','13-24m','25-48m','49-72m'], include_lowest=True)
        avg_by_group = raw_df2.groupby(['TenureGroup', 'Churn'])['AvgSpend'].mean().reset_index()
        for churn_val, color in [('No', COLORS['good']), ('Yes', COLORS['danger'])]:
            sub = avg_by_group[avg_by_group['Churn'] == churn_val]
            fig.add_trace(go.Bar(x=sub['TenureGroup'].astype(str), y=sub['AvgSpend'],
                                  name=churn_val, marker_color=color, showlegend=False), row=1, col=2)
        fig.update_layout(**PLOTLY_LAYOUT, height=380, title_text='Charge Analysis')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        cat_col = st.selectbox("Select categorical feature",
                               ['Contract', 'InternetService', 'PaymentMethod', 'gender', 'SeniorCitizen'])
        churn_rate = raw_df.groupby(cat_col)['Churn'].apply(lambda x: (x=='Yes').mean()*100).reset_index()
        churn_rate.columns = [cat_col, 'ChurnRate']
        churn_rate = churn_rate.sort_values('ChurnRate', ascending=True)
        fig = go.Figure(go.Bar(
            x=churn_rate['ChurnRate'], y=churn_rate[cat_col].astype(str),
            orientation='h', marker_color=COLORS['accent'],
            text=[f"{v:.1f}%" for v in churn_rate['ChurnRate']], textposition='outside'
        ))
        fig.add_vline(x=26.5, line_dash='dash', line_color=COLORS['warn'],
                      annotation_text='Avg 26.5%', annotation_font_color=COLORS['warn'])
        fig.update_layout(**PLOTLY_LAYOUT, height=380,
                          title=dict(text=f'Churn Rate by {cat_col}', font=dict(color=COLORS['text'])))
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
        corr = raw_df[num_cols + ['Churn']].copy()
        corr['Churn'] = (corr['Churn'] == 'Yes').astype(int)
        corr_matrix = corr.corr().round(3)
        fig = go.Figure(go.Heatmap(
            z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
            colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
            text=corr_matrix.values.round(2), texttemplate='%{text}',
            textfont=dict(color='white', size=13),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=380,
                          title=dict(text='Correlation Matrix', font=dict(color=COLORS['text'])))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ▸ PAGE: PRE-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
if "Pre-processing" in PAGE:
    st.markdown("# ⚙️ Data Pre-processing")

    steps = [
        ("1. Missing Value Imputation", "danger",
         "TotalCharges had 11 blank entries. Converted to numeric then filled with median.",
         "df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')\ndf['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)"),
        ("2. Label Encoding (Binary Columns)", "warn",
         "12 Yes/No columns encoded as 1/0 using a binary_map dict for consistency.",
         "binary_map = {'Yes':1,'No':0,'Male':1,'Female':0,'No phone service':0}\nfor col in binary_cols:\n    df[col] = df[col].map(binary_map).fillna(0).astype(int)"),
        ("3. One-Hot Encoding (Multi-Class)", "accent",
         "4 multi-category columns one-hot encoded with drop_first=True to avoid multicollinearity.",
         "df = pd.get_dummies(df, columns=['Contract','InternetService',\n                              'PaymentMethod','MultipleLines'],\n                drop_first=True)"),
        ("4. StandardScaler (Distance Models)", "good",
         "LR, SVM, and KNN are distance-sensitive. Scaler fit on train only, transform both.",
         "scaler = StandardScaler()\nX_train_sc = scaler.fit_transform(X_train)\nX_test_sc  = scaler.transform(X_test)"),
        ("5. Stratified Train-Test Split (80/20)", "accent2",
         "stratify=y preserves the 26.5% churn ratio in both train and test sets.",
         "X_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42, stratify=y)"),
    ]

    for title, color_key, why, code in steps:
        with st.expander(f"**{title}**", expanded=True):
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(f"<div class='info-box'><b>Why?</b><br>{why}</div>", unsafe_allow_html=True)
            with c2:
                st.code(code, language='python')

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Original Features",    "21")
    col2.metric("After Pre-processing", f"{len(feat_names)}")
    col3.metric("Training Samples",     f"{len(X_train):,}")


# ══════════════════════════════════════════════════════════════════════════════
#  ▸ PAGE: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
if "Feature Engineering" in PAGE:
    st.markdown("# 🧬 Feature Engineering")
    st.markdown("<div class='info-box'><b>Purpose:</b> Create new features that encode domain knowledge. A well-engineered feature can improve model performance more than hyperparameter tuning.</div>", unsafe_allow_html=True)

    feats = [
        ("AvgMonthlySpend", "TotalCharges / (tenure + 1)",
         "Normalises spend by tenure. Reveals whether billing rate is rising. +1 prevents division-by-zero.",
         "df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)",
         "0.1187", "4th", COLORS['accent']),
        ("TenureGroup", "pd.cut(tenure, bins=[0,12,24,48,72], labels=[0,1,2,3])",
         "Tenure–churn relationship is non-linear. Binning captures 4 lifecycle stages as ordinal variable.",
         "df['TenureGroup'] = pd.cut(df['tenure'],\n    bins=[0,12,24,48,72], labels=[0,1,2,3],\n    include_lowest=True).astype(float).fillna(0).astype(int)",
         "0.0318", "8th", COLORS['accent2']),
        ("ServiceCount", "Sum of 6 add-on service columns",
         "More services = higher switching cost. One integer is more parsimonious than 6 binary columns.",
         "service_cols = ['OnlineSecurity','OnlineBackup','DeviceProtection',\n                'TechSupport','StreamingTV','StreamingMovies']\ndf['ServiceCount'] = df[service_cols].sum(axis=1)",
         "0.0412", "7th", COLORS['good']),
    ]

    for name, formula, why, code, importance, rank, color in feats:
        st.markdown(f"<div class='section-header'>{name}</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            st.markdown(f"<div class='info-box'><b>Formula:</b> <code>{formula}</code><br><br>{why}</div>", unsafe_allow_html=True)
        with c2:
            st.code(code, language='python')
        with c3:
            st.metric("RF Importance", importance)
            st.metric("Rank", rank)

    st.markdown("<div class='section-header'>Top 15 Feature Importances (Random Forest)</div>", unsafe_allow_html=True)
    fi_top = fi.head(15)
    eng_feats = {'AvgMonthlySpend', 'TenureGroup', 'ServiceCount'}
    colors_fi = [COLORS['warn'] if n in eng_feats else COLORS['accent'] for n in fi_top.index]
    fig = go.Figure(go.Bar(
        x=fi_top.values, y=fi_top.index, orientation='h',
        marker_color=colors_fi,
        text=[f"{v:.4f}" for v in fi_top.values], textposition='outside'
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=480,
                      title=dict(text='🟡 Engineered features highlighted', font=dict(color=COLORS['warn'], size=13)),
                      xaxis_title='Importance Score')
    fig.update_yaxes(autorange='reversed')
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ▸ PAGE: MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
if "Model Training" in PAGE:
    st.markdown("# 🤖 Model Training")

    model_info = {
        'Logistic Regression': ("Linear baseline. Coefficients = log-odds of churn.", "✓ Interpretable  ✓ Fast  ✗ Linear boundary only", COLORS['accent']),
        'Decision Tree':       ("Rule tree: 'If tenure<12 AND Contract=Month-to-month → Churn'.", "✓ Highly interpretable  ✗ Overfits without pruning", COLORS['accent2']),
        'Random Forest':       ("150 trees via bootstrap sampling (bagging). Best overall performer.", "✓ Robust  ✓ Feature importance  ✗ Slow to explain", COLORS['good']),
        'Gradient Boosting':   ("Sequential tree building — each corrects previous errors. High precision.", "✓ High accuracy  ✓ Non-linear  ✗ Slower training", COLORS['warn']),
        'SVM':                 ("RBF kernel finds optimal hyperplane in high-dimensional space.", "✓ High dimensions  ✗ Slow on large data", COLORS['danger']),
        'KNN':                 ("Majority vote of k=7 nearest neighbours.", "✓ Simple  ✓ Non-parametric  ✗ Slow at inference", COLORS['teal']),
        'Naive Bayes':         ("Probabilistic, assumes feature independence. Fast and competitive AUC.", "✓ Very fast  ✓ Good baseline  ✗ Independence assumption", COLORS['pink']),
    }

    col1, col2 = st.columns(2)
    for i, (name, (desc, pros, color)) in enumerate(model_info.items()):
        col = col1 if i % 2 == 0 else col2
        r = results[name]
        with col:
            st.markdown(f"""
            <div class='info-box' style='border-left:3px solid {color}; margin-bottom:10px'>
            <b style='color:{color}'>{name}</b><br>
            <span style='font-size:12px;color:#6a84b0'>{pros}</span><br>
            <span style='font-size:13px'>{desc}</span><br>
            <div style='margin-top:8px;font-family:"DM Mono",monospace;font-size:13px'>
            AUC <b style='color:{color}'>{r['auc']:.3f}</b> &nbsp;·&nbsp;
            Recall <b style='color:{color}'>{r['recall']:.3f}</b> &nbsp;·&nbsp;
            F1 <b style='color:{color}'>{r['f1']:.3f}</b>
            </div></div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>5-Fold Cross-Validation AUC</div>", unsafe_allow_html=True)
    fig = go.Figure()
    for i, (name, scores) in enumerate(cv_scores.items()):
        fig.add_trace(go.Box(y=scores, name=name, marker_color=CAT7[i],
                              boxpoints='all', jitter=0.3, pointpos=-1.8))
    fig.update_layout(**PLOTLY_LAYOUT, height=360,
                      title=dict(text='Distribution of AUC across 5 folds', font=dict(color=COLORS['text'])),
                      yaxis_title='AUC-ROC')
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ▸ PAGE: EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
if "Evaluation" in PAGE:
    st.markdown("# 📊 Model Evaluation")

    sel_model = st.selectbox("Select model to inspect", list(results.keys()), index=2)
    r = results[sel_model]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{r['accuracy']:.3f}")
    c2.metric("Precision", f"{r['precision']:.3f}")
    c3.metric("Recall ⭐", f"{r['recall']:.3f}")
    c4.metric("F1-Score",  f"{r['f1']:.3f}")
    c5.metric("AUC-ROC",   f"{r['auc']:.3f}")

    col_a, col_b = st.columns(2)

    with col_a:
        cm = r['cm']
        labels_text = [[f"TN\n{cm[0][0]}", f"FP\n{cm[0][1]}"],
                       [f"FN\n{cm[1][0]}", f"TP\n{cm[1][1]}"]]
        quad_colors = [COLORS['good'], COLORS['warn'], COLORS['danger'], COLORS['accent']]
        fig = go.Figure(go.Heatmap(
            z=[[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]],
            text=labels_text, texttemplate='%{text}',
            colorscale=[[0,'#0d1628'],[1,'#0d1628']], showscale=False, xgap=4, ygap=4,
        ))
        for row_i in [0, 1]:
            for col_i in [0, 1]:
                fig.add_shape(type='rect', x0=col_i-0.5, x1=col_i+0.5,
                              y0=row_i-0.5, y1=row_i+0.5,
                              fillcolor=quad_colors[row_i*2+col_i], opacity=0.25, line_width=0)
        layout = dict(PLOTLY_LAYOUT)
        layout.update({
            'height': 320,
            'title': dict(text=f'Confusion Matrix — {sel_model}', font=dict(color=COLORS['text'])),
            'xaxis': dict(PLOTLY_LAYOUT.get('xaxis', {}), ticktext=['Pred No', 'Pred Yes'], tickvals=[0, 1]),
            'yaxis': dict(PLOTLY_LAYOUT.get('yaxis', {}), ticktext=['Actual No', 'Actual Yes'], tickvals=[0, 1], autorange='reversed'),
        })
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = go.Figure()
        for i, (name, (fpr, tpr)) in enumerate(roc_data.items()):
            width = 3 if name == sel_model else 1
            opacity = 1 if name == sel_model else 0.45
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} ({results[name]['auc']:.3f})",
                                      mode='lines', line=dict(color=CAT7[i], width=width), opacity=opacity))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                  line=dict(color='gray', dash='dash', width=1), showlegend=False))
        fig.update_layout(**PLOTLY_LAYOUT, height=320,
                          title=dict(text='ROC Curves — All Models', font=dict(color=COLORS['text'])),
                          xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                          legend=dict(**LEGEND, font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'>Full Comparison Table</div>", unsafe_allow_html=True)
    df_metrics = pd.DataFrame({
        'Model':     list(results.keys()),
        'Accuracy':  [f"{r['accuracy']:.3f}"  for r in results.values()],
        'Precision': [f"{r['precision']:.3f}" for r in results.values()],
        'Recall ⭐': [f"{r['recall']:.3f}"    for r in results.values()],
        'F1':        [f"{r['f1']:.3f}"        for r in results.values()],
        'AUC-ROC':   [f"{r['auc']:.3f}"       for r in results.values()],
    })
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ▸ PAGE: LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
if "Live Predictor" in PAGE:
    st.markdown("# 🎯 Live Single-Customer Predictor")
    st.markdown("<div class='info-box'>Fill in customer details and click <b>Predict</b> to get an instant churn probability from the trained Random Forest model.</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='sub-header'>Demographics & Account</div>", unsafe_allow_html=True)
        gender     = st.selectbox("Gender", ["Male", "Female"])
        senior     = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner    = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure     = st.slider("Tenure (months)", 0, 72, 12)
        contract   = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    with col2:
        st.markdown("<div class='sub-header'>Services</div>", unsafe_allow_html=True)
        phone_svc    = st.selectbox("Phone Service", ["Yes", "No"])
        multi_lines  = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_svc = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_sec   = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_bkp   = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        dev_prot     = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

    with col3:
        st.markdown("<div class='sub-header'>Billing</div>", unsafe_allow_html=True)
        tech_sup     = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        stream_tv    = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        stream_movie = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        paperless    = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment      = st.selectbox("Payment Method", ["Electronic check", "Mailed check",
                                                        "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_chg  = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
        total_chg    = st.slider("Total Charges ($)", 0.0, 9000.0, monthly_chg * tenure, step=10.0)

    if st.button("🔮 Predict Churn Probability", use_container_width=True):
        def yn(v): return 1 if v == "Yes" else 0

        row = {n: 0 for n in feat_names}
        row['gender']          = 1 if gender == "Male" else 0
        row['SeniorCitizen']   = senior
        row['Partner']         = yn(partner)
        row['Dependents']      = yn(dependents)
        row['tenure']          = tenure
        row['PhoneService']    = yn(phone_svc)
        row['PaperlessBilling']= yn(paperless)
        row['MonthlyCharges']  = monthly_chg
        row['TotalCharges']    = total_chg
        row['OnlineSecurity']  = yn(online_sec)
        row['OnlineBackup']    = yn(online_bkp)
        row['DeviceProtection']= yn(dev_prot)
        row['TechSupport']     = yn(tech_sup)
        row['StreamingTV']     = yn(stream_tv)
        row['StreamingMovies'] = yn(stream_movie)

        if multi_lines == "Yes":                    row['MultipleLines_Yes'] = 1
        elif multi_lines == "No phone service":     row['MultipleLines_No phone service'] = 1
        if internet_svc == "Fiber optic":           row['InternetService_Fiber optic'] = 1
        elif internet_svc == "No":                  row['InternetService_No'] = 1
        if contract == "One year":                  row['Contract_One year'] = 1
        elif contract == "Two year":                row['Contract_Two year'] = 1
        if payment == "Credit card (automatic)":    row['PaymentMethod_Credit card (automatic)'] = 1
        elif payment == "Electronic check":         row['PaymentMethod_Electronic check'] = 1
        elif payment == "Mailed check":             row['PaymentMethod_Mailed check'] = 1

        row['AvgMonthlySpend'] = total_chg / (tenure + 1)
        row['TenureGroup']     = int(pd.cut([tenure], bins=[0,12,24,48,72], labels=[0,1,2,3], include_lowest=True)[0] or 0)
        svc_count = sum([yn(online_sec), yn(online_bkp), yn(dev_prot),
                         yn(tech_sup), yn(stream_tv), yn(stream_movie)])
        row['ServiceCount'] = svc_count

        prob, pred = predict_customer(row)
        pct = prob * 100

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if pct >= 60:
                badge = "<span class='risk-high'>🔴 HIGH RISK</span>"
                msg   = "Immediate retention action recommended"
            elif pct >= 35:
                badge = "<span class='risk-medium'>🟡 MEDIUM RISK</span>"
                msg   = "Monitor and offer a loyalty incentive"
            else:
                badge = "<span class='risk-low'>🟢 LOW RISK</span>"
                msg   = "Customer is likely to stay"

            st.markdown(f"<div style='text-align:center;margin:20px 0'>{badge}</div>", unsafe_allow_html=True)
            st.metric("Churn Probability", f"{pct:.1f}%")
            st.markdown(f"<div class='info-box'>{msg}</div>", unsafe_allow_html=True)

        with res_col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=pct,
                number={'suffix': '%', 'font': {'size': 42, 'color': COLORS['text']}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': COLORS['text2']},
                    'bar': {'color': COLORS['danger'] if pct >= 60 else COLORS['warn'] if pct >= 35 else COLORS['good']},
                    'steps': [{'range': [0, 35], 'color': '#0d2d1f'},
                               {'range': [35, 60], 'color': '#2a1f0a'},
                               {'range': [60, 100], 'color': '#2a0d14'}],
                    'threshold': {'line': {'color': 'white', 'width': 2}, 'thickness': 0.75, 'value': 50}
                }
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=280)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ▸ PAGE: BATCH PREDICTION  (with validation)
# ══════════════════════════════════════════════════════════════════════════════
if "Batch" in PAGE:
    st.markdown("# 📦 Batch Prediction")
    st.markdown("""
    <div class='info-box'>
    Upload a CSV with the same 21 columns as the Telco dataset. The app validates column names
    before scoring, shows exact missing columns if invalid, and lets you download results.<br>
    <b>Tip:</b> Go to <b>📋 Template</b> page to download a sample CSV with the correct format.
    </div>
    """, unsafe_allow_html=True)

    # Quick template link
    st.info("📋 Need the format? Go to **📋 Template** in the sidebar to download a sample CSV.")

    batch_file = st.file_uploader("Upload customer CSV for batch scoring", type="csv",
                                   key="batch_uploader")

    if batch_file:
        try:
            batch_df = pd.read_csv(batch_file)
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            st.stop()

        size_kb = batch_file.size / 1024
        is_valid, missing, extra = validate_csv_columns(batch_df)

        if not is_valid:
            # Show validation error with exact missing columns
            missing_html = ''.join([f'<code style="background:#1e2d4a;padding:2px 6px;border-radius:4px;margin:2px;display:inline-block">{c}</code>' for c in missing])
            st.markdown(f"""
            <div class='val-fail'>
            ❌ <b>Validation Failed</b> — {len(missing)} column(s) missing from your file:<br><br>
            {missing_html}<br><br>
            Please fix your CSV and re-upload. Go to <b>📋 Template</b> to see the full required format.
            </div>""", unsafe_allow_html=True)

            add_upload_record(batch_file.name, size_kb, len(batch_df), False, missing)

            if extra:
                st.markdown(f"""
                <div class='val-warn'>
                ⚠️ These columns in your file are <b>not</b> in the required schema: {', '.join(f'<code>{c}</code>' for c in extra)}
                </div>""", unsafe_allow_html=True)

        else:
            if extra:
                st.markdown(f"""
                <div class='val-warn'>
                ⚠️ Extra columns found (ignored during scoring): {', '.join(f'<code>{c}</code>' for c in extra)}
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class='val-ok'>
            ✅ <b>Valid CSV</b> · {len(batch_df):,} rows · {len(batch_df.columns)} columns · {size_kb:.1f} KB
            </div>""", unsafe_allow_html=True)

            add_upload_record(batch_file.name, size_kb, len(batch_df), True)

            st.markdown(f"**Loaded {len(batch_df):,} rows · {len(batch_df.columns)} columns**")

            if st.button("▶ Score All Customers"):
                with st.spinner("Scoring…"):
                    tmp = batch_df.copy()
                    tmp.drop(columns=['customerID'], inplace=True, errors='ignore')
                    tmp['TotalCharges'] = pd.to_numeric(tmp['TotalCharges'], errors='coerce').fillna(0)
                    if 'Churn' in tmp.columns:
                        tmp['Churn'] = (tmp['Churn'] == 'Yes').astype(int)
                    le = LabelEncoder()
                    binary_cols = ['gender','Partner','Dependents','PhoneService','PaperlessBilling',
                                   'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
                                   'StreamingTV','StreamingMovies']
                    for col in binary_cols:
                        if col in tmp.columns:
                            tmp[col] = le.fit_transform(tmp[col].astype(str))
                    ohe_cols = [c for c in ['MultipleLines','InternetService','Contract','PaymentMethod'] if c in tmp.columns]
                    tmp = pd.get_dummies(tmp, columns=ohe_cols, drop_first=True)
                    bool_cols2 = tmp.select_dtypes(include='bool').columns
                    tmp[bool_cols2] = tmp[bool_cols2].astype(int)
                    tmp['AvgMonthlySpend'] = tmp['TotalCharges'] / (tmp['tenure'] + 1)
                    tmp['TenureGroup'] = pd.cut(tmp['tenure'], bins=[0,12,24,48,72], labels=[0,1,2,3], include_lowest=True).astype(float).fillna(0).astype(int)
                    svc_ex = [c for c in ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies'] if c in tmp.columns]
                    tmp['ServiceCount'] = tmp[svc_ex].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)

                    X_batch = tmp.reindex(columns=feat_names, fill_value=0).astype(np.float64)
                    rf = results['Random Forest']['model']
                    probs = rf.predict_proba(X_batch)[:, 1]

                    out = batch_df.copy()
                    out['ChurnProbability'] = (probs * 100).round(1)
                    out['ChurnPrediction']  = ['Yes' if p >= 50 else 'No' for p in probs]
                    out['RiskSegment']      = ['High' if p >= 60 else 'Medium' if p >= 35 else 'Low' for p in probs]

                    st.success(f"✓ Scored {len(out):,} customers")

                    risk_counts = out['RiskSegment'].value_counts()
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("🔴 High Risk",    f"{risk_counts.get('High',0):,}")
                    c2.metric("🟡 Medium Risk",  f"{risk_counts.get('Medium',0):,}")
                    c3.metric("🟢 Low Risk",     f"{risk_counts.get('Low',0):,}")
                    c4.metric("Avg Churn Prob",  f"{probs.mean()*100:.1f}%")

                    # Histogram of probabilities
                    fig = go.Figure(go.Histogram(
                        x=probs*100, nbinsx=20,
                        marker_color=COLORS['accent'], opacity=0.8,
                        name='Churn Probability Distribution'
                    ))
                    fig.update_layout(**PLOTLY_LAYOUT, height=280,
                                      title=dict(text='Distribution of Churn Probabilities', font=dict(color=COLORS['text'])),
                                      xaxis_title='Churn Probability (%)', yaxis_title='Count')
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(out.head(50), use_container_width=True)

                    csv_out = out.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Full Results CSV", csv_out,
                                       "churn_predictions.csv", "text/csv")
    else:
        st.markdown("""
        <div class='info-box' style='text-align:center;padding:24px'>
        <div style='font-size:36px;margin-bottom:8px'>📦</div>
        Upload a CSV file above to start batch scoring.<br>
        The file will be validated before scoring begins.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ▸ PAGE: ROI CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
if "ROI" in PAGE:
    st.markdown("# 💰 Business ROI Calculator")
    st.markdown("<div class='info-box'>Translate model performance into business value. Adjust the inputs to see expected revenue saved per churn cycle.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='sub-header'>Customer & Revenue Inputs</div>", unsafe_allow_html=True)
        total_customers = st.number_input("Total active customers", value=7043, step=100)
        churn_rate      = st.slider("Current monthly churn rate (%)", 1.0, 50.0, 26.5, step=0.5)
        monthly_revenue = st.number_input("Avg monthly revenue per customer ($)", value=65.0, step=1.0)
        cac             = st.number_input("Customer Acquisition Cost ($)", value=300.0, step=10.0)
        retention_cost  = st.number_input("Retention campaign cost per customer ($)", value=15.0, step=1.0)
        model_recall    = st.slider("Model Recall (% churners caught)", 50, 100, 76, step=1)
        model_precision = st.slider("Model Precision (% alerts are real)", 40, 100, 65, step=1)

    with c2:
        st.markdown("<div class='sub-header'>Results</div>", unsafe_allow_html=True)
        n_churners      = int(total_customers * churn_rate / 100)
        churners_caught = int(n_churners * model_recall / 100)
        false_alerts    = int(churners_caught / (model_precision/100) * (1 - model_precision/100))
        total_alerted   = churners_caught + false_alerts
        retention_spend = total_alerted * retention_cost
        revenue_saved   = churners_caught * monthly_revenue * 12
        cac_saved       = churners_caught * cac
        net_benefit     = revenue_saved + cac_saved - retention_spend

        st.metric("Expected churners / cycle",  f"{n_churners:,}")
        st.metric("Churners caught by model",   f"{churners_caught:,}")
        st.metric("Total alerts sent",           f"{total_alerted:,}")
        st.metric("Retention campaign cost",     f"${retention_spend:,.0f}")
        st.metric("Annual revenue saved",        f"${revenue_saved:,.0f}", delta="churners retained × 12-month LTV")
        st.metric("CAC equivalent saved",        f"${cac_saved:,.0f}", delta="avoided re-acquisition")
        st.metric("🏆 Net Benefit / Cycle",      f"${net_benefit:,.0f}",
                  delta="positive ROI" if net_benefit > 0 else "negative ROI")

    st.markdown("<div class='section-header'>Sensitivity: Net Benefit vs Model Recall</div>", unsafe_allow_html=True)
    recalls  = np.arange(0.4, 1.01, 0.05)
    benefits = []
    for rec in recalls:
        caught  = int(n_churners * rec)
        alerts  = int(caught / (model_precision/100))
        benefit = caught * monthly_revenue * 12 + caught * cac - alerts * retention_cost
        benefits.append(benefit)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recalls*100, y=benefits, mode='lines+markers',
        line=dict(color=COLORS['accent'], width=2),
        marker=dict(color=COLORS['accent'], size=6),
        fill='tozeroy', fillcolor='rgba(79,142,247,0.12)',
        name='Net Benefit'
    ))
    fig.add_vline(x=model_recall, line_dash='dash', line_color=COLORS['warn'],
                  annotation_text=f"Current model ({model_recall}% recall)",
                  annotation_font_color=COLORS['warn'])
    fig.update_layout(**PLOTLY_LAYOUT, height=360,
                      xaxis_title='Model Recall (%)', yaxis_title='Net Benefit ($)',
                      yaxis_tickprefix='$', yaxis_tickformat=',')
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Select a page from the sidebar →")
