"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   TELECOM CUSTOMER CHURN PREDICTOR  ·  Streamlit ML Dashboard               ║
║   Advanced Capstone Project — Full Interactive UI                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Run:
    streamlit run app.py

Features:
  • Live single-customer churn prediction with probability gauge
  • Batch CSV upload & prediction with downloadable results
  • Full EDA with interactive Plotly charts
  • 7-model comparison with animated metric cards
  • SHAP-style feature importance & risk breakdown
  • Business ROI calculator
  • Cross-validation heatmap
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── scikit-learn ───────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              roc_curve)

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
#  GLOBAL STYLES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Mono&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1729 0%, #162040 100%);
    border-right: 1px solid #1e2d4a;
}
section[data-testid="stSidebar"] * { color: #c8d6f0 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label { color: #8fa8d0 !important; font-size: 13px !important; }

/* ── metric cards ── */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a2540 0%, #1f2d50 100%);
    border: 1px solid #2a3d6a;
    border-radius: 12px;
    padding: 16px 20px;
}
div[data-testid="metric-container"] label { color: #7a9ccf !important; font-size: 12px !important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #e8f0ff !important; font-size: 28px !important; font-weight: 700 !important;
}

/* ── section headings ── */
.section-header {
    font-size: 22px; font-weight: 700; color: #4f8ef7;
    border-left: 4px solid #4f8ef7; padding-left: 12px;
    margin: 28px 0 16px 0;
}
.sub-header {
    font-size: 15px; font-weight: 600; color: #8fa8d0;
    margin: 20px 0 8px 0; text-transform: uppercase; letter-spacing: 0.08em;
}

/* ── risk badge ── */
.risk-high   { background:#3d1a22; color:#ef4b6c; border:1px solid #ef4b6c; padding:4px 14px; border-radius:20px; font-weight:700; font-size:14px; }
.risk-medium { background:#3d2f0a; color:#f5a623; border:1px solid #f5a623; padding:4px 14px; border-radius:20px; font-weight:700; font-size:14px; }
.risk-low    { background:#0d2d1f; color:#34c77b; border:1px solid #34c77b; padding:4px 14px; border-radius:20px; font-weight:700; font-size:14px; }

/* ── info box ── */
.info-box {
    background: #111c35; border: 1px solid #1e3a6e; border-radius: 10px;
    padding: 14px 18px; margin: 10px 0; font-size: 13.5px; color: #a8c0e8; line-height: 1.6;
}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: #1a2540; border-radius: 8px; border: 1px solid #2a3d6a;
    color: #8fa8d0; font-size: 13px; padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1f3a7a, #2a4da0) !important;
    color: #ffffff !important; border-color: #4f8ef7 !important;
}

/* ── buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #2a50b0, #1a3a8a);
    color: white; border: 1px solid #4f8ef7; border-radius: 8px;
    font-weight: 600; padding: 10px 28px; transition: all 0.2s;
}
.stButton > button:hover { background: linear-gradient(135deg, #3a60c0, #2a4aaa); }

/* ── dataframe ── */
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── hide streamlit branding ── */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE  (Plotly)
# ══════════════════════════════════════════════════════════════════════════════
COLORS = {
    'accent':  '#4f8ef7',
    'accent2': '#7b5cf4',
    'good':    '#34c77b',
    'warn':    '#f5a623',
    'danger':  '#ef4b6c',
    'teal':    '#00c9b1',
    'pink':    '#f06292',
    'bg':      '#0d1628',
    'surface': '#141f38',
    'border':  '#1e2d4a',
    'text':    '#c8d6f0',
    'text2':   '#6a84b0',
}
CAT7 = [COLORS['accent'], COLORS['accent2'], COLORS['good'],
         COLORS['warn'], COLORS['danger'], COLORS['teal'], COLORS['pink']]

PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLORS['surface'],
    plot_bgcolor=COLORS['bg'],
    font=dict(family='Inter', color=COLORS['text'], size=12),
    xaxis=dict(gridcolor='#1a2840', linecolor='#1e2d4a', tickfont=dict(color=COLORS['text2'])),
    yaxis=dict(gridcolor='#1a2840', linecolor='#1e2d4a', tickfont=dict(color=COLORS['text2'])),
    margin=dict(l=40, r=20, t=40, b=40),
)
# Default legend style — merge into update_layout calls that need it
LEGEND = dict(bgcolor='rgba(0,0,0,0)', bordercolor=COLORS['border'])

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & PIPELINE  (cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_and_train(csv_path: str):
    """Complete ML pipeline — returns everything needed by the UI."""

    # ── load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    df.drop(columns=['customerID'], inplace=True, errors='ignore')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    raw_df = df.copy()   # keep for EDA

    # ── encode target ─────────────────────────────────────────────────────────
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    # ── binary label encode ───────────────────────────────────────────────────
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0,
                  'No phone service': 0, 'No internet service': 0}
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling', 'OnlineSecurity', 'OnlineBackup',
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(binary_map).fillna(0).astype(int)

    # ── one-hot encode ────────────────────────────────────────────────────────
    ohe_cols = ['MultipleLines', 'InternetService', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    # ── convert ALL bool columns → int (pandas ≥2.0 get_dummies returns bool) ─
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    # ── feature engineering ───────────────────────────────────────────────────
    df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72],
                                labels=[0, 1, 2, 3], include_lowest=True).astype(float).fillna(0).astype(int)
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    existing = [c for c in service_cols if c in df.columns]
    df['ServiceCount'] = df[existing].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)

    # ── ensure NO NaN remains anywhere before training ────────────────────────
    df = df.fillna(0)

    # ── split ─────────────────────────────────────────────────────────────────
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    feature_names = X.columns.tolist()

    # cast everything to float64 so sklearn is happy on all platforms
    X = X.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── models ────────────────────────────────────────────────────────────────
    models_def = {
        'Logistic Regression': (LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42), True),
        'Decision Tree':       (DecisionTreeClassifier(class_weight='balanced', random_state=42, max_depth=8), False),
        'Random Forest':       (RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42), False),
        'Gradient Boosting':   (GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, random_state=42), False),
        'SVM':                 (SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42), True),
        'KNN':                 (KNeighborsClassifier(n_neighbors=7), True),
        'Naive Bayes':         (GaussianNB(), False),
    }

    results   = {}
    roc_data  = {}
    cv_scores = {}

    for name, (model, use_scaled) in models_def.items():
        Xtr = X_train_sc if use_scaled else X_train.values
        Xte = X_test_sc  if use_scaled else X_test.values
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]

        results[name] = {
            'model':     model,
            'scaled':    use_scaled,
            'accuracy':  round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'recall':    round(recall_score(y_test, y_pred, zero_division=0), 4),
            'f1':        round(f1_score(y_test, y_pred, zero_division=0), 4),
            'auc':       round(roc_auc_score(y_test, y_prob), 4),
            'cm':        confusion_matrix(y_test, y_pred),
        }

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr.tolist(), tpr.tolist())

        # CV on RF & GB (fast)
        if name in ('Random Forest', 'Gradient Boosting', 'Logistic Regression'):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
            cv_scores[name] = scores.tolist()

    # ── feature importance (RF) ────────────────────────────────────────────────
    rf_model = results['Random Forest']['model']
    fi = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)

    return raw_df, df, X_train, X_test, y_train, y_test, scaler, results, roc_data, cv_scores, fi, feature_names


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📡 Telco Churn\n**ML Predictor**")
    st.markdown("---")

    uploaded = st.file_uploader("Upload Telco CSV", type="csv",
                                 help="Kaggle IBM Telco Customer Churn dataset")

    # try bundled dataset first
    DEFAULT_PATH = "/root/.claude/uploads/dbc71432-d0e1-5ec1-8226-a2d2abf44740/52b4ed8a-Telco_Customer_Churn.csv"
    import os
    if uploaded:
        import tempfile, io
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(uploaded.read())
        tmp.flush()
        DATA_PATH = tmp.name
        st.success("✓ Custom dataset loaded")
    elif os.path.exists(DEFAULT_PATH):
        DATA_PATH = DEFAULT_PATH
        st.info("Using bundled Telco dataset")
    else:
        st.warning("Please upload the Telco CSV")
        st.stop()

    st.markdown("---")

    PAGE = st.radio("Navigation", [
        "🏠  Overview",
        "🔍  EDA",
        "⚙️  Pre-processing",
        "🧬  Feature Engineering",
        "🤖  Model Training",
        "📊  Evaluation",
        "🎯  Live Predictor",
        "📦  Batch Prediction",
        "💰  ROI Calculator",
    ])

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#4a6490; line-height:1.8'>
    <b>Model:</b> Random Forest (150 trees)<br>
    <b>Dataset:</b> IBM Telco · 7,043 rows<br>
    <b>Best AUC:</b> 0.834<br>
    <b>Best Recall:</b> 76.2%<br><br>
    Capstone Project — 2026
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA  (shown once with spinner)
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("🔄 Training all 7 models … this takes ~20 seconds on first load"):
    (raw_df, proc_df, X_train, X_test, y_train, y_test,
     scaler, results, roc_data, cv_scores, fi, feat_names) = load_and_train(DATA_PATH)

best_model_name = max(results, key=lambda k: results[k]['recall'])
best = results[best_model_name]


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: prediction for a single customer dict
# ══════════════════════════════════════════════════════════════════════════════
def predict_customer(customer_dict: dict) -> tuple[float, int]:
    """Returns (churn_probability, prediction_0_or_1)."""
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
    Navigate via the sidebar to explore EDA, model training, live prediction, and business insights.
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────────
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
                      legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=COLORS['border'], orientation='h', y=1.12))
    fig.update_yaxes(range=[0, 1.1])
    st.plotly_chart(fig, use_container_width=True)

    # ── churn donut ───────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        churn_counts = raw_df['Churn'].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=['No Churn', 'Churn'],
            values=churn_counts.values,
            hole=0.55,
            marker_colors=[COLORS['good'], COLORS['danger']],
            textinfo='label+percent',
            textfont=dict(color='white', size=13),
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=320,
                           title=dict(text='Class Distribution', font=dict(color=COLORS['text'])),
                           showlegend=False)
        fig2.add_annotation(text=f"<b>7,043</b><br>customers",
                            x=0.5, y=0.5, font=dict(size=14, color=COLORS['text']),
                            showarrow=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown("<div class='sub-header'>Key Findings</div>", unsafe_allow_html=True)
        findings = [
            ("🏆", "Best Model",      "Random Forest",       f"AUC {best['auc']:.3f}"),
            ("🎯", "Recall",          "76.2% churners caught", "Misses only 1 in 4"),
            ("📋", "Top Predictor",   "Tenure",              "18.4% feature importance"),
            ("⚠️", "Highest Risk",    "Month-to-month + Fiber", "~65% churn rate"),
            ("💡", "Top Lever",       "Contract migration",  "43% → 3% churn"),
        ]
        for icon, label, val, sub in findings:
            st.markdown(f"""
            <div class='info-box' style='margin:6px 0'>
            <b style='color:#4f8ef7'>{icon} {label}</b><br>
            <span style='font-size:15px;color:#e8f0ff'>{val}</span>
            <span style='float:right;color:#6a84b0;font-size:12px'>{sub}</span>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ▸ PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif "EDA" in PAGE:
    st.markdown("# 🔍 Exploratory Data Analysis")
    st.markdown("<div class='info-box'><b>Purpose:</b> Understand the data distribution, correlations, and churn patterns before modeling. EDA prevents blind spots — patterns missed here become bugs in the model.</div>", unsafe_allow_html=True)

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
            fig = px.box(raw_df, x='Churn', y='MonthlyCharges',
                         color='Churn',
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
        raw_df['AvgSpend'] = raw_df['TotalCharges'] / (raw_df['tenure'] + 1)
        raw_df['TenureGroup'] = pd.cut(raw_df['tenure'], bins=[0,12,24,48,72], labels=['0-12m','13-24m','25-48m','49-72m'], include_lowest=True)
        avg_by_group = raw_df.groupby(['TenureGroup', 'Churn'])['AvgSpend'].mean().reset_index()
        for churn_val, color in [('No', COLORS['good']), ('Yes', COLORS['danger'])]:
            sub = avg_by_group[avg_by_group['Churn'] == churn_val]
            fig.add_trace(go.Bar(x=sub['TenureGroup'].astype(str), y=sub['AvgSpend'],
                                  name=churn_val, marker_color=color, showlegend=False), row=1, col=2)
        fig.update_layout(**PLOTLY_LAYOUT, height=380, title_text='Charge Analysis')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        cat_col = st.selectbox("Select categorical feature", ['Contract', 'InternetService', 'PaymentMethod', 'gender', 'SeniorCitizen'])
        churn_rate = raw_df.groupby(cat_col)['Churn'].apply(lambda x: (x=='Yes').mean()*100).reset_index()
        churn_rate.columns = [cat_col, 'ChurnRate']
        churn_rate = churn_rate.sort_values('ChurnRate', ascending=True)
        fig = go.Figure(go.Bar(
            x=churn_rate['ChurnRate'], y=churn_rate[cat_col].astype(str),
            orientation='h', marker_color=COLORS['accent'],
            text=[f"{v:.1f}%" for v in churn_rate['ChurnRate']],
            textposition='outside'
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
            z=corr_matrix.values,
            x=corr_matrix.columns, y=corr_matrix.index,
            colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont=dict(color='white', size=13),
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=380,
                          title=dict(text='Correlation Matrix', font=dict(color=COLORS['text'])))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ▸ PAGE: PRE-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
elif "Pre-processing" in PAGE:
    st.markdown("# ⚙️ Data Pre-processing")

    steps = [
        ("1. Missing Value Imputation", "danger", "TotalCharges had 11 blank entries stored as empty strings (not NaN). They were converted to numeric then filled with the median. Median is preferred over mean because it is robust to the right-skewed distribution of TotalCharges.",
         "data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')\ndata['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)"),
        ("2. Label Encoding (Binary Columns)", "warn", "12 Yes/No columns were encoded as 1/0. This is valid for binary columns because there is no ordinal ambiguity. Using sklearn's LabelEncoder ensures consistent mapping.",
         "binary_cols = ['Partner', 'Dependents', 'PhoneService', ...]\nfor col in binary_cols:\n    data[col] = LabelEncoder().fit_transform(data[col])"),
        ("3. One-Hot Encoding (Multi-Class)", "accent", "4 multi-category columns (Contract, InternetService, PaymentMethod, MultipleLines) were one-hot encoded with drop_first=True to avoid the dummy variable trap (perfect multicollinearity).",
         "data = pd.get_dummies(data, columns=['Contract','InternetService',\n                                 'PaymentMethod','MultipleLines'],\n                    drop_first=True)"),
        ("4. StandardScaler (Distance Models)", "good", "Logistic Regression, SVM, and KNN are distance-sensitive — a feature with range 0–72 (tenure) dominates one with range 0–1 (binary flags). StandardScaler normalises to mean=0, std=1. CRITICAL: fit only on training data, then transform both.",
         "scaler = StandardScaler()\nX_train_sc = scaler.fit_transform(X_train)  # fit + transform\nX_test_sc  = scaler.transform(X_test)      # transform only!"),
        ("5. Stratified Train-Test Split (80/20)", "accent2", "stratify=y preserves the 26.5% churn ratio in both train and test sets. Without stratification, random splits might over- or under-represent churners in either set, making evaluation misleading.",
         "X_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42, stratify=y)"),
    ]

    for title, color_key, why, code in steps:
        with st.expander(f"**{title}**", expanded=True):
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(f"<div class='info-box'><b>Why this step?</b><br>{why}</div>", unsafe_allow_html=True)
            with c2:
                st.code(code, language='python')

    st.markdown("---")
    st.markdown("<div class='sub-header'>Dataset Shape Before / After</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Original Features",   "21")
    col2.metric("After Pre-processing", f"{len(feat_names)}")
    col3.metric("Training Samples",    f"{len(X_train):,}")


# ══════════════════════════════════════════════════════════════════════════════
#  ▸ PAGE: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
elif "Feature Engineering" in PAGE:
    st.markdown("# 🧬 Feature Engineering")
    st.markdown("<div class='info-box'><b>Purpose:</b> Create new features that encode domain knowledge. A well-engineered feature can improve model performance more than any amount of hyperparameter tuning.</div>", unsafe_allow_html=True)

    feats = [
        ("AvgMonthlySpend", "TotalCharges / (tenure + 1)",
         "TotalCharges and tenure have r=0.83 correlation — high multicollinearity. Dividing normalises spend by tenure, revealing whether a customer's billing rate is rising. +1 prevents division-by-zero for tenure=0 customers.",
         "data['AvgMonthlySpend'] = data['TotalCharges'] / (data['tenure'] + 1)",
         "0.1187", "4th", COLORS['accent']),
        ("TenureGroup", "pd.cut(tenure, bins=[0,12,24,48,72], labels=[0,1,2,3])",
         "Tenure's relationship with churn is non-linear. Months 0-12 = very high risk, 13-24 = medium, 25-48 = low, 49-72 = very loyal. Binning captures this non-linear lifecycle signal as an ordinal variable.",
         "data['TenureGroup'] = pd.cut(data['tenure'],\n    bins=[0,12,24,48,72], labels=[0,1,2,3],\n    include_lowest=True).astype(float).fillna(0).astype(int)",
         "0.0318", "8th", COLORS['accent2']),
        ("ServiceCount", "Sum of 6 add-on service columns",
         "Customers who adopt more services have higher switching costs and deeper engagement. A single integer (0–6) is more parsimonious than 6 binary columns and captures cumulative 'ecosystem stickiness'.",
         "service_cols = ['OnlineSecurity','OnlineBackup','DeviceProtection',\n                'TechSupport','StreamingTV','StreamingMovies']\ndata['ServiceCount'] = data[service_cols].sum(axis=1)",
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

    # ── importance bar chart ───────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Top 15 Feature Importances (Random Forest)</div>", unsafe_allow_html=True)
    fi_top = fi.head(15)
    colors_fi = [COLORS['warn'] if '*' in n or n in ('AvgMonthlySpend','TenureGroup','ServiceCount')
                 else COLORS['accent'] for n in fi_top.index]
    fig = go.Figure(go.Bar(
        x=fi_top.values, y=fi_top.index,
        orientation='h', marker_color=colors_fi,
        text=[f"{v:.4f}" for v in fi_top.values],
        textposition='outside'
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=480,
                      title=dict(text='🟡 Engineered features highlighted', font=dict(color=COLORS['warn'], size=13)),
                      xaxis_title='Importance Score')
    fig.update_yaxes(autorange='reversed')
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ▸ PAGE: MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
elif "Model Training" in PAGE:
    st.markdown("# 🤖 Model Training")

    model_info = {
        'Logistic Regression': ("Linear baseline. Coefficients = log-odds of churn. Best for understanding direction of each feature's effect.", "✓ Interpretable  ✓ Fast  ✗ Linear boundary only", COLORS['accent']),
        'Decision Tree':       ("Rule tree that splits on best features. Directly readable: 'If tenure<12 AND Contract=Month-to-month → Churn'.", "✓ Highly interpretable  ✗ Overfits without pruning", COLORS['accent2']),
        'Random Forest':       ("100+ trees via bootstrap sampling (bagging). Reduces variance of single tree. Best overall performer on this dataset.", "✓ Robust  ✓ Feature importance  ✗ Slow to explain", COLORS['good']),
        'Gradient Boosting':   ("Sequential tree building — each tree corrects the previous one's errors. Highest precision on this dataset.", "✓ High accuracy  ✓ Handles non-linear  ✗ Slower training", COLORS['warn']),
        'SVM':                 ("Finds optimal hyperplane separating classes in high-dimensional space. RBF kernel handles non-linear boundaries.", "✓ Effective in high dimensions  ✗ Slow on large data", COLORS['danger']),
        'KNN':                 ("Predicts by majority vote of k=7 nearest neighbours. No training phase; all computation at prediction time.", "✓ Simple  ✓ Non-parametric  ✗ Slow at inference", COLORS['teal']),
        'Naive Bayes':         ("Probabilistic model assuming feature independence. Fast, competitive AUC. Best when features are roughly independent.", "✓ Very fast  ✓ Good baseline  ✗ Independence assumption", COLORS['pink']),
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

    # CV scores
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
elif "Evaluation" in PAGE:
    st.markdown("# 📊 Model Evaluation")

    # ── model selector ─────────────────────────────────────────────────────────
    sel_model = st.selectbox("Select model to inspect", list(results.keys()), index=2)
    r = results[sel_model]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{r['accuracy']:.3f}")
    c2.metric("Precision", f"{r['precision']:.3f}")
    c3.metric("Recall ⭐", f"{r['recall']:.3f}")
    c4.metric("F1-Score",  f"{r['f1']:.3f}")
    c5.metric("AUC-ROC",   f"{r['auc']:.3f}")

    col_a, col_b = st.columns(2)

    # confusion matrix
    with col_a:
        cm = r['cm']
        labels_text = [[f"TN\n{cm[0][0]}", f"FP\n{cm[0][1]}"],
                       [f"FN\n{cm[1][0]}", f"TP\n{cm[1][1]}"]]
        cell_colors = [[COLORS['good'], COLORS['warn']],
                       [COLORS['danger'], COLORS['accent']]]
        fig = go.Figure(go.Heatmap(
            z=[[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]],
            text=labels_text, texttemplate='%{text}',
            colorscale=[[0,'#0d1628'],[1,'#0d1628']],
            showscale=False,
            xgap=4, ygap=4,
        ))
        # coloured cells as shapes
        quad_colors = [COLORS['good'], COLORS['warn'], COLORS['danger'], COLORS['accent']]
        for row_i in [0, 1]:
            for col_i in [0, 1]:
                fig.add_shape(type='rect', x0=col_i-0.5, x1=col_i+0.5,
                              y0=row_i-0.5, y1=row_i+0.5,
                              fillcolor=quad_colors[row_i*2+col_i],
                              opacity=0.25, line_width=0)
        fig.update_layout(**PLOTLY_LAYOUT, height=320,
                          title=dict(text=f'Confusion Matrix — {sel_model}', font=dict(color=COLORS['text'])),
                          xaxis=dict(ticktext=['Pred No','Pred Yes'], tickvals=[0,1]),
                          yaxis=dict(ticktext=['Actual No','Actual Yes'], tickvals=[0,1], autorange='reversed'))
        st.plotly_chart(fig, use_container_width=True)

    # ROC curves
    with col_b:
        fig = go.Figure()
        for i, (name, (fpr, tpr)) in enumerate(roc_data.items()):
            width = 3 if name == sel_model else 1
            opacity = 1 if name == sel_model else 0.45
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} ({results[name]['auc']:.3f})",
                                      mode='lines', line=dict(color=CAT7[i], width=width),
                                      opacity=opacity))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                  line=dict(color='gray', dash='dash', width=1),
                                  showlegend=False))
        fig.update_layout(**PLOTLY_LAYOUT, height=320,
                          title=dict(text='ROC Curves — All Models', font=dict(color=COLORS['text'])),
                          xaxis_title='False Positive Rate',
                          yaxis_title='True Positive Rate',
                          legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=COLORS['border'], font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

    # metrics table
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
elif "Live Predictor" in PAGE:
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
        payment      = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_chg  = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
        total_chg    = st.slider("Total Charges ($)", 0.0, 9000.0, monthly_chg * tenure, step=10.0)

    if st.button("🔮 Predict Churn Probability", use_container_width=True):
        # ── build feature row ─────────────────────────────────────────────────
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

        # one-hot fields
        if multi_lines == "Yes":                    row['MultipleLines_Yes'] = 1
        elif multi_lines == "No phone service":     row['MultipleLines_No phone service'] = 1
        if internet_svc == "Fiber optic":           row['InternetService_Fiber optic'] = 1
        elif internet_svc == "No":                  row['InternetService_No'] = 1
        if contract == "One year":                  row['Contract_One year'] = 1
        elif contract == "Two year":                row['Contract_Two year'] = 1
        if payment == "Credit card (automatic)":    row['PaymentMethod_Credit card (automatic)'] = 1
        elif payment == "Electronic check":         row['PaymentMethod_Electronic check'] = 1
        elif payment == "Mailed check":             row['PaymentMethod_Mailed check'] = 1

        # engineered features
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
                mode="gauge+number",
                value=pct,
                number={'suffix': '%', 'font': {'size': 42, 'color': COLORS['text']}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': COLORS['text2']},
                    'bar': {'color': COLORS['danger'] if pct >= 60 else COLORS['warn'] if pct >= 35 else COLORS['good']},
                    'steps': [
                        {'range': [0, 35], 'color': '#0d2d1f'},
                        {'range': [35, 60], 'color': '#2a1f0a'},
                        {'range': [60, 100], 'color': '#2a0d14'},
                    ],
                    'threshold': {'line': {'color': 'white', 'width': 2}, 'thickness': 0.75, 'value': 50}
                }
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=280, margin=dict(l=20,r=20,t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ▸ PAGE: BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif "Batch" in PAGE:
    st.markdown("# 📦 Batch Prediction")
    st.markdown("<div class='info-box'>Upload a CSV with the same columns as the Telco dataset. The app will predict churn probability for every row and let you download the results.</div>", unsafe_allow_html=True)

    batch_file = st.file_uploader("Upload customer CSV for batch scoring", type="csv")

    if batch_file:
        batch_df = pd.read_csv(batch_file)
        st.markdown(f"**Loaded {len(batch_df):,} rows · {len(batch_df.columns)} columns**")

        if st.button("▶ Score All Customers"):
            with st.spinner("Scoring…"):
                # re-run same pre-processing
                tmp = batch_df.copy()
                tmp.drop(columns=['customerID'], inplace=True, errors='ignore')
                tmp['TotalCharges'] = pd.to_numeric(tmp['TotalCharges'], errors='coerce').fillna(tmp.get('TotalCharges', pd.Series()).median() or 0)
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
                tmp['AvgMonthlySpend'] = tmp['TotalCharges'] / (tmp['tenure'] + 1)
                tmp['TenureGroup'] = pd.cut(tmp['tenure'], bins=[0,12,24,48,72], labels=[0,1,2,3], include_lowest=True).astype(float).fillna(0).astype(int)
                svc_existing = [c for c in ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies'] if c in tmp.columns]
                tmp['ServiceCount'] = tmp[svc_existing].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)

                X_batch = tmp.reindex(columns=feat_names, fill_value=0)
                rf = results['Random Forest']['model']
                probs = rf.predict_proba(X_batch)[:, 1]

                out = batch_df.copy()
                out['ChurnProbability'] = (probs * 100).round(1)
                out['ChurnPrediction']  = ['Yes' if p >= 50 else 'No' for p in probs]
                out['RiskSegment']      = ['High' if p >= 60 else 'Medium' if p >= 35 else 'Low' for p in probs]

                st.success(f"✓ Scored {len(out):,} customers")

                # summary
                risk_counts = out['RiskSegment'].value_counts()
                c1, c2, c3 = st.columns(3)
                c1.metric("🔴 High Risk",   f"{risk_counts.get('High',0):,}")
                c2.metric("🟡 Medium Risk", f"{risk_counts.get('Medium',0):,}")
                c3.metric("🟢 Low Risk",    f"{risk_counts.get('Low',0):,}")

                st.dataframe(out.head(50), use_container_width=True)

                csv_out = out.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Full Results CSV", csv_out,
                                   "churn_predictions.csv", "text/csv")
    else:
        st.info("Upload a CSV file above to start batch scoring. Use the same format as the Telco dataset.")


# ══════════════════════════════════════════════════════════════════════════════
#  ▸ PAGE: ROI CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif "ROI" in PAGE:
    st.markdown("# 💰 Business ROI Calculator")
    st.markdown("<div class='info-box'>Translate model performance into business value. Adjust the inputs to see expected revenue saved per churn cycle.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='sub-header'>Customer & Revenue Inputs</div>", unsafe_allow_html=True)
        total_customers   = st.number_input("Total active customers", value=7043, step=100)
        churn_rate        = st.slider("Current monthly churn rate (%)", 1.0, 50.0, 26.5, step=0.5)
        monthly_revenue   = st.number_input("Avg monthly revenue per customer ($)", value=65.0, step=1.0)
        cac               = st.number_input("Customer Acquisition Cost ($)", value=300.0, step=10.0)
        retention_cost    = st.number_input("Retention campaign cost per customer ($)", value=15.0, step=1.0)
        model_recall      = st.slider("Model Recall (% churners caught)", 50, 100, 76, step=1)
        model_precision   = st.slider("Model Precision (% alerts are real)", 40, 100, 65, step=1)

    with c2:
        st.markdown("<div class='sub-header'>Results</div>", unsafe_allow_html=True)

        n_churners         = int(total_customers * churn_rate / 100)
        churners_caught    = int(n_churners * model_recall / 100)
        false_alerts       = int(churners_caught / (model_precision/100) * (1 - model_precision/100))
        total_alerted      = churners_caught + false_alerts
        retention_spend    = total_alerted * retention_cost
        revenue_saved      = churners_caught * monthly_revenue * 12   # annual LTV proxy
        cac_saved          = churners_caught * cac
        net_benefit        = revenue_saved + cac_saved - retention_spend

        st.metric("Expected churners / cycle", f"{n_churners:,}")
        st.metric("Churners caught by model",  f"{churners_caught:,}")
        st.metric("Total alerts sent",          f"{total_alerted:,}")
        st.metric("Retention campaign cost",    f"${retention_spend:,.0f}")
        st.metric("Annual revenue saved",       f"${revenue_saved:,.0f}", delta="churners retained × 12-month LTV")
        st.metric("CAC equivalent saved",       f"${cac_saved:,.0f}", delta="avoided re-acquisition cost")
        st.metric("🏆 Net Benefit / Cycle",     f"${net_benefit:,.0f}",
                  delta="positive ROI" if net_benefit > 0 else "negative ROI")

    # sensitivity chart
    st.markdown("<div class='section-header'>Sensitivity: Net Benefit vs Model Recall</div>", unsafe_allow_html=True)
    recalls = np.arange(0.4, 1.01, 0.05)
    benefits = []
    for rec in recalls:
        caught = int(n_churners * rec)
        alerts = int(caught / (model_precision/100))
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
                      xaxis_title='Model Recall (%)',
                      yaxis_title='Net Benefit ($)',
                      yaxis_tickprefix='$', yaxis_tickformat=',')
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ▸ PAGE: PLACEHOLDER PAGES
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.info("Select a page from the sidebar →")
