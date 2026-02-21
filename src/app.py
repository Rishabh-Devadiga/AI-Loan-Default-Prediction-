import sys
import pathlib
import streamlit as st

# ensure local 'pages' package (same folder) is importable
APP_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    layout="wide",
    page_title="CreditRisk AI",
    page_icon="shield",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      --bg: #f3f5f8;
      --surface: #ffffff;
      --text: #0f172a;
      --muted: #5b6b85;
      --border: #dbe3ef;
      --primary: #2563eb;
      --success: #1f9d66;
      --warning: #e49b0f;
      --danger: #d9363e;
      --sidebar-bg: #0b152a;
      --sidebar-bg-2: #101f3c;
      --sidebar-text: #d7e3ff;
      --sidebar-active: #1f3a69;
    }
    .stApp { background: var(--bg); color: var(--text); }
    [data-testid="stMain"], [data-testid="stMainBlockContainer"] {
      --text-color: #0f172a;
      --background-color: #f3f5f8;
      --secondary-background-color: #ffffff;
      color: var(--text) !important;
    }
    [data-testid="stMain"], [data-testid="stMain"] * {
      color: var(--text) !important;
    }
    [data-testid="stMain"] p, [data-testid="stMain"] span, [data-testid="stMain"] label,
    [data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3, [data-testid="stMain"] h4,
    [data-testid="stMain"] h5, [data-testid="stMain"] h6, [data-testid="stMain"] li, [data-testid="stMain"] div {
      color: var(--text) !important;
    }
    [data-testid="stMain"] small, [data-testid="stMain"] .caption { color: var(--muted) !important; }
    [data-testid="stMain"] [data-testid="stWidgetLabel"] p,
    [data-testid="stMain"] [data-testid="stWidgetLabel"] label,
    [data-testid="stMain"] [data-testid="stWidgetLabel"] span {
      color: var(--text) !important;
      opacity: 1 !important;
    }
    [data-testid="stMain"] [data-baseweb="input"] > div,
    [data-testid="stMain"] [data-baseweb="base-input"] > div,
    [data-testid="stMain"] [data-baseweb="select"] > div,
    [data-testid="stMain"] [data-baseweb="select"] > div > div {
      background: #ffffff !important;
      border-color: #cfd8e6 !important;
      color: var(--text) !important;
    }
    [data-testid="stMain"] input,
    [data-testid="stMain"] textarea,
    [data-testid="stMain"] [data-baseweb="select"] input {
      color: var(--text) !important;
      -webkit-text-fill-color: var(--text) !important;
      background: transparent !important;
      opacity: 1 !important;
    }
    [data-testid="stMain"] input::placeholder,
    [data-testid="stMain"] textarea::placeholder {
      color: #64748b !important;
      opacity: 1 !important;
    }
    [data-testid="stMain"] [data-baseweb="select"] svg,
    [data-testid="stMain"] [data-baseweb="input"] svg {
      fill: var(--text) !important;
      color: var(--text) !important;
    }
    [data-testid="stMain"] [data-baseweb="input"] button,
    [data-testid="stMain"] [data-baseweb="base-input"] button {
      color: var(--text) !important;
      background: #ffffff !important;
      border-left: 1px solid #cfd8e6 !important;
    }
    [data-testid="stMain"] [data-baseweb="input"] [role="group"],
    [data-testid="stMain"] [data-baseweb="base-input"] [role="group"],
    [data-testid="stMain"] [data-baseweb="input"] [data-testid="stNumberInputStepUp"],
    [data-testid="stMain"] [data-baseweb="input"] [data-testid="stNumberInputStepDown"] {
      background: #ffffff !important;
      color: var(--text) !important;
    }
    [data-testid="stMain"] [data-baseweb="input"] [data-testid="stNumberInputStepUp"] svg,
    [data-testid="stMain"] [data-baseweb="input"] [data-testid="stNumberInputStepDown"] svg {
      fill: var(--text) !important;
      color: var(--text) !important;
    }
    [data-testid="stMain"] button[data-testid="stNumberInputStepUp"],
    [data-testid="stMain"] button[data-testid="stNumberInputStepDown"] {
      background: #ffffff !important;
      color: var(--text) !important;
      border-left: 1px solid #cfd8e6 !important;
      box-shadow: none !important;
    }
    [data-testid="stMain"] button[data-testid="stNumberInputStepUp"]:hover,
    [data-testid="stMain"] button[data-testid="stNumberInputStepDown"]:hover {
      background: #f8fafc !important;
      color: var(--text) !important;
    }
    [data-testid="stMain"] .stButton > button,
    [data-testid="stMain"] .stFormSubmitButton > button,
    [data-testid="stMain"] button[kind="primary"],
    [data-testid="stMain"] [data-baseweb="button"] {
      background: #ffffff !important;
      color: var(--text) !important;
      border: 1px solid #cfd8e6 !important;
      box-shadow: none !important;
    }
    [data-testid="stMain"] .stButton > button:hover,
    [data-testid="stMain"] .stFormSubmitButton > button:hover,
    [data-testid="stMain"] button[kind="primary"]:hover,
    [data-testid="stMain"] [data-baseweb="button"]:hover {
      background: #f8fafc !important;
      border-color: #bfcadc !important;
      color: var(--text) !important;
    }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, var(--sidebar-bg), var(--sidebar-bg-2));
      border-right: 1px solid #1e335b;
    }
    [data-testid="stSidebarNav"], [data-testid="stSidebarNavSeparator"] {
      display: none !important;
    }
    [data-testid="stSidebar"] * { color: var(--sidebar-text); }
    .brand-wrap {
      border: 1px solid #23457f; border-radius: 14px; padding: 14px 12px; margin-bottom: 14px;
      background: rgba(21, 40, 74, 0.75);
    }
    .brand-title { font-size: 1.12rem; font-weight: 700; margin-bottom: 2px; }
    .brand-sub { font-size: 0.85rem; opacity: 0.85; }
    .metric-card {
      background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 16px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .metric-label { color: var(--muted); font-size: 0.94rem; }
    .metric-value { font-size: 2rem; font-weight: 700; margin: 6px 0 4px 0; line-height: 1.05; }
    .metric-note { font-size: 0.9rem; }
    .metric-note.note-green { color: #1f9d66 !important; }
    .metric-note.note-red { color: #d9363e !important; }
    .panel {
      background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 16px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] {
      border: 1px solid var(--border); border-radius: 10px; padding: 0.3rem 0.9rem;
      background: var(--surface);
    }
    .stTabs [aria-selected="true"] {
      border-color: #1d4ed8 !important; color: #1d4ed8 !important; font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div class="brand-wrap">
      <div class="brand-title">CreditRisk AI</div>
      <div class="brand-sub">Loan Default Prediction</div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Loan Risk Prediction", "Model Performance", "Business Impact"],
    label_visibility="collapsed",
)

if page == "Overview":
    import pages.overview as page_module
elif page == "Loan Risk Prediction":
    import pages.predict as page_module
elif page == "Model Performance":
    import pages.model_performance as page_module
else:
    import pages.business_impact as page_module

page_module.app()
