import streamlit as st


def apply_theme() -> None:
    # Styl jest tutaj, żeby aplikacja nie potrzebowała osobnego frontendu.
    st.markdown("""
        <style>
        :root {
            --navy: #172033;
            --muted: #64748B;
            --line: #DCE3ED;
            --surface: rgba(255, 255, 255, .92);
            --primary: #4F6FEF;
        }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 88% 4%, rgba(79, 111, 239, .09), transparent 24rem),
                linear-gradient(180deg, #F8FAFD 0%, #F3F6FA 100%);
        }
        [data-testid="stHeader"] {
            background: rgba(248, 250, 253, .82);
            backdrop-filter: blur(12px);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #EEF3FA 0%, #F6F8FC 100%);
            border-right: 1px solid var(--line);
        }
        .block-container { max-width: 1380px; padding-top: 2.2rem; padding-bottom: 3rem; }
        h1, h2, h3, h1 span, h2 span, h3 span {
            color: var(--navy) !important;
            letter-spacing: -.025em;
        }
        h1 { font-weight: 750; }
        h2 { margin-top: 1.4rem; }
        p, label { color: #334155; }
        [data-testid="stCaptionContainer"] { color: var(--muted); }
        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid rgba(220, 227, 237, .95);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 24px rgba(23, 32, 51, .045);
        }
        [data-testid="stMetricLabel"] { color: var(--muted); font-weight: 600; }
        [data-testid="stMetricValue"] { color: var(--navy); font-weight: 720; }
        [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: .45rem;
            box-shadow: 0 8px 24px rgba(23, 32, 51, .035);
            overflow: hidden;
        }
        [role="group"]:has(input[role="combobox"]) {
            background: #FFFFFF !important;
            border: 1px solid var(--line) !important;
            border-radius: 11px !important;
        }
        input[role="combobox"] {
            color: var(--navy) !important;
            background: transparent !important;
        }
        [role="group"]:has(input[role="combobox"]) button {
            color: var(--muted) !important;
            background: transparent !important;
        }
        .stAlert { border-radius: 13px; border-width: 1px; }
        .stButton > button {
            background: #FFFFFF !important;
            border-radius: 11px;
            border-color: var(--line);
            color: var(--navy) !important;
            font-weight: 600;
        }
        .stButton > button p { color: var(--navy) !important; }
        .stButton > button:hover { border-color: var(--primary); color: var(--primary); }
        details {
            background: rgba(255, 255, 255, .75);
            border: 1px solid var(--line) !important;
            border-radius: 13px !important;
        }
        </style>
    """, unsafe_allow_html=True)
