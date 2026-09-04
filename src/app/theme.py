import streamlit as st


def apply_theme() -> None:
    # Styl jest tutaj, żeby aplikacja nie potrzebowała osobnego frontendu.
    st.markdown("""
        <style>
        :root {
            --navy: #111827;
            --muted: #2D3748;
            --background: #E9EEF5;
            --surface: #EEF2F7;
            --surface-light: #F7F9FC;
            --line: rgba(255, 255, 255, .72);
            --primary: #2B6CB0;
            --positive: #2F855A;
            --negative: #C53030;
            --shadow-dark: rgba(148, 163, 184, .34);
            --shadow-light: rgba(255, 255, 255, .92);
            --radius: 14px;
            --radius-small: 10px;
        }
        [data-testid="stAppViewContainer"] { background: var(--background); }
        html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
            color: var(--navy) !important;
        }
        [data-testid="stHeader"] {
            background: rgba(233, 238, 245, .88);
            backdrop-filter: blur(12px);
        }
        [data-testid="stSidebar"] {
            background: var(--background);
            border-right: 1px solid rgba(255, 255, 255, .55);
            box-shadow: 7px 0 18px rgba(148, 163, 184, .18);
        }
        .block-container { max-width: 1380px; padding-top: 2.2rem; padding-bottom: 3rem; }
        h1, h2, h3, h4, h5, h6,
        h1 span, h2 span, h3 span, h4 span, h5 span, h6 span {
            color: var(--navy) !important;
            letter-spacing: -.025em;
        }
        h1 { font-weight: 750; }
        h2 { margin-top: 1.4rem; }
        p, li, label, strong, em,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stWidgetLabel"] p,
        [data-testid="stCaptionContainer"] {
            color: var(--navy) !important;
        }
        code {
            background: #DCE4EE !important;
            border-radius: 5px;
            color: var(--navy) !important;
        }
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, var(--surface-light), #E1E7EF);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            padding: 1.05rem 1.15rem;
            box-shadow: 8px 8px 18px var(--shadow-dark), -8px -8px 18px var(--shadow-light);
            overflow: hidden;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
            color: var(--navy) !important;
            font-weight: 600;
        }
        [data-testid="stMetricValue"] {
            color: var(--navy) !important;
            font-size: 1.75rem;
            font-weight: 720;
        }
        [data-testid="stMetricValue"] > div { color: var(--navy) !important; }
        [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
            background: linear-gradient(145deg, var(--surface-light), #E3E9F1);
            border: 1px solid var(--line);
            border-radius: var(--radius) !important;
            padding: .65rem;
            box-shadow: 9px 9px 20px var(--shadow-dark), -9px -9px 20px var(--shadow-light);
            overflow: hidden !important;
        }
        [data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"]) {
            border-radius: var(--radius) !important;
        }
        [data-testid="stPlotlyChart"] > div,
        [data-testid="stPlotlyChart"] .js-plotly-plot,
        [data-testid="stPlotlyChart"] .plot-container,
        [data-testid="stPlotlyChart"] .svg-container,
        [data-testid="stDataFrame"] > div {
            border-radius: calc(var(--radius) - 2px) !important;
            overflow: hidden !important;
        }
        [data-testid="stPlotlyChart"] .svg-container {
            clip-path: inset(0 round calc(var(--radius) - 2px));
        }
        [data-testid="stPlotlyChart"] .main-svg:first-of-type {
            border-radius: calc(var(--radius) - 2px) !important;
        }
        [data-testid="stDataFrame"] {
            --gdg-bg-cell: #F7F9FC;
            --gdg-bg-header: #E2E8F0;
            --gdg-text-dark: #111827;
            --gdg-text-medium: #1F2937;
            --gdg-border-color: #CBD5E1;
        }
        [role="group"]:has(input[role="combobox"]) {
            background: var(--surface) !important;
            border: 0 !important;
            border-radius: var(--radius-small) !important;
            box-shadow: inset 3px 3px 7px rgba(148, 163, 184, .3),
                        inset -3px -3px 7px rgba(255, 255, 255, .86);
            overflow: hidden;
        }
        [data-baseweb="select"] > div {
            background: var(--surface) !important;
            border: 0 !important;
            border-radius: var(--radius-small) !important;
            box-shadow: inset 3px 3px 7px rgba(148, 163, 184, .3),
                        inset -3px -3px 7px rgba(255, 255, 255, .86);
            color: var(--navy) !important;
            overflow: hidden;
        }
        input[role="combobox"] {
            color: var(--navy) !important;
            background: transparent !important;
        }
        [role="group"]:has(input[role="combobox"]) button {
            color: var(--muted) !important;
            background: transparent !important;
        }
        label[data-testid="stRadioOption"][data-selected="true"] > div > div > div:first-child {
            background-color: var(--primary) !important;
            border-color: var(--primary) !important;
        }
        label[data-testid="stRadioOption"][data-selected="true"] > div > div > div:first-child > div {
            background-color: var(--primary) !important;
        }
        .stAlert {
            border: 1px solid var(--line);
            border-radius: var(--radius-small);
            box-shadow: inset 2px 2px 5px rgba(148, 163, 184, .18),
                        inset -2px -2px 5px rgba(255, 255, 255, .72);
            overflow: hidden;
        }
        .stButton > button {
            background: linear-gradient(145deg, var(--surface-light), #E1E7EF) !important;
            border: 1px solid var(--line);
            border-radius: var(--radius-small);
            color: var(--navy) !important;
            font-weight: 600;
            box-shadow: 5px 5px 11px rgba(148, 163, 184, .3),
                        -5px -5px 11px rgba(255, 255, 255, .86);
        }
        .stButton > button p { color: var(--navy) !important; }
        .stButton > button:hover {
            border-color: rgba(43, 108, 176, .35);
            color: var(--primary) !important;
            transform: translateY(-1px);
        }
        .stButton > button:active {
            box-shadow: inset 3px 3px 7px rgba(148, 163, 184, .32),
                        inset -3px -3px 7px rgba(255, 255, 255, .82);
            transform: translateY(0);
        }
        details {
            background: var(--surface);
            border: 1px solid var(--line) !important;
            border-radius: var(--radius) !important;
            box-shadow: 6px 6px 14px rgba(148, 163, 184, .25),
                        -6px -6px 14px rgba(255, 255, 255, .78);
            overflow: hidden;
        }
        .sentiment-block {
            background: linear-gradient(145deg, var(--surface-light), #E4EAF1);
            border: 1px solid var(--line);
            border-left-width: 4px;
            border-radius: var(--radius);
            color: var(--navy);
            line-height: 1.65;
            padding: 1.25rem 1.35rem;
            box-shadow: 7px 7px 16px rgba(148, 163, 184, .28),
                        -7px -7px 16px rgba(255, 255, 255, .84);
            white-space: pre-wrap;
        }
        .sentiment-positive { border-left-color: var(--positive); }
        .sentiment-negative { border-left-color: var(--negative); }
        [data-testid="stDataFrame"] [role="columnheader"] {
            background: #E4EAF1;
            color: var(--navy) !important;
            font-weight: 650;
        }
        [data-testid="stImage"] img, [data-testid="stImage"] > div {
            border-radius: var(--radius) !important;
            box-shadow: 7px 7px 16px rgba(148, 163, 184, .25),
                        -7px -7px 16px rgba(255, 255, 255, .8);
            overflow: hidden;
        }
        [data-testid="stJson"], [data-testid="stCode"] {
            border-radius: var(--radius) !important;
            box-shadow: inset 3px 3px 7px rgba(148, 163, 184, .22),
                        inset -3px -3px 7px rgba(255, 255, 255, .78);
            overflow: hidden;
        }
        </style>
    """, unsafe_allow_html=True)
