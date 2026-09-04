from pathlib import Path
from html import escape
import logging

import pandas as pd
import streamlit as st

from src.app.charts import model_comparison_chart, prediction_chart, reaction_chart, reaction_distribution_chart, sentiment_chart
from src.app.data import ProjectData, VARIANT_DESCRIPTIONS, first_value, format_accession
from src.app.theme import apply_theme


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
XAI_DIR = ROOT / "artifacts" / "xai"
PUBLICATION_LABELS = {
    "AFTER_HOURS": "Po sesji", "PRE_MARKET": "Przed sesją",
    "INTRADAY": "W trakcie sesji", "NON_TRADING_DAY": "Poza sesją",
}

st.set_page_config(page_title="Ewaluacja modeli predykcyjnych", page_icon="📊", layout="wide")
apply_theme()


@st.cache_resource
def get_data() -> ProjectData:
    return ProjectData(DATA_DIR, XAI_DIR)


@st.cache_data(show_spinner=False)
def load_events(_store: ProjectData) -> pd.DataFrame:
    return _store.events()


def show_missing_data() -> None:
    st.warning("Nie znaleziono `data/processed/model_dataset.csv`.")
    st.caption(f"Sprawdzany katalog: {DATA_DIR}")


def variant_help() -> None:
    st.caption("**A** — rynek  \n**B** — rynek + SEC  \n**C** — rynek + SEC + sentyment FinBERT")


def overview(store: ProjectData, events: pd.DataFrame) -> None:
    st.title("Ewaluacja modeli predykcyjnych")
    st.caption("Aplikacja demonstracyjna prezentująca zapisane wyniki modeli i ich interpretację")
    if events.empty:
        show_missing_data()
        return

    dates = events["Filing_Date"]
    primary_filings = events[events["Use_In_Primary_Model"] == 1] if "Use_In_Primary_Model" in events else events
    primary = primary_filings.drop_duplicates(["Ticker", "Event_Session"])
    a, b, c, d = st.columns(4)
    a.metric("Komunikaty", f"{events['Accession'].nunique():,}")
    b.metric("Spółki", events["Ticker"].nunique())
    c.metric("Zakres danych", f"{dates.min().year}–{dates.max().year}")
    d.metric("Obserwacje modelowe", f"{len(primary):,}")

    st.subheader("Porównanie modeli")
    variant_help()
    metrics = store.model_metrics()
    if metrics.empty:
        st.info("Nie znaleziono plików `*_summary.csv`.")
    else:
        f1, f2 = st.columns([1, 2])
        variant = f1.selectbox("Wariant", ["Wszystkie", "A", "B", "C"], index=3)
        choices = {
            "ROC-AUC": "Pooled_ROC_AUC", "Balanced accuracy": "Pooled_Balanced_Accuracy",
            "F1": "Pooled_F1", "Accuracy": "Pooled_Accuracy",
        }
        label = f2.selectbox("Metryka OOS", choices)
        shown = metrics if variant == "Wszystkie" else metrics[metrics["Wariant"] == variant]
        chart = model_comparison_chart(shown, choices[label])
        if chart:
            st.plotly_chart(chart, width="stretch")
        columns = ["Rodzina", "Wariant", "Model", choices[label]]
        table = shown[columns].rename(columns={"Rodzina": "Model", "Model": "Nazwa modelu", choices[label]: label})
        st.dataframe(table, width="stretch", hide_index=True,
                     column_config={label: st.column_config.NumberColumn(label, format="%.3f")})

    st.subheader("Rozkład nadwyżkowej stopy zwrotu")
    return_column = ("Tradable_Abnormal_Return_1D" if "Tradable_Abnormal_Return_1D" in primary
                     else "Abnormal_Event_Return_1D")
    values = pd.to_numeric(primary[return_column], errors="coerce").dropna()
    chart = reaction_distribution_chart(values)
    if chart:
        st.plotly_chart(chart, width="stretch")
    st.caption("Target modelowy: zwrot spółki od otwarcia do zamknięcia sesji pomniejszony o analogiczny zwrot QQQ.")


def snippet_card(block: pd.Series | None, positive: bool) -> None:
    if block is None:
        st.info("Nie znaleziono fragmentu.")
        return
    probability = block["Prob_Positive"] if positive else block["Prob_Negative"]
    st.metric("Prawdopodobieństwo FinBERT", f"{float(probability):.1%}")
    st.caption(f"Źródło: {block['Source_Type']} · blok {block['Block_ID']}")
    css_class = "sentiment-positive" if positive else "sentiment-negative"
    st.markdown(f'<div class="sentiment-block {css_class}">{escape(str(block["Text_Snippet"]))}</div>',
                unsafe_allow_html=True)


def event_explorer(store: ProjectData, events: pd.DataFrame) -> None:
    st.title("Analiza zdarzeń (SEC)")
    st.caption("Komunikat SEC → sentyment → reakcja rynku → zapisane predykcje OOS")
    if events.empty:
        show_missing_data()
        return

    filtered = events.copy()
    f1, f2, f3 = st.columns(3)
    ticker = f1.selectbox("Ticker", sorted(filtered["Ticker"].dropna().astype(str).unique()))
    filtered = filtered[filtered["Ticker"].astype(str) == ticker]
    years = sorted(filtered["Filing_Date"].dt.year.dropna().astype(int).unique(), reverse=True)
    year = f2.selectbox("Rok", ["Wszystkie", *years])
    if year != "Wszystkie":
        filtered = filtered[filtered["Filing_Date"].dt.year == year]
    accessions = filtered["Accession"].dropna().astype(str).unique().tolist()
    accession = f3.selectbox("Accession", accessions, format_func=format_accession)
    row = filtered[filtered["Accession"].astype(str) == accession].iloc[0]

    meta = st.columns(4)
    meta[0].metric("Spółka", row["Ticker"])
    meta[1].metric("Data filingu", first_value(row, "Filing_Date"))
    meta[2].metric("Sesja zdarzenia", first_value(row, "Event_Session"))
    publication = first_value(row, "Publication_Period")
    meta[3].metric("Publikacja", PUBLICATION_LABELS.get(publication, publication))
    st.caption(
        f"Przyjęcie przez SEC: {first_value(row, 'Acceptance_DateTime_ET')} · "
        f"Items: {first_value(row, 'Item_Numbers')} · źródła: {first_value(row, 'Source_Types')}"
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Sentyment komunikatu")
        chart = sentiment_chart(row)
        if chart:
            st.plotly_chart(chart, width="stretch")
        st.metric("Średni sentyment netto", f"{float(row['Mean_Net_Sentiment']):+.3f}")
    with right:
        st.subheader("Reakcja rynku (Close–Close)")
        chart = reaction_chart(row)
        if chart:
            st.plotly_chart(chart, width="stretch")
        st.caption("Miara opisowa pokazuje reakcję od poprzedniego zamknięcia do zamknięcia sesji zdarzenia.")

    st.markdown("#### Zwrot możliwy do wykorzystania po publikacji")
    tradable = st.columns(3)
    return_columns = [("Dzienna stopa zwrotu spółki (Open–Close)", "Event_Return_Open_Close"),
                      ("Stopa zwrotu QQQ (Open–Close)", "QQQ_Event_Return_Open_Close"),
                      ("Nadwyżkowa stopa zwrotu (AR)", "Tradable_Abnormal_Return_1D")]
    for column, (label, field) in zip(tradable, return_columns):
        value = pd.to_numeric(pd.Series([row.get(field)]), errors="coerce").iloc[0]
        column.metric(label, f"{value:+.2%}" if pd.notna(value) else "—")
    st.caption("To ta miara wyznacza klasę modelową. Jest obserwacją rynkową, a nie dowodem przyczynowego wpływu filingu.")

    st.subheader("Najbardziej charakterystyczne fragmenty")
    positive, negative = store.sentiment_snippets(row)
    pos_col, neg_col = st.columns(2)
    with pos_col:
        st.markdown("#### Najbardziej pozytywny")
        snippet_card(positive, True)
    with neg_col:
        st.markdown("#### Najbardziej negatywny")
        snippet_card(negative, False)

    st.subheader("Predykcje modeli OOS")
    variant_help()
    predictions = store.event_predictions(accession, str(row["Ticker"]), row.get("Event_Session"))
    if predictions.empty:
        st.info("Predykcje OOS są dostępne dla zdarzeń z lat testowych 2023–2026.")
    else:
        variant = st.radio("Wariant modelu", ["A", "B", "C"], index=2, horizontal=True)
        shown = predictions[predictions["Wariant"] == variant]
        chart = prediction_chart(shown)
        if chart:
            st.plotly_chart(chart, width="stretch")
        table = shown.rename(columns={"y_prob": "Prawdopodobieństwo dodatniego zwrotu"})
        st.dataframe(table, width="stretch", hide_index=True, column_config={
            "Prawdopodobieństwo dodatniego zwrotu": st.column_config.ProgressColumn(format="percent", min_value=0., max_value=1.),
        })

    with st.expander("Wszystkie dostępne pola zdarzenia"):
        metadata = pd.DataFrame({"Pole": row.index, "Wartość": [str(value) for value in row.values]})
        st.dataframe(metadata, width="stretch", hide_index=True)


def xai_explorer(store: ProjectData) -> None:
    st.title("Interpretowalność modeli (XAI)")
    st.caption("Gotowe wyniki interpretacji zapisane po zakończeniu eksperymentu — bez ponownego obliczania modeli")
    models = store.xai_models()
    if not models:
        st.warning("Nie znaleziono katalogu `artifacts/xai`.")
        return
    model = st.selectbox("Model", models)
    scope = st.radio("Zakres", ["Globalny", "Lokalny"], horizontal=True)

    if scope == "Globalny":
        variants = store.xai_variants(model)
        if not variants:
            st.warning("Ten model nie ma zapisanych wariantów A/B/C.")
            return
        variant = st.selectbox("Wariant", variants, index=max(0, len(variants) - 1))
        st.caption(f"Wariant {variant}: {VARIANT_DESCRIPTIONS[variant]}")
        years = store.xai_years(model, variant)
        year = st.selectbox("Agregacja", ["Stabilność między foldami", *years])
        artifacts = store.global_artifacts(model, variant, None if isinstance(year, str) else year)
    else:
        st.info("Lokalne TP/TN/FP/FN zostały zapisane dla pełnego wariantu C.")
        examples = store.xai_local_events(model)
        if examples.empty:
            st.warning("Ten model nie ma zapisanych przykładów lokalnych.")
            return
        labels = {}
        for _, example in examples.iterrows():
            accession = str(example["Accession"])
            labels[accession] = f"{example.get('Ticker', '')} · {example.get('XAI_Example_Type', '')} · {accession}"
        accession = st.selectbox("Przykład", list(labels), format_func=labels.get)
        artifacts = store.local_artifacts(model, accession)

    if not artifacts:
        st.info("Brak artefaktów dla wybranego zestawu filtrów.")
        return
    chosen = st.selectbox("Wynik eksperymentu", artifacts, format_func=lambda path: path.name)
    store.render_artifact(chosen)


store = get_data()
events = load_events(store)
page = st.sidebar.radio("Widok", ["Ewaluacja modeli", "Analiza zdarzeń (SEC)", "Interpretowalność modeli (XAI)"])
st.sidebar.caption(f"Projekt: `{ROOT}`")
if st.sidebar.button("Odśwież dane"):
    st.cache_data.clear()
    st.rerun()

if page == "Ewaluacja modeli":
    overview(store, events)
elif page == "Analiza zdarzeń (SEC)":
    event_explorer(store, events)
else:
    xai_explorer(store)
