from pathlib import Path
import logging

import pandas as pd
import streamlit as st

from src.app.charts import model_comparison_chart, prediction_chart, reaction_chart, sentiment_chart
from src.app.data import ProjectData, VARIANT_DESCRIPTIONS, first_value, format_accession


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
XAI_DIR = ROOT / "artifacts" / "xai"

st.set_page_config(page_title="SEC Market Reaction Explorer", page_icon="📊", layout="wide")


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
    st.caption("A — rynek · B — rynek + SEC · C — rynek + SEC + sentyment FinBERT")


def overview(store: ProjectData, events: pd.DataFrame) -> None:
    st.title("SEC Market Reaction Explorer")
    st.caption("Demonstrator wyników pracy — korzysta wyłącznie z zapisanych danych i artefaktów")
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
            "ROC AUC": "Pooled_ROC_AUC", "Balanced accuracy": "Pooled_Balanced_Accuracy",
            "F1": "Pooled_F1", "Accuracy": "Pooled_Accuracy",
        }
        label = f2.selectbox("Metryka OOS", choices)
        shown = metrics if variant == "Wszystkie" else metrics[metrics["Wariant"] == variant]
        chart = model_comparison_chart(shown, choices[label])
        if chart:
            st.plotly_chart(chart, width="stretch")
        columns = ["Rodzina", "Wariant", "Model", choices[label]]
        st.dataframe(shown[columns], width="stretch", hide_index=True)

    st.subheader("Rzeczywista reakcja rynku")
    return_column = ("Tradable_Abnormal_Return_1D" if "Tradable_Abnormal_Return_1D" in primary
                     else "Abnormal_Event_Return_1D")
    values = pd.to_numeric(primary[return_column], errors="coerce").dropna()
    st.bar_chart(pd.DataFrame({"Liczba": [(values < 0).sum(), (values >= 0).sum()]}, index=["Ujemna", "Dodatnia"]))
    st.caption("Target modelowy: zwrot spółki od otwarcia do zamknięcia sesji pomniejszony o analogiczny zwrot QQQ.")


def snippet_card(block: pd.Series | None, positive: bool) -> None:
    if block is None:
        st.info("Nie znaleziono fragmentu.")
        return
    probability = block["Prob_Positive"] if positive else block["Prob_Negative"]
    st.metric("Prawdopodobieństwo FinBERT", f"{float(probability):.1%}")
    st.caption(f"Źródło: {block['Source_Type']} · blok {block['Block_ID']}")
    (st.info if positive else st.error)(str(block["Text_Snippet"]))


def event_explorer(store: ProjectData, events: pd.DataFrame) -> None:
    st.title("Event Explorer")
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
    meta[3].metric("Publikacja", first_value(row, "Publication_Period"))
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
        st.subheader("Reakcja rynku close–close")
        chart = reaction_chart(row)
        if chart:
            st.plotly_chart(chart, width="stretch")
        st.caption("Miara opisowa pokazuje reakcję od poprzedniego zamknięcia do zamknięcia sesji zdarzenia.")

    st.markdown("#### Zwrot możliwy do wykorzystania po publikacji")
    tradable = st.columns(3)
    return_columns = [("Spółka open–close", "Event_Return_Open_Close"),
                      ("QQQ open–close", "QQQ_Event_Return_Open_Close"),
                      ("Zwrot abnormalny", "Tradable_Abnormal_Return_1D")]
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
    st.title("XAI Explorer")
    st.caption("Gotowe interpretacje z `artifacts/xai` — bez przeliczania modeli")
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
    chosen = st.selectbox("Artefakt", artifacts, format_func=lambda path: path.name)
    store.render_artifact(chosen)


store = get_data()
events = load_events(store)
page = st.sidebar.radio("Widok", ["Overview", "Event Explorer", "XAI Explorer"])
st.sidebar.caption(f"Projekt: `{ROOT}`")
if st.sidebar.button("Odśwież dane"):
    st.cache_data.clear()
    st.rerun()

if page == "Overview":
    overview(store, events)
elif page == "Event Explorer":
    event_explorer(store, events)
else:
    xai_explorer(store)
