from __future__ import annotations

from pathlib import Path
import logging

import pandas as pd
import streamlit as st


logger = logging.getLogger(__name__)
MODEL_DIRS = {
    "Logistic Regression": "logistic_regression", "Decision Tree": "decision_tree",
    "XGBoost": "xgboost", "CatBoost": "catboost", "TabNet": "tabnet",
    "LSTM": "lstm", "Transformer": "transformer",
}
PREDICTION_FILES = {
    "Logistic Regression": "oos_predictions.csv", "Decision Tree": "decision_tree_oos_predictions.csv",
    "XGBoost": "xgboost_oos_predictions.csv", "CatBoost": "catboost_oos_predictions.csv",
    "TabNet": "tabnet_oos_predictions.csv", "LSTM": "lstm_oos_predictions.csv",
    "Transformer": "transformer_oos_predictions.csv",
}
VARIANT_DESCRIPTIONS = {
    "A": "dane historyczne rynku", "B": "rynek + cechy SEC", "C": "rynek + cechy SEC + sentyment FinBERT",
}
LOGISTIC_FINAL_MODELS = {
    "MODEL A - MARKET": "A", "MODEL B - MARKET + SEC": "B",
    "MODEL C - MARKET + SEC + FINBERT": "C",
}


def model_variant(model: str) -> str | None:
    name = str(model).upper()
    if name in LOGISTIC_FINAL_MODELS:
        return LOGISTIC_FINAL_MODELS[name]
    for variant in ("A", "B", "C"):
        if f" {variant} -" in name:
            return variant
    return None


def format_accession(value: object) -> str:
    text = str(value)
    return text if len(text) <= 28 else f"{text[:12]}…{text[-12:]}"


def first_value(row: pd.Series, *columns: str | None) -> str:
    for column in columns:
        if column and column in row.index and pd.notna(row[column]):
            value = row[column]
            if "date" in column.lower() or "session" in column.lower():
                date = pd.to_datetime(value, errors="coerce")
                if pd.notna(date):
                    return date.strftime("%Y-%m-%d") if date.hour == date.minute == date.second == 0 else date.strftime("%Y-%m-%d %H:%M")
                return str(value)
            return str(value)
    return "—"


class ProjectData:
    def __init__(self, data_dir: Path, xai_dir: Path) -> None:
        self.data_dir, self.xai_dir = data_dir, xai_dir

    @staticmethod
    def column(frame: pd.DataFrame, *names: str) -> str | None:
        lookup = {str(column).lower(): column for column in frame.columns}
        return next((lookup[name.lower()] for name in names if name.lower() in lookup), None)

    @staticmethod
    @st.cache_data(show_spinner=False)
    def _read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
        try:
            return pd.read_csv(path, low_memory=False, **kwargs)
        except Exception as error:
            logger.warning("Nie udało się wczytać %s: %s", path, error)
            return pd.DataFrame()

    def events(self) -> pd.DataFrame:
        path = self.data_dir / "model_dataset.csv"
        if not path.exists():
            logger.warning("Nie znaleziono zbioru modelowego: %s", path)
            return pd.DataFrame()
        frame = self._read_csv(path)
        # Czas akceptacji ma mieszane offsety EST/EDT, więc zostaje tekstem.
        for column in ("Filing_Date", "Event_Session", "Feature_Cutoff_Session"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
        columns = [column for column in ("Ticker", "Filing_Date") if column in frame]
        return frame.sort_values(columns).reset_index(drop=True) if columns else frame

    def prediction_files(self) -> list[Path]:
        return [self.data_dir / name for name in PREDICTION_FILES.values() if (self.data_dir / name).exists()]

    def event_predictions(self, accession: str, ticker: str | None = None,
                          event_session: object | None = None) -> pd.DataFrame:
        frames = []
        for family, filename in PREDICTION_FILES.items():
            path = self.data_dir / filename
            if not path.exists():
                continue
            frame = self._read_csv(path)
            access_col = self.column(frame, "Accession")
            if not access_col:
                continue
            matches = frame[frame[access_col].astype(str) == str(accession)].copy()
            if matches.empty and ticker is not None and event_session is not None:
                ticker_col = self.column(frame, "Ticker")
                session_col = self.column(frame, "Event_Session")
                if ticker_col and session_col:
                    sessions = pd.to_datetime(frame[session_col], errors="coerce").dt.normalize()
                    selected_session = pd.to_datetime(event_session, errors="coerce")
                    if pd.notna(selected_session):
                        matches = frame[(frame[ticker_col].astype(str) == str(ticker))
                                        & (sessions == selected_session.normalize())].copy()
            if family == "Logistic Regression" and "Model" in matches:
                matches = matches[matches["Model"].str.upper().isin(LOGISTIC_FINAL_MODELS)]
            if matches.empty:
                continue
            matches.insert(0, "Rodzina", family)
            matches.insert(1, "Wariant", matches["Model"].map(model_variant))
            frames.append(matches)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        result["Sygnał"] = result["y_pred"].map({1: "Dodatni", 0: "Ujemny"})
        result["Rzeczywista klasa"] = result["y_true"].map({1: "Dodatnia", 0: "Ujemna"})
        result["Poprawna"] = (result["y_pred"] == result["y_true"]).map({True: "Tak", False: "Nie"})
        result["Zakres danych"] = result["Wariant"].map(VARIANT_DESCRIPTIONS)
        columns = ["Rodzina", "Wariant", "Zakres danych", "Model", "y_prob", "Sygnał", "Rzeczywista klasa", "Poprawna", "Test_Year"]
        return result[[column for column in columns if column in result]]

    def sentiment_snippets(self, event: pd.Series) -> tuple[pd.Series | None, pd.Series | None]:
        path = self.data_dir / "sentiment_blocks.csv"
        if not path.exists():
            return None, None
        columns = ["Accession", "Source_Type", "Block_ID", "Text_Snippet", "Prob_Positive", "Prob_Negative", "Predicted_Label"]
        blocks = self._read_csv(path, usecols=columns, dtype={"Accession": str, "Source_Type": str, "Block_ID": str})
        if blocks.empty:
            return None, None
        filing = blocks[blocks["Accession"] == str(event["Accession"])]

        def find(prefix: str) -> pd.Series | None:
            block_id = str(event.get(f"Most_{prefix}_Block_ID", ""))
            source = str(event.get(f"Most_{prefix}_Source_Type", ""))
            found = filing[(filing["Block_ID"] == block_id) & (filing["Source_Type"] == source)]
            if found.empty:
                found = filing[filing["Block_ID"] == block_id]
            return found.iloc[0] if not found.empty else None

        return find("Positive"), find("Negative")

    def model_metrics(self) -> pd.DataFrame:
        frames = []
        for family, filename in PREDICTION_FILES.items():
            summary = "logistic_summary.csv" if filename == "oos_predictions.csv" else filename.replace("_oos_predictions", "_summary")
            path = self.data_dir / summary
            if not path.exists():
                continue
            frame = self._read_csv(path)
            if frame.empty:
                continue
            if family == "Logistic Regression" and "Model" in frame:
                frame = frame[frame["Model"].str.upper().isin(LOGISTIC_FINAL_MODELS)].copy()
            frame.insert(0, "Rodzina", family)
            frame.insert(1, "Wariant", frame["Model"].map(model_variant))
            frame.insert(2, "Zakres danych", frame["Wariant"].map(VARIANT_DESCRIPTIONS))
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def xai_models(self) -> list[str]:
        return [name for name, folder in MODEL_DIRS.items() if (self.xai_dir / folder).is_dir()]

    def _xai_model_dir(self, model: str) -> Path:
        return self.xai_dir / MODEL_DIRS[model]

    def xai_variants(self, model: str) -> list[str]:
        root = self._xai_model_dir(model)
        variants = {path.name for path in root.iterdir() if path.is_dir() and path.name in {"A", "B", "C"}}
        for path in root.glob("*.*"):
            variants.update(variant for variant in ("A", "B", "C") if path.stem.endswith(f"_{variant}"))
        return sorted(variants)

    def xai_years(self, model: str, variant: str) -> list[int]:
        root = self._xai_model_dir(model) / variant
        return sorted(int(path.name.removeprefix("test_")) for path in root.glob("test_*") if path.is_dir() and path.name.removeprefix("test_").isdigit())

    def xai_local_events(self, model: str) -> pd.DataFrame:
        root = self._xai_model_dir(model) / "C"
        path = root / "local_examples.csv"
        if path.exists():
            return self._read_csv(path, dtype={"Accession": str})
        local = root / "local"
        return pd.DataFrame({"Accession": [path.name for path in local.iterdir() if path.is_dir()]}) if local.exists() else pd.DataFrame()

    def global_artifacts(self, model: str, variant: str, year: int | None) -> list[Path]:
        root = self._xai_model_dir(model)
        if year is not None:
            return self._artifact_files(root / variant / f"test_{year}")
        common = [path for path in root.iterdir() if path.is_file() and path.suffix.lower() != ".json"
                  and not path.name.startswith("local_")]
        variant_files = [path for path in common if path.stem.endswith(f"_{variant}")]
        stability = [path for path in common if "stability" in path.stem or "by_fold" in path.stem]
        return sorted(set(variant_files + stability))

    def local_artifacts(self, model: str, accession: str) -> list[Path]:
        return self._artifact_files(self._xai_model_dir(model) / "C" / "local" / str(accession))

    @staticmethod
    def _artifact_files(folder: Path) -> list[Path]:
        extensions = {".png", ".jpg", ".jpeg", ".svg", ".csv", ".txt", ".md", ".html"}
        return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in extensions) if folder.exists() else []

    def render_artifact(self, path: Path) -> None:
        suffix = path.suffix.lower()
        st.caption(str(path.relative_to(self.xai_dir.parent.parent)))
        if suffix in {".png", ".jpg", ".jpeg", ".svg"}:
            st.image(str(path), width="stretch")
        elif suffix == ".csv":
            st.dataframe(self._read_csv(path), width="stretch", hide_index=True)
        elif suffix == ".html":
            st.components.v1.html(path.read_text(encoding="utf-8"), height=650, scrolling=True)
        else:
            st.code(path.read_text(encoding="utf-8"), language="markdown")
