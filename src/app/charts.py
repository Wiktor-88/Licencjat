from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


COLORS = {
    "Spółka": "#2B6CB0", "QQQ": "#718096", "Nadwyżkowa (AR)": "#2F855A",
    "Pozytywny": "#2F855A", "Neutralny": "#718096", "Negatywny": "#C53030",
}
VARIANT_COLORS = {"A": "#A0AEC0", "B": "#5B8DB8", "C": "#2B6CB0"}
METRIC_LABELS = {
    "Pooled_ROC_AUC": "ROC-AUC", "Pooled_Balanced_Accuracy": "Balanced accuracy",
    "Pooled_F1": "F1", "Pooled_Accuracy": "Accuracy",
}


def _style(figure: go.Figure, height: int | None = None) -> go.Figure:
    figure.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#111827", size=13),
        hoverlabel=dict(bgcolor="#F7F9FC", font_color="#111827", bordercolor="#A0AEC0"),
        legend=dict(font=dict(color="#111827"), title_font=dict(color="#111827"),
                    bgcolor="rgba(247,249,252,.78)"),
        margin=dict(l=18, r=18, t=18, b=18), height=height,
    )
    axis_style = dict(showline=False, gridcolor="#E2E8F0", zerolinecolor="#A0AEC0",
                      title_font=dict(color="#111827", size=14), tickfont=dict(color="#111827", size=12))
    figure.update_xaxes(**axis_style)
    figure.update_yaxes(**axis_style)
    return figure


def _number(row: pd.Series, *names: str) -> float | None:
    lookup = {str(name).lower(): name for name in row.index}
    for name in names:
        column = lookup.get(name.lower())
        if column:
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    return None


def sentiment_chart(row: pd.Series) -> go.Figure | None:
    values = {
        "Pozytywny": _number(row, "Mean_Positive"),
        "Neutralny": _number(row, "Mean_Neutral"),
        "Negatywny": _number(row, "Mean_Negative"),
    }
    values = {key: value for key, value in values.items() if value is not None}
    if not values:
        return None
    figure = px.bar(x=list(values.values()), y=list(values), orientation="h", labels={"x": "Udział / prawdopodobieństwo", "y": ""}, color=list(values), color_discrete_map=COLORS)
    figure.update_layout(showlegend=False)
    figure.update_xaxes(range=[0, 1], tickformat=".0%")
    return _style(figure, 260)


def reaction_chart(row: pd.Series) -> go.Figure | None:
    values = {"Spółka": _number(row, "Event_Return_1D"), "QQQ": _number(row, "QQQ_Event_Return_1D"), "Nadwyżkowa (AR)": _number(row, "Abnormal_Event_Return_1D")}
    values = {key: value for key, value in values.items() if value is not None}
    if not values:
        return None
    bar_colors = ["#2F855A" if value >= 0 else "#C53030" for value in values.values()]
    figure = px.bar(x=list(values), y=list(values.values()), labels={"x": "", "y": "Stopa zwrotu"}, text_auto=".2%")
    figure.update_traces(marker_color=bar_colors)
    figure.update_yaxes(tickformat=".1%", zeroline=True, zerolinecolor="#475569")
    figure.update_layout(showlegend=False)
    return _style(figure, 260)


def reaction_distribution_chart(values: pd.Series) -> go.Figure | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return None
    frame = pd.DataFrame({"Kierunek": ["Ujemna", "Dodatnia"],
                          "Liczba obserwacji": [(values < 0).sum(), (values >= 0).sum()]})
    figure = px.bar(frame, x="Kierunek", y="Liczba obserwacji", text_auto=True)
    figure.update_traces(marker_color=["#C53030", "#2F855A"])
    figure.update_layout(showlegend=False)
    return _style(figure, 320)


def prediction_chart(predictions: pd.DataFrame) -> go.Figure | None:
    column = "y_prob"
    if column not in predictions:
        return None
    values = pd.to_numeric(predictions[column], errors="coerce")
    if values.notna().sum() == 0:
        return None
    frame = predictions.assign(**{column: values}).dropna(subset=[column])
    figure = px.bar(frame, x="Rodzina", y=column, range_y=[0, 1], text_auto=".2f", color=column,
                    color_continuous_scale=[(0, "#C53030"), (.5, "#718096"), (1, "#2F855A")],
                    hover_data=["Model", "Wariant", "Poprawna"],
                    labels={column: "Prawdopodobieństwo dodatniego zwrotu open–close", "Rodzina": "Model"})
    figure.add_hline(y=.5, line_dash="dash", line_color="#718096")
    figure.update_layout(coloraxis_showscale=False)
    return _style(figure, 390)


def model_comparison_chart(metrics: pd.DataFrame, metric: str = "Pooled_ROC_AUC") -> go.Figure | None:
    if not metric or "Model" not in metrics:
        return None
    figure = px.bar(metrics, x="Rodzina", y=metric, text_auto=".3f", color="Wariant", barmode="group",
                    color_discrete_map=VARIANT_COLORS, hover_data=["Model", "Zakres danych"],
                    labels={"Rodzina": "Model", metric: METRIC_LABELS.get(metric, metric)})
    figure.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return _style(figure, 430)
