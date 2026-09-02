from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


COLORS = {
    "Spółka": "#4F6FEF", "QQQ": "#94A3B8", "Abnormal": "#F59E0B",
    "Pozytywny": "#0F9D8A", "Neutralny": "#94A3B8", "Negatywny": "#E05A67",
}
VARIANT_COLORS = {"A": "#64748B", "B": "#0F9D8A", "C": "#6D5CE7"}


def _style(figure: go.Figure, height: int | None = None) -> go.Figure:
    figure.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#334155", size=13),
        hoverlabel=dict(bgcolor="#172033", font_color="#FFFFFF", bordercolor="#172033"),
        margin=dict(l=18, r=18, t=18, b=18), height=height,
    )
    figure.update_xaxes(showline=False, gridcolor="#E9EEF5", zerolinecolor="#CBD5E1")
    figure.update_yaxes(showline=False, gridcolor="#E9EEF5", zerolinecolor="#CBD5E1")
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
    values = {"Spółka": _number(row, "Event_Return_1D"), "QQQ": _number(row, "QQQ_Event_Return_1D"), "Abnormal": _number(row, "Abnormal_Event_Return_1D")}
    values = {key: value for key, value in values.items() if value is not None}
    if not values:
        return None
    figure = px.bar(x=list(values), y=list(values.values()), labels={"x": "", "y": "Stopa zwrotu"}, color=list(values), color_discrete_map=COLORS, text_auto=".2%")
    figure.update_yaxes(tickformat=".1%", zeroline=True, zerolinecolor="#475569")
    figure.update_layout(showlegend=False)
    return _style(figure, 260)


def prediction_chart(predictions: pd.DataFrame) -> go.Figure | None:
    column = "y_prob"
    if column not in predictions:
        return None
    values = pd.to_numeric(predictions[column], errors="coerce")
    if values.notna().sum() == 0:
        return None
    frame = predictions.assign(**{column: values}).dropna(subset=[column])
    figure = px.bar(frame, x="Rodzina", y=column, range_y=[0, 1], text_auto=".2f", color=column,
                    color_continuous_scale=[(0, "#E05A67"), (.5, "#F3C969"), (1, "#0F9D8A")],
                    hover_data=["Model", "Wariant", "Poprawna"],
                    labels={column: "Prawdopodobieństwo dodatniego zwrotu open–close", "Rodzina": "Model"})
    figure.add_hline(y=.5, line_dash="dash", line_color="#64748b")
    figure.update_layout(coloraxis_showscale=False)
    return _style(figure, 390)


def model_comparison_chart(metrics: pd.DataFrame, metric: str = "Pooled_ROC_AUC") -> go.Figure | None:
    if not metric or "Model" not in metrics:
        return None
    figure = px.bar(metrics, x="Rodzina", y=metric, text_auto=".3f", color="Wariant", barmode="group",
                    color_discrete_map=VARIANT_COLORS, hover_data=["Model", "Zakres danych"],
                    labels={"Rodzina": "Model", metric: metric.replace("_", " ")})
    figure.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return _style(figure, 430)
