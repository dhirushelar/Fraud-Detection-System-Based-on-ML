"""
Fraud Detection System (FDS) - Dashboard Plotly Chart Builders
Author: Alpha
"""

import json
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go


DARK_THEME_LAYOUT = dict(
    paper_bgcolor="#1e293b",  # slate-800
    plot_bgcolor="#0f172a",   # slate-900
    font=dict(family="Inter, system-ui, sans-serif", color="#e2e8f0", size=12),
    margin=dict(l=40, r=40, t=50, b=40),
    xaxis=dict(
        gridcolor="#334155",
        zerolinecolor="#475569",
        tickfont=dict(color="#94a3b8"),
    ),
    yaxis=dict(
        gridcolor="#334155",
        zerolinecolor="#475569",
        tickfont=dict(color="#94a3b8"),
    ),
    legend=dict(
        bgcolor="rgba(15, 23, 42, 0.7)",
        bordercolor="#334155",
        borderwidth=1,
        font=dict(color="#e2e8f0"),
    ),
)


def create_fraud_trend_chart(trend_records: List[Dict[str, Any]]) -> str:
    """
    Renders time-series chart showing total transactions vs. flagged anomalies.
    """
    if not trend_records:
        # Fallback dummy data for visualization
        dates = pd.date_range(end=pd.Timestamp.now(), periods=14, freq="D").strftime("%b %d").tolist()
        total_counts = [120, 145, 132, 160, 175, 190, 165, 180, 210, 195, 220, 235, 210, 240]
        flagged_counts = [1, 2, 0, 1, 3, 2, 1, 4, 2, 1, 3, 5, 2, 3]
    else:
        dates = [r["date"] for r in trend_records]
        total_counts = [r["total"] for r in trend_records]
        flagged_counts = [r["flagged"] for r in trend_records]

    fig = go.Figure()

    # Total Transactions Bar
    fig.add_trace(
        go.Bar(
            x=dates,
            y=total_counts,
            name="Normal Ingest Volume",
            marker=dict(color="#3b82f6", opacity=0.7),
            hovertemplate="Date: %{x}<br>Total: %{y}<extra></extra>",
        )
    )

    # Flagged Anomalies Line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=flagged_counts,
            name="Flagged Anomalies",
            mode="lines+markers",
            line=dict(color="#ef4444", width=3),
            marker=dict(size=8, color="#ef4444", symbol="circle"),
            yaxis="y2",
            hovertemplate="Date: %{x}<br>Flagged: %{y}<extra></extra>",
        )
    )

    layout = DARK_THEME_LAYOUT.copy()
    layout.update(
        title=dict(text="<b>Transaction Volume & Flagged Anomalies Over Time</b>", font=dict(size=14, color="#f8fafc")),
        barmode="overlay",
        yaxis2=dict(
            title=dict(text="Flagged Fraud Count", font=dict(color="#ef4444")),
            overlaying="y",
            side="right",
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(color="#ef4444"),
        ),
        xaxis=dict(title="Timeline", gridcolor="#334155"),
        yaxis=dict(title="Total Transactions", gridcolor="#334155"),
    )
    fig.update_layout(layout)
    return fig.to_json()


def create_risk_distribution_chart(scores: List[float], decision_threshold: float = 81.5) -> str:
    """
    Renders risk score histogram with threshold indicator line.
    """
    if not scores or len(scores) < 5:
        # Fallback distribution if DB is cold
        np.random.seed(42)
        normal = np.random.beta(1.5, 10.0, 500) * 80.0
        fraud = np.random.beta(8.0, 2.0, 30) * 30.0 + 70.0
        scores = np.clip(np.concatenate([normal, fraud]), 0.0, 100.0).tolist()

    fig = go.Figure()

    # Histogram of scores
    fig.add_trace(
        go.Histogram(
            x=scores,
            nbinsx=35,
            marker=dict(
                color="#6366f1",
                line=dict(color="#4338ca", width=1),
            ),
            opacity=0.85,
            name="Score Frequency",
            hovertemplate="Risk Range: %{x}<br>Transactions: %{y}<extra></extra>",
        )
    )

    # Threshold Line
    fig.add_vline(
        x=decision_threshold,
        line_width=3,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Decision Threshold ({decision_threshold:.1f})",
        annotation_position="top right",
        annotation_font=dict(color="#ef4444", size=12),
    )

    layout = DARK_THEME_LAYOUT.copy()
    layout.update(
        title=dict(text="<b>Calibrated Risk Score Distribution (0 - 100)</b>", font=dict(size=14, color="#f8fafc")),
        xaxis=dict(title="Calibrated Risk Score", range=[0, 100], gridcolor="#334155"),
        yaxis=dict(title="Frequency", gridcolor="#334155"),
    )
    fig.update_layout(layout)
    return fig.to_json()


def create_top_merchants_chart(merchant_data: List[Dict[str, Any]]) -> str:
    """
    Renders horizontal bar chart of merchants with highest fraud exposure.
    """
    if not merchant_data:
        merchant_names = ["Apex Electronics", "CoinVault Global", "Luxury Horizons", "Skyline Air Direct", "QuickCash Digital"]
        flag_counts = [14, 11, 8, 6, 5]
    else:
        merchant_names = [m["name"] for m in merchant_data][:8]
        flag_counts = [m["flagged_count"] for m in merchant_data][:8]

    # Reverse for top-down descending order in horizontal bar
    merchant_names = list(reversed(merchant_names))
    flag_counts = list(reversed(flag_counts))

    fig = go.Figure(
        go.Bar(
            x=flag_counts,
            y=merchant_names,
            orientation="h",
            marker=dict(
                color=flag_counts,
                colorscale="Reds",
                showscale=False,
            ),
            hovertemplate="Merchant: %{y}<br>Flagged Events: %{x}<extra></extra>",
        )
    )

    layout = DARK_THEME_LAYOUT.copy()
    layout.update(
        title=dict(text="<b>Top High-Risk Merchants (Flagged Ingest)</b>", font=dict(size=14, color="#f8fafc")),
        xaxis=dict(title="Flagged Transactions", gridcolor="#334155"),
        yaxis=dict(title="", tickfont=dict(color="#f8fafc")),
        margin=dict(l=140, r=40, t=50, b=40),
    )
    fig.update_layout(layout)
    return fig.to_json()


def create_shap_bar_chart(feature_names: List[str], shap_values: List[float]) -> str:
    """
    Renders interactive horizontal SHAP explanation bar chart.
    Red represents features pushing toward fraud; Green represents normal features.
    """
    if not feature_names or not shap_values:
        feature_names = ["amount_zscore", "tx_velocity_1h", "dist_from_last_km", "is_night", "hour"]
        shap_values = [0.42, 0.31, 0.18, 0.08, -0.05]

    # Sort by absolute impact
    pairs = sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True)[:10]
    sorted_features = [p[0] for p in reversed(pairs)]
    sorted_vals = [p[1] for p in reversed(pairs)]

    colors = ["#ef4444" if val > 0 else "#10b981" for val in sorted_vals]

    fig = go.Figure(
        go.Bar(
            x=sorted_vals,
            y=sorted_features,
            orientation="h",
            marker=dict(color=colors),
            hovertemplate="Feature: %{y}<br>SHAP Contribution: %{x:+.4f}<extra></extra>",
        )
    )

    layout = DARK_THEME_LAYOUT.copy()
    layout.update(
        title=dict(text="<b>Feature Contributions to Anomaly Score (TreeSHAP)</b>", font=dict(size=14, color="#f8fafc")),
        xaxis=dict(
            title="← Decreases Risk (Normal) | Increases Risk (Fraud) →",
            zeroline=True,
            zerolinecolor="#94a3b8",
            zerolinewidth=2,
            gridcolor="#334155",
        ),
        yaxis=dict(title="", tickfont=dict(color="#f8fafc")),
        margin=dict(l=160, r=40, t=50, b=40),
    )
    fig.update_layout(layout)
    return fig.to_json()
