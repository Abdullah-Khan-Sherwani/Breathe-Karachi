"""
Streamlit dashboard — Breathe Karachi AQI Predictor.
Five tabs; all data from MongoDB.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from config.db import (
    get_collection,
    COLLECTION_FEATURE_STORE,
    COLLECTION_PREDICTIONS,
    COLLECTION_MODEL_REGISTRY,
    COLLECTION_MODEL_LOGS,
    COLLECTION_LIME,
)

# ── Constants ─────────────────────────────────────────────────────────────────

AQI_BANDS = [
    (0,   50,  "Good",                   "#00e400"),
    (51,  100, "Moderate",               "#d4d400"),
    (101, 150, "Unhealthy (Sensitive)",  "#ff7e00"),
    (151, 200, "Unhealthy",              "#ff0000"),
    (201, 300, "Very Unhealthy",         "#8f3f97"),
    (301, 500, "Hazardous",             "#7e0023"),
]

# WHO 24-hour guideline values (µg/m³)
WHO_24H = {
    "PM2_5": 15.0,
    "PM10":  45.0,
    "NO2":   25.0,
    "SO2":   40.0,
    "O3":    100.0,
    "CO":    4000.0,
}

POLLUTANT_LABELS = {
    "PM2_5": "PM₂.₅",
    "PM10":  "PM₁₀",
    "NO2":   "NO₂",
    "SO2":   "SO₂",
    "O3":    "O₃",
    "CO":    "CO",
}

SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]

SEASON_COLS = {
    "Winter": "season_Winter",
    "Spring": "season_Spring",
    "Summer": "season_Summer",
}


def _aqi_band(aqi: float) -> tuple[str, str]:
    for lo, hi, label, color in AQI_BANDS:
        if aqi <= hi:
            return label, color
    return "Hazardous", "#7e0023"


def _season_from_row(row: pd.Series) -> str:
    for season, col in SEASON_COLS.items():
        if col in row.index and row[col] == 1:
            return season
    return "Autumn"


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_feature_store() -> pd.DataFrame:
    docs = list(
        get_collection(COLLECTION_FEATURE_STORE)
        .find({"AQI": {"$exists": True}}, {"_id": 0})
    )
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=3600)
def load_latest_prediction() -> dict | None:
    return get_collection(COLLECTION_PREDICTIONS).find_one(
        {}, sort=[("predicted_at", -1)]
    )


@st.cache_data(ttl=3600)
def load_model_logs() -> pd.DataFrame:
    docs = list(
        get_collection(COLLECTION_MODEL_LOGS)
        .find({}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(100)
    )
    return pd.DataFrame(docs) if docs else pd.DataFrame()


@st.cache_data(ttl=3600)
def load_active_models() -> list[dict]:
    return list(
        get_collection(COLLECTION_MODEL_REGISTRY)
        .find({"status": "active"}, {"model_binary": 0, "scaler_binary": 0})
        .sort("trained_at", -1)
    )


@st.cache_data(ttl=3600)
def load_lime() -> pd.DataFrame | None:
    doc = get_collection(COLLECTION_LIME).find_one(
        {}, sort=[("created_at", -1)]
    )
    if doc and "explanation" in doc:
        return pd.DataFrame(doc["explanation"]), doc.get("model_type", "")
    local = Path(__file__).parent / "lime_explanations" / "lime_explanation.csv"
    if local.exists():
        return pd.read_csv(local), "unknown"
    return None, ""


# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Breathe Karachi",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .forecast-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 14px;
        padding: 1.4rem 1rem;
        text-align: center;
        margin: 0.25rem;
    }
    .metric-chip {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 10px;
        padding: 0.9rem 0.75rem;
        text-align: center;
    }
    .stat-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        height: 100%;
    }
    .aqi-pill {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    h1 { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────

df        = load_feature_store()
pred_doc  = load_latest_prediction()
logs_df   = load_model_logs()
lime_data = load_lime()
lime_df, lime_model_type = lime_data if isinstance(lime_data, tuple) else (None, "")


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("# 🌬️ Breathe Karachi")
st.caption("Air quality monitoring & 4-day forecasts · Karachi, Pakistan (24.86°N, 67.00°E)")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📍 Live Snapshot",
    "📈 AQI Trends",
    "🔬 Pollution Breakdown",
    "💡 Insights",
    "🗂️ Model Logs",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Snapshot
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    if df.empty:
        st.warning("No data in feature_store. Run the data pipeline first.")
        st.stop()

    latest       = df.iloc[-1]
    aqi          = float(latest["AQI"])
    aqi_label, aqi_color = _aqi_band(aqi)

    if aqi > 150:
        st.error(f"⚠️ Air quality is {aqi_label} (AQI {aqi:.0f}). Sensitive groups should avoid outdoor activity.")
    elif aqi > 100:
        st.warning(f"Air quality is {aqi_label} (AQI {aqi:.0f}). Sensitive groups may be affected.")

    col_gauge, col_right = st.columns([1, 1.7], gap="large")

    # ── AQI Gauge ────────────────────────────────────────────────────────────
    with col_gauge:
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=aqi,
            number={"font": {"size": 60, "color": aqi_color}, "suffix": ""},
            gauge={
                "axis": {
                    "range": [0, 300],
                    "tickvals": [0, 50, 100, 150, 200, 300],
                    "ticktext": ["0", "50", "100", "150", "200", "300"],
                    "tickwidth": 1,
                    "tickcolor": "#aaa",
                },
                "bar": {"color": aqi_color, "thickness": 0.22},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,   50],  "color": "rgba(0,228,0,0.12)"},
                    {"range": [51,  100], "color": "rgba(212,212,0,0.12)"},
                    {"range": [101, 150], "color": "rgba(255,126,0,0.15)"},
                    {"range": [151, 200], "color": "rgba(255,0,0,0.15)"},
                    {"range": [201, 300], "color": "rgba(143,63,151,0.18)"},
                ],
                "threshold": {
                    "line": {"color": aqi_color, "width": 3},
                    "thickness": 0.8,
                    "value": aqi,
                },
            },
            title={
                "text": f"<b>{aqi_label}</b><br><span style='font-size:0.75em;opacity:0.6'>"
                        f"{latest['date'].strftime('%d %b %Y')}</span>",
                "font": {"size": 18},
            },
        ))
        gauge_fig.update_layout(
            height=300,
            margin=dict(t=40, b=0, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0",
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

    # ── 4-Day Forecast ───────────────────────────────────────────────────────
    with col_right:
        st.markdown("#### 4-Day Forecast")
        if pred_doc and "forecasts" in pred_doc:
            fc_cols = st.columns(4)
            for i, fc in enumerate(pred_doc["forecasts"][:4]):
                fc_aqi = float(fc["predicted_AQI"])
                fc_label, fc_color = _aqi_band(fc_aqi)
                day_name = pd.Timestamp(fc["date"]).strftime("%a")
                with fc_cols[i]:
                    st.markdown(f"""
                    <div class="forecast-card">
                        <div style="font-size:0.75rem;opacity:0.55;text-transform:uppercase;
                                    letter-spacing:0.06em">{day_name}</div>
                        <div style="font-size:0.85rem;opacity:0.7;margin-bottom:0.5rem">{fc['date']}</div>
                        <div style="font-size:2.4rem;font-weight:700;color:{fc_color};
                                    line-height:1">{fc_aqi:.0f}</div>
                        <div style="font-size:0.72rem;margin-top:0.5rem">
                            <span class="aqi-pill"
                                  style="background:{fc_color}28;color:{fc_color}">{fc_label}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
            pred_time = pred_doc.get("predicted_at")
            if pred_time:
                st.caption(f"Generated {pred_time.strftime('%d %b %Y %H:%M UTC')} · "
                           f"model: {pred_doc.get('model_type', '—')}")
        else:
            st.info("No forecast yet. Run `python src/predict.py` to generate one.")

        # ── Today's Pollutants ───────────────────────────────────────────────
        st.markdown("#### Today's Pollutants")
        poll_cols = st.columns(6)
        for i, (key, label) in enumerate(POLLUTANT_LABELS.items()):
            val = float(latest[key]) if key in latest.index and pd.notna(latest[key]) else None
            limit = WHO_24H[key]
            with poll_cols[i]:
                if val is not None:
                    pct = val / limit * 100
                    bar_color = "#00e400" if pct <= 100 else "#ff0000"
                    st.markdown(f"""
                    <div class="metric-chip">
                        <div style="font-size:0.72rem;opacity:0.6;margin-bottom:0.2rem">{label}</div>
                        <div style="font-size:1.3rem;font-weight:600">{val:.1f}</div>
                        <div style="font-size:0.65rem;opacity:0.5">µg/m³</div>
                        <div style="margin-top:0.4rem;height:3px;border-radius:2px;
                                    background:#333;overflow:hidden">
                            <div style="width:{min(pct,100):.0f}%;height:100%;
                                        background:{bar_color}"></div>
                        </div>
                        <div style="font-size:0.62rem;opacity:0.5;margin-top:0.15rem">
                            {pct:.0f}% WHO</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-chip">
                        <div style="font-size:0.72rem;opacity:0.6">{label}</div>
                        <div style="font-size:1.1rem;opacity:0.3">—</div>
                    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AQI Trends
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    if df.empty:
        st.warning("No data available.")
        st.stop()

    col_ctrl, _ = st.columns([2, 3])
    with col_ctrl:
        date_range = st.selectbox(
            "Time window",
            ["Last 30 days", "Last 90 days", "Last 6 months", "Last year", "All time"],
            index=1,
        )

    window_map = {
        "Last 30 days":   30,
        "Last 90 days":   90,
        "Last 6 months":  180,
        "Last year":      365,
        "All time":       99999,
    }
    cutoff = df["date"].max() - pd.Timedelta(days=window_map[date_range])
    view   = df[df["date"] >= cutoff].copy()

    # 7-day rolling average
    view["AQI_7d"] = view["AQI"].rolling(7, min_periods=1).mean()

    fig_trend = go.Figure()

    # EPA AQI band shading
    band_annotations = []
    for lo, hi, label, color in AQI_BANDS:
        fig_trend.add_hrect(
            y0=lo, y1=min(hi, 320),
            fillcolor=color, opacity=0.06,
            line_width=0,
        )
        if lo < 300:
            band_annotations.append(
                go.layout.Annotation(
                    x=1.01, xref="paper",
                    y=(lo + min(hi, 300)) / 2, yref="y",
                    text=label, showarrow=False,
                    font=dict(size=9, color=color),
                    xanchor="left",
                )
            )

    # AQI line
    fig_trend.add_trace(go.Scatter(
        x=view["date"], y=view["AQI"],
        mode="lines",
        name="Daily AQI",
        line=dict(color="#5b9bd5", width=1.2),
        opacity=0.6,
    ))

    # 7-day rolling average
    fig_trend.add_trace(go.Scatter(
        x=view["date"], y=view["AQI_7d"],
        mode="lines",
        name="7-day avg",
        line=dict(color="#f5a623", width=2.2),
    ))

    # Forecast overlay
    if pred_doc and "forecasts" in pred_doc:
        fc_dates = [pd.Timestamp(f["date"]) for f in pred_doc["forecasts"]]
        fc_aqis  = [float(f["predicted_AQI"]) for f in pred_doc["forecasts"]]
        fig_trend.add_trace(go.Scatter(
            x=fc_dates, y=fc_aqis,
            mode="markers+lines",
            name="4-day forecast",
            line=dict(color="#e74c3c", width=2, dash="dot"),
            marker=dict(size=9, color="#e74c3c"),
        ))

    fig_trend.update_layout(
        title=f"AQI — {date_range}",
        xaxis_title=None,
        yaxis_title="US AQI",
        yaxis_range=[0, max(320, view["AQI"].max() + 20)],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        annotations=band_annotations,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#d0d0d0",
        height=420,
        margin=dict(t=60, b=40, l=60, r=120),
    )
    fig_trend.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
    fig_trend.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")

    st.plotly_chart(fig_trend, use_container_width=True)

    # Distribution of AQI categories
    st.markdown("#### AQI Category Distribution")
    view_copy = view.copy()
    view_copy["category"] = view_copy["AQI"].apply(lambda v: _aqi_band(v)[0])
    cat_counts = view_copy["category"].value_counts().reset_index()
    cat_counts.columns = ["category", "days"]
    cat_order  = [b[2] for b in AQI_BANDS]
    cat_colors = {b[2]: b[3] for b in AQI_BANDS}

    fig_bar = px.bar(
        cat_counts,
        x="category", y="days",
        color="category",
        color_discrete_map=cat_colors,
        category_orders={"category": cat_order},
        labels={"days": "Days", "category": ""},
        text="days",
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        showlegend=False, height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#d0d0d0",
        margin=dict(t=20, b=40, l=40, r=20),
    )
    fig_bar.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
    st.plotly_chart(fig_bar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Pollution Breakdown
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    if df.empty:
        st.warning("No data available.")
        st.stop()

    latest = df.iloc[-1]

    col_radar, col_pie = st.columns(2, gap="large")

    # ── Radar: today's pollutants vs WHO 24h limits ──────────────────────────
    with col_radar:
        st.markdown("#### Today vs WHO 24-Hour Limits")
        poll_keys   = list(WHO_24H.keys())
        poll_labels = [POLLUTANT_LABELS[k] for k in poll_keys]
        limits      = [WHO_24H[k] for k in poll_keys]
        values      = [
            float(latest[k]) if k in latest.index and pd.notna(latest[k]) else 0.0
            for k in poll_keys
        ]
        pct_values  = [v / l * 100 for v, l in zip(values, limits)]
        who_line    = [100.0] * len(poll_keys)

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=pct_values + [pct_values[0]],
            theta=poll_labels + [poll_labels[0]],
            fill="toself",
            name="Today",
            line_color="#5b9bd5",
            fillcolor="rgba(91,155,213,0.2)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=who_line + [who_line[0]],
            theta=poll_labels + [poll_labels[0]],
            name="WHO limit",
            line=dict(color="#e74c3c", dash="dash", width=2),
            fill=None,
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max(max(pct_values) * 1.15, 120)],
                                ticksuffix="%", gridcolor="rgba(255,255,255,0.1)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#d0d0d0",
            height=380,
            margin=dict(t=20, b=60, l=30, r=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Pie: share of total WHO-limit-weighted pollution ─────────────────────
    with col_pie:
        st.markdown("#### Pollution Load by Pollutant")
        st.caption("Each pollutant's share of the total WHO-limit-normalised burden")
        nonzero = [(POLLUTANT_LABELS[k], pct)
                   for k, pct in zip(poll_keys, pct_values) if pct > 0]
        if nonzero:
            pie_labels, pie_vals = zip(*nonzero)
            fig_pie = go.Figure(go.Pie(
                labels=pie_labels,
                values=pie_vals,
                hole=0.45,
                textinfo="label+percent",
                marker=dict(colors=["#5b9bd5", "#f5a623", "#2ecc71",
                                    "#e74c3c", "#9b59b6", "#1abc9c"]),
            ))
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#d0d0d0",
                height=380,
                margin=dict(t=20, b=20, l=20, r=20),
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ── LIME Explanation ─────────────────────────────────────────────────────
    st.markdown("#### LIME Feature Importance (AQI t+1 prediction)")
    if lime_df is not None and not lime_df.empty:
        st.caption(f"Model: {lime_model_type} · Top features driving tomorrow's AQI forecast")
        lime_plot = lime_df.copy()
        lime_plot = lime_plot.sort_values("weight", ascending=True)
        lime_plot["color"] = lime_plot["weight"].apply(
            lambda w: "#e74c3c" if w < 0 else "#2ecc71"
        )

        fig_lime = go.Figure(go.Bar(
            x=lime_plot["weight"],
            y=lime_plot["feature"],
            orientation="h",
            marker_color=lime_plot["color"].tolist(),
        ))
        fig_lime.add_vline(x=0, line_width=1, line_color="rgba(255,255,255,0.3)")
        fig_lime.update_layout(
            xaxis_title="LIME weight (impact on predicted AQI t+1)",
            yaxis_title=None,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#d0d0d0",
            height=420,
            margin=dict(t=20, b=40, l=20, r=20),
        )
        fig_lime.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
        fig_lime.update_yaxes(showgrid=False)
        st.plotly_chart(fig_lime, use_container_width=True)
    else:
        st.info("LIME explanation not yet generated. Run `python src/create_lime.py` to build it.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Insights
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    if df.empty:
        st.warning("No data available.")
        st.stop()

    # ── Compute insight metrics ──────────────────────────────────────────────
    df_ins = df.copy()

    # WHO exceedance (AQI > 100 = unhealthy for sensitive groups)
    who_exc_pct = (df_ins["AQI"] > 100).mean() * 100

    # Overall stats
    avg_aqi  = df_ins["AQI"].mean()
    max_aqi  = df_ins["AQI"].max()
    max_date = df_ins.loc[df_ins["AQI"].idxmax(), "date"].strftime("%d %b %Y")
    min_aqi  = df_ins["AQI"].min()
    n_days   = len(df_ins)

    # Season
    df_ins["season"] = df_ins.apply(_season_from_row, axis=1)
    season_avg = df_ins.groupby("season")["AQI"].mean()
    worst_season = season_avg.idxmax() if not season_avg.empty else "—"
    best_season  = season_avg.idxmin() if not season_avg.empty else "—"

    # Day of week
    df_ins["weekday"] = pd.to_datetime(df_ins["date"]).dt.day_name()
    dow_avg = df_ins.groupby("weekday")["AQI"].mean()
    worst_day = dow_avg.idxmax() if not dow_avg.empty else "—"
    best_day  = dow_avg.idxmin() if not dow_avg.empty else "—"

    # ── Stat cards row ───────────────────────────────────────────────────────
    st.markdown("#### Summary Statistics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="stat-card">
            <div style="font-size:0.8rem;opacity:0.55;margin-bottom:0.3rem">Total Days</div>
            <div style="font-size:2rem;font-weight:700">{n_days:,}</div>
            <div style="font-size:0.75rem;opacity:0.5">in dataset</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        cat, col = _aqi_band(avg_aqi)
        st.markdown(f"""<div class="stat-card">
            <div style="font-size:0.8rem;opacity:0.55;margin-bottom:0.3rem">Mean AQI</div>
            <div style="font-size:2rem;font-weight:700;color:{col}">{avg_aqi:.0f}</div>
            <div style="font-size:0.75rem;opacity:0.5">{cat}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        cat_max, col_max = _aqi_band(max_aqi)
        st.markdown(f"""<div class="stat-card">
            <div style="font-size:0.8rem;opacity:0.55;margin-bottom:0.3rem">Peak AQI</div>
            <div style="font-size:2rem;font-weight:700;color:{col_max}">{max_aqi:.0f}</div>
            <div style="font-size:0.75rem;opacity:0.5">{max_date}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        exc_col = "#ff7e00" if who_exc_pct > 50 else ("#d4d400" if who_exc_pct > 25 else "#00e400")
        st.markdown(f"""<div class="stat-card">
            <div style="font-size:0.8rem;opacity:0.55;margin-bottom:0.3rem">WHO Exceedance</div>
            <div style="font-size:2rem;font-weight:700;color:{exc_col}">{who_exc_pct:.0f}%</div>
            <div style="font-size:0.75rem;opacity:0.5">days AQI &gt; 100</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown(f"""<div class="stat-card">
            <div style="font-size:0.8rem;opacity:0.55;margin-bottom:0.3rem">Worst Season</div>
            <div style="font-size:1.7rem;font-weight:700">{worst_season}</div>
            <div style="font-size:0.75rem;opacity:0.5">
                avg AQI {season_avg.get(worst_season, 0):.0f} · Best: {best_season}
                ({season_avg.get(best_season, 0):.0f})</div>
        </div>""", unsafe_allow_html=True)
    with c6:
        st.markdown(f"""<div class="stat-card">
            <div style="font-size:0.8rem;opacity:0.55;margin-bottom:0.3rem">Worst Day of Week</div>
            <div style="font-size:1.7rem;font-weight:700">{worst_day}</div>
            <div style="font-size:0.75rem;opacity:0.5">
                avg AQI {dow_avg.get(worst_day, 0):.0f} · Best: {best_day}
                ({dow_avg.get(best_day, 0):.0f})</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Seasonal AQI bar chart ───────────────────────────────────────────────
    col_s, col_d = st.columns(2, gap="large")

    with col_s:
        st.markdown("#### Average AQI by Season")
        if not season_avg.empty:
            s_df = season_avg.reset_index().rename(columns={"AQI": "avg_AQI"})
            s_colors = [_aqi_band(v)[1] for v in s_df["avg_AQI"]]
            fig_s = px.bar(s_df, x="season", y="avg_AQI",
                           color="season",
                           color_discrete_sequence=s_colors,
                           labels={"avg_AQI": "Avg AQI", "season": ""},
                           text=s_df["avg_AQI"].map("{:.0f}".format))
            fig_s.update_traces(textposition="outside", showlegend=False)
            fig_s.update_layout(
                height=300, showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#d0d0d0",
                margin=dict(t=10, b=40, l=40, r=10),
            )
            fig_s.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
            st.plotly_chart(fig_s, use_container_width=True)

    with col_d:
        st.markdown("#### Average AQI by Day of Week")
        if not dow_avg.empty:
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                         "Saturday", "Sunday"]
            d_df = dow_avg.reindex(day_order).dropna().reset_index()
            d_df.columns = ["weekday", "avg_AQI"]
            d_colors = [_aqi_band(v)[1] for v in d_df["avg_AQI"]]
            fig_d = px.bar(d_df, x="weekday", y="avg_AQI",
                           color="weekday",
                           color_discrete_sequence=d_colors,
                           labels={"avg_AQI": "Avg AQI", "weekday": ""},
                           text=d_df["avg_AQI"].map("{:.0f}".format))
            fig_d.update_traces(textposition="outside", showlegend=False)
            fig_d.update_layout(
                height=300, showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#d0d0d0",
                margin=dict(t=10, b=40, l=40, r=10),
            )
            fig_d.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
            st.plotly_chart(fig_d, use_container_width=True)

    # ── Monthly trend heatmap ────────────────────────────────────────────────
    st.markdown("#### Monthly AQI Heatmap")
    df_ins["year"]  = df_ins["date"].dt.year
    df_ins["month"] = df_ins["date"].dt.month
    pivot = df_ins.pivot_table(values="AQI", index="year", columns="month", aggfunc="mean")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = [month_names[m - 1] for m in pivot.columns]

    fig_heat = px.imshow(
        pivot,
        color_continuous_scale=[
            [0.0,  "#00e400"],
            [0.17, "#d4d400"],
            [0.33, "#ff7e00"],
            [0.5,  "#ff0000"],
            [0.67, "#8f3f97"],
            [1.0,  "#7e0023"],
        ],
        zmin=0, zmax=300,
        aspect="auto",
        labels=dict(color="AQI"),
        text_auto=".0f",
    )
    fig_heat.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#d0d0d0",
        margin=dict(t=10, b=40, l=60, r=20),
        coloraxis_colorbar=dict(title="AQI", tickvals=[0,50,100,150,200,300]),
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Model Logs
# ═══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("#### Training Run History")

    active_models = load_active_models()
    if active_models:
        st.markdown("**Active models in registry:**")
        active_cols = st.columns(min(len(active_models), 3))
        for i, m in enumerate(active_models[:3]):
            with active_cols[i]:
                mtype     = m.get("model_type", "—")
                rmse      = m.get("RMSE", None)
                r2        = m.get("R2",   None)
                ver       = m.get("version", "—")
                rmse_str  = f"{rmse:.2f}" if rmse is not None else "—"
                r2_str    = f"{r2:.3f}"   if r2   is not None else "—"
                st.markdown(f"""<div class="stat-card">
                    <div style="font-size:0.8rem;opacity:0.55">Active model</div>
                    <div style="font-size:1.4rem;font-weight:700;text-transform:uppercase;
                                letter-spacing:0.05em">{mtype}</div>
                    <div style="font-size:0.8rem;opacity:0.7;margin-top:0.3rem">
                        RMSE: {rmse_str} &nbsp;&middot;&nbsp; R&sup2;: {r2_str}</div>
                    <div style="font-size:0.72rem;opacity:0.45;margin-top:0.2rem">{ver}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        for m in active_models[:2]:
            horizon_rows = []
            for d in range(1, 5):
                if f"MAE_d{d}" in m:
                    horizon_rows.append({
                        "Horizon": f"Day {d} (t+{d})",
                        "MAE": round(m[f"MAE_d{d}"], 2),
                        "RMSE": round(m[f"RMSE_d{d}"], 2),
                        "R²": round(m[f"R2_d{d}"], 3),
                    })
            if horizon_rows:
                st.caption(f"Per-horizon metrics — {m.get('model_type','').upper()} v{m.get('version','')}")
                st.dataframe(pd.DataFrame(horizon_rows), hide_index=True, use_container_width=True)

    if logs_df.empty:
        st.info("No training logs yet. Run `python src/train.py` first.")
    else:
        # Metric trend chart
        success_logs = logs_df[logs_df.get("status", pd.Series(dtype=str)) == "success"].copy() \
            if "status" in logs_df.columns else logs_df.copy()

        if "timestamp" in success_logs.columns and "RMSE" in success_logs.columns:
            success_logs["timestamp"] = pd.to_datetime(success_logs["timestamp"])
            success_logs = success_logs.sort_values("timestamp")

            fig_log = go.Figure()
            model_types = success_logs["model_type"].unique() if "model_type" in success_logs.columns else []
            colors_map  = {"ridge": "#5b9bd5", "lgbm": "#f5a623", "lstm": "#2ecc71"}
            for mt in model_types:
                sub = success_logs[success_logs["model_type"] == mt]
                fig_log.add_trace(go.Scatter(
                    x=sub["timestamp"], y=sub["RMSE"],
                    mode="lines+markers",
                    name=mt.upper(),
                    line=dict(color=colors_map.get(mt, "#aaa"), width=2),
                    marker=dict(size=7),
                ))

            fig_log.update_layout(
                title="RMSE over Training Runs",
                xaxis_title=None,
                yaxis_title="RMSE",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#d0d0d0",
                height=300,
                margin=dict(t=50, b=40, l=60, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            fig_log.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
            fig_log.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
            st.plotly_chart(fig_log, use_container_width=True)

        # Log table
        st.markdown("#### Run Details")
        display_cols = [c for c in ["timestamp", "model_type", "status", "MAE", "RMSE", "R2", "error"]
                        if c in logs_df.columns]
        disp = logs_df[display_cols].copy()
        if "timestamp" in disp.columns:
            disp["timestamp"] = pd.to_datetime(disp["timestamp"]).dt.strftime("%Y-%m-%d %H:%M UTC")
        for col in ["MAE", "RMSE", "R2"]:
            if col in disp.columns:
                disp[col] = disp[col].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
        st.dataframe(disp, use_container_width=True, height=400)
