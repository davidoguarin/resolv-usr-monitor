"""
Resolv USR Strategy Monitor — Streamlit Dashboard
==================================================
Simulates a $50K allocation to Resolv USR / stUSR over 6 months
(Oct 1 2025 → Apr 1 2026) and tracks protocol risk metrics.

Run locally:
    streamlit run app.py

Data:
    Pre-fetched CSVs in data/cached/ (run data/fetch_dashboard_data.py first).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resolv USR Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
  html, body, [class*="css"] { font-family: 'Inter', 'Helvetica Neue', sans-serif; }
  .block-container { padding-top: 2rem; padding-bottom: 2rem; }

  .kpi-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 4px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  .kpi-label {
    font-size: 0.70rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 6px;
  }
  .kpi-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: #111827;
    line-height: 1.15;
  }
  .kpi-delta-pos { font-size: 0.78rem; color: #16a34a; font-weight: 600; }
  .kpi-delta-neg { font-size: 0.78rem; color: #dc2626; font-weight: 600; }
  .kpi-delta-neu { font-size: 0.78rem; color: #6b7280; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Constants ────────────────────────────────────────────────────────────────
INITIAL_INVESTMENT = 50_000.0
DEPEG_THRESHOLD    = 0.005          # ±0.5%
EXPLOIT_DATE       = pd.Timestamp("2026-03-22", tz="UTC")
DATA_END           = pd.Timestamp("2026-03-24 23:59", tz="UTC")

C_USR    = "#ea580c"
C_CURVE  = "#7c3aed"
C_NAV    = "#16a34a"
C_TVL    = "#2563eb"
C_MINT   = "#059669"
C_BURN   = "#dc2626"
C_THRESH = "#f59e0b"
C_EXPL   = "#6366f1"

DATA_DIR = Path("data/cached")

WINDOWS = {
    "1 Week":  7,    # Mar 17–24
    "3 Weeks": 21,   # Mar  3–24
}


# ─── Data loading ─────────────────────────────────────────────────────────────

def _read_daily(name: str) -> pd.DataFrame:
    p = DATA_DIR / name
    if not p.exists():
        st.error(f"Missing {name}. Run: python data/fetch_dashboard_data.py")
        st.stop()
    df = pd.read_csv(p, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _read_hourly(name: str, dt_col: str = "datetime") -> pd.DataFrame:
    p = DATA_DIR / name
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=[dt_col])
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True)
    return df.dropna(subset=[dt_col]).sort_values(dt_col).reset_index(drop=True)


@st.cache_data
def load_all() -> dict:
    return {
        "tvl":    _read_daily("resolv_tvl.csv"),
        "apy":    _read_daily("usr_apy.csv"),
        "price":  _read_daily("usr_price_daily.csv"),
        "mb":     _read_daily("usr_mint_burn.csv"),
        "h_v3":   _read_hourly("usr_price_hourly_v3.csv"),
        "h_cu":   _read_hourly("usr_price_hourly_curve.csv"),
        "mb_h":   _read_hourly("usr_mint_burn_hourly.csv"),
    }


@st.cache_data
def compute_nav(price_df: pd.DataFrame, apy_df: pd.DataFrame) -> pd.DataFrame:
    full = pd.DataFrame({"date": pd.date_range(
        price_df["date"].min(), price_df["date"].max(), freq="D", tz="UTC"
    )})
    pf = full.merge(price_df[["date", "close"]], on="date", how="left")
    pf["close"] = pf["close"].ffill().bfill()
    pf = pf.merge(apy_df[["date", "apy"]], on="date", how="left")
    pf["apy"] = pf["apy"].ffill().bfill().fillna(5.0)

    p0 = float(pf["close"].iloc[0]) or 1.0
    tokens = INITIAL_INVESTMENT / p0
    navs = []
    for _, row in pf.iterrows():
        tokens *= 1.0 + row["apy"] / 100.0 / 365.0
        navs.append(tokens * row["close"])

    df = pf[["date"]].copy()
    df["nav"] = navs
    df["drawdown"] = (df["nav"] / df["nav"].cummax() - 1.0) * 100.0
    return df


# ─── Plotly layout base ───────────────────────────────────────────────────────

def _base_layout(title: str, height: int = 300, yformat: str | None = None) -> dict:
    """Return a Plotly layout dict with legend below the plot (no title overlap)."""
    layout = dict(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        height=height,
        margin=dict(l=8, r=8, t=42, b=72),   # generous bottom for legend
        hovermode="x unified",
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=13, color="#111827"),
            x=0, xanchor="left", y=1, yanchor="top",
            pad=dict(t=4, b=0),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,          # below x-axis — no overlap with title
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=10.5),
        ),
        font=dict(family="Inter, Helvetica Neue, sans-serif", size=11, color="#374151"),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#f3f4f6", zeroline=False, tickfont=dict(size=10)),
    )
    if yformat:
        layout["yaxis"]["tickformat"] = yformat
    return layout


def _exploit_vline(fig: go.Figure, label: bool = True) -> None:
    fig.add_vline(
        x=EXPLOIT_DATE.timestamp() * 1000,
        line_dash="dot", line_color=C_EXPL, line_width=1.3,
        annotation_text="  USR exploit" if label else None,
        annotation_font_size=9, annotation_font_color=C_EXPL,
        annotation_position="top right",
    )


# ─── Charts ───────────────────────────────────────────────────────────────────

def chart_nav(nav_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nav_df["date"], y=nav_df["nav"],
        name="Portfolio NAV",
        line=dict(color=C_NAV, width=2),
        fill="tozeroy", fillcolor="rgba(22,163,74,0.08)",
        hovertemplate="<b>NAV: $%{y:,.0f}</b><extra></extra>",
    ))
    fig.add_hline(y=INITIAL_INVESTMENT, line_dash="dash", line_color="#9ca3af",
                  line_width=1.2, annotation_text="$50K entry",
                  annotation_font_size=9, annotation_font_color="#9ca3af",
                  annotation_position="top right")
    _exploit_vline(fig)
    fig.update_layout(**_base_layout(
        "Portfolio NAV — $50K initial · stUSR daily compounding", height=290))
    fig.update_yaxes(tickprefix="$")
    return fig


def chart_tvl(tvl_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tvl_df["date"], y=tvl_df["tvl_usd"] / 1e6,
        name="Resolv TVL",
        line=dict(color=C_TVL, width=2),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
        hovertemplate="<b>TVL: $%{y:.1f}M</b><extra></extra>",
    ))
    _exploit_vline(fig, label=False)
    fig.update_layout(**_base_layout("Resolv Protocol TVL", height=250))
    fig.update_yaxes(ticksuffix="M", tickprefix="$")
    return fig


def chart_depeg(h_v3: pd.DataFrame, h_cu: pd.DataFrame, days: int) -> go.Figure:
    upper = 1.0 + DEPEG_THRESHOLD
    lower = 1.0 - DEPEG_THRESHOLD
    cutoff = DATA_END - pd.Timedelta(days=days)

    def _trim(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return df[df["datetime"] >= cutoff].copy()

    v3 = _trim(h_v3)
    cu = _trim(h_cu)

    fig = go.Figure()

    if not v3.empty:
        fig.add_trace(go.Scatter(
            x=v3["datetime"], y=v3["close"],
            name="USR/USDC — Uniswap V3",
            line=dict(color=C_USR, width=1.8),
            hovertemplate="<b>V3: $%{y:.4f}</b><extra></extra>",
        ))

    if not cu.empty:
        fig.add_trace(go.Scatter(
            x=cu["datetime"], y=cu["close"],
            name="USR/USDC — Curve",
            line=dict(color=C_CURVE, width=1.4, dash="dot"),
            hovertemplate="<b>Curve: $%{y:.4f}</b><extra></extra>",
        ))

    fig.add_hline(y=upper, line_dash="dash", line_color=C_THRESH, line_width=1.2,
                  annotation_text=f"+0.5%  ({upper:.3f})",
                  annotation_font_size=9, annotation_font_color=C_THRESH,
                  annotation_position="top right")
    fig.add_hline(y=lower, line_dash="dash", line_color=C_THRESH, line_width=1.2,
                  annotation_text=f"−0.5%  ({lower:.3f})",
                  annotation_font_size=9, annotation_font_color=C_THRESH,
                  annotation_position="bottom right")
    fig.add_hline(y=1.0, line_dash="dot", line_color="#9ca3af", line_width=0.7)

    _exploit_vline(fig)
    fig.update_layout(**_base_layout(
        "USR / USDC Price — ±0.5% withdrawal trigger bands (hourly)", height=300))
    fig.update_yaxes(tickprefix="$")
    return fig


def chart_liquidity(h_v3: pd.DataFrame, h_cu: pd.DataFrame, days: int) -> go.Figure:
    cutoff = DATA_END - pd.Timedelta(days=days)

    def _trim_roll(df: pd.DataFrame, label: str) -> tuple:
        if df.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        d = df[df["datetime"] >= cutoff].set_index("datetime").sort_index()
        roll = d["volume_usd"].fillna(0).rolling("24h", min_periods=1).sum() / 1e3
        return roll.index, roll.values

    x_v3, y_v3 = _trim_roll(h_v3, "v3")
    x_cu, y_cu = _trim_roll(h_cu, "curve")

    fig = go.Figure()
    if len(x_v3):
        fig.add_trace(go.Scatter(
            x=x_v3, y=y_v3,
            name="Uniswap V3 — 24h rolling vol.",
            line=dict(color=C_USR, width=1.8),
            fill="tozeroy", fillcolor="rgba(234,88,12,0.10)",
            hovertemplate="<b>V3: $%{y:.1f}K (24h)</b><extra></extra>",
        ))
    if len(x_cu):
        fig.add_trace(go.Scatter(
            x=x_cu, y=y_cu,
            name="Curve — 24h rolling vol.",
            line=dict(color=C_CURVE, width=1.4, dash="dot"),
            fill="tozeroy", fillcolor="rgba(124,58,237,0.06)",
            hovertemplate="<b>Curve: $%{y:.1f}K (24h)</b><extra></extra>",
        ))
    _exploit_vline(fig, label=False)
    fig.update_layout(**_base_layout(
        "USR Pool Liquidity — 24h Rolling Volume (hourly, both pools)", height=260))
    fig.update_yaxes(ticksuffix="K", tickprefix="$")
    return fig


def chart_mint_burn(mb_hourly: pd.DataFrame, days: int) -> go.Figure:
    cutoff = DATA_END - pd.Timedelta(days=days)
    df = mb_hourly[mb_hourly["datetime"] >= cutoff].copy()

    if df.empty:
        fig = go.Figure()
        fig.update_layout(**_base_layout("USR Mint / Burn — Hourly", height=280))
        return fig

    # Threshold based on negative (outflow) hours only — ignores large mints
    neg_flows = df["net"][df["net"] < 0]
    if len(neg_flows) > 3:
        threshold = float(neg_flows.mean() - 1.5 * neg_flows.std())
        threshold = min(threshold, -500_000)   # floor: at least −500K USR/h
    else:
        threshold = -500_000

    # Find first hour where net < threshold (withdraw signal)
    breach = df[df["net"] < threshold]
    withdraw_dt = breach["datetime"].iloc[0] if not breach.empty else None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["datetime"], y=df["minted"] / 1e6,
        name="Minted (USR)", marker_color=C_MINT, opacity=0.75,
        hovertemplate="<b>%{x|%b %d %H:%M UTC}<br>Minted: %{y:.2f}M USR</b><extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=df["datetime"], y=-df["burned"] / 1e6,
        name="Burned (USR)", marker_color=C_BURN, opacity=0.75,
        hovertemplate="<b>%{x|%b %d %H:%M UTC}<br>Burned: %{y:.2f}M USR</b><extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["net"] / 1e6,
        name="Net flow",
        line=dict(color="#111827", width=1.5),
        hovertemplate="<b>%{x|%b %d %H:%M UTC}<br>Net: %{y:.2f}M USR</b><extra></extra>",
    ))

    # Threshold line
    fig.add_hline(
        y=threshold / 1e6,
        line_dash="dash", line_color=C_THRESH, line_width=1.3,
        annotation_text=f"Withdrawal trigger ({threshold/1e6:.1f}M USR/h)",
        annotation_font_size=9, annotation_font_color=C_THRESH,
        annotation_position="bottom right",
    )

    # "Withdraw funds" vertical marker at first breach
    if withdraw_dt is not None:
        fig.add_vline(
            x=withdraw_dt.timestamp() * 1000,
            line_color="#dc2626", line_width=2, line_dash="solid",
        )
        fig.add_annotation(
            x=withdraw_dt.timestamp() * 1000,
            y=0.97, yref="paper",
            text="⚠ Withdraw funds",
            showarrow=False,
            font=dict(size=10, color="#dc2626", family="Inter, sans-serif"),
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#dc2626", borderwidth=1,
            xanchor="left", yanchor="top",
            xshift=6,
        )

    _exploit_vline(fig, label=False)
    fig.update_layout(
        **_base_layout(
            "USR Mint / Burn — Hourly Net Flow (Etherscan Transfer Events)", height=300),
        barmode="overlay",
    )
    fig.update_yaxes(ticksuffix="M USR")
    return fig


# ─── KPI helper ───────────────────────────────────────────────────────────────

def kpi_card(label: str, value: str, delta: str = "", dtype: str = "neu") -> str:
    d = f'<div class="kpi-delta-{dtype}">{delta}</div>' if delta else ""
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{d}'
        f'</div>'
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Header ──────────────────────────────────────────────────────────────
    st.title("Resolv USR Strategy Monitor")
    st.caption("$50K allocation · stUSR yield compounding · Oct 2025 – Apr 2026")

    # ── Load & compute ───────────────────────────────────────────────────────
    data   = load_all()
    nav_df = compute_nav(data["price"], data["apy"])

    # ── Window selector ──────────────────────────────────────────────────────
    st.markdown("**Display window** (hourly resolution for price & liquidity)")
    window = st.radio(
        "Display window",
        list(WINDOWS.keys()),
        index=0,           # default: 1 Week
        horizontal=True,
        label_visibility="collapsed",
    )
    days = WINDOWS[window]

    st.markdown("---")

    # ── KPI cards ────────────────────────────────────────────────────────────
    cutoff_daily = DATA_END - pd.Timedelta(days=days)
    nav_w   = nav_df[nav_df["date"] >= cutoff_daily]
    price_w = data["price"][data["price"]["date"] >= cutoff_daily]
    apy_w   = data["apy"][data["apy"]["date"]   >= cutoff_daily]
    tvl_w   = data["tvl"][data["tvl"]["date"]   >= cutoff_daily]

    cur_nav   = float(nav_w["nav"].iloc[-1])          if len(nav_w) else INITIAL_INVESTMENT
    peak_nav  = float(nav_df["nav"].max())             # lifetime peak, not window
    max_dd    = float(nav_w["drawdown"].min())         if len(nav_w) else 0.0
    min_price = float(price_w["close"].min())          if len(price_w) else 1.0
    avg_apy   = float(apy_w["apy"].mean())             if len(apy_w) else 0.0
    cur_tvl   = float(tvl_w["tvl_usd"].iloc[-1]) / 1e6 if len(tvl_w) else 0.0

    pnl     = cur_nav - INITIAL_INVESTMENT
    pnl_pct = pnl / INITIAL_INVESTMENT * 100

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(kpi_card(
            "Current NAV", f"${cur_nav:,.0f}",
            f"{'▲' if pnl>=0 else '▼'} ${abs(pnl):,.0f} ({pnl_pct:+.1f}%)",
            "pos" if pnl >= 0 else "neg",
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Peak NAV", f"${peak_nav:,.0f}"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card(
            "Max Drawdown", f"{max_dd:.1f}%", "from peak",
            "neg" if max_dd < -5 else "neu",
        ), unsafe_allow_html=True)
    with c4:
        depegged = min_price < 0.995
        st.markdown(kpi_card(
            "Min USR Price", f"${min_price:.4f}",
            "⚠ DEPEGGED" if depegged else "✓ In range",
            "neg" if depegged else "pos",
        ), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi_card(
            "Avg APY (stUSR)", f"{avg_apy:.1f}%", "Annualised yield",
        ), unsafe_allow_html=True)
    with c6:
        st.markdown(kpi_card(
            "Latest TVL", f"${cur_tvl:.1f}M", "Resolv protocol",
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    # NAV — daily granularity across full history always; window filters display
    st.plotly_chart(chart_nav(nav_w), use_container_width=True,
                    config={"displayModeBar": False})

    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(chart_tvl(tvl_w), use_container_width=True,
                        config={"displayModeBar": False})
    with col_r:
        st.plotly_chart(chart_liquidity(data["h_v3"], data["h_cu"], days),
                        use_container_width=True, config={"displayModeBar": False})

    st.plotly_chart(chart_depeg(data["h_v3"], data["h_cu"], days),
                    use_container_width=True, config={"displayModeBar": False})

    st.plotly_chart(chart_mint_burn(data["mb_h"], days),
                    use_container_width=True, config={"displayModeBar": False})

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "Data: DeFiLlama (TVL · stUSR APY) · GeckoTerminal (USR/USDC hourly OHLCV — "
        "Uniswap V3 & Curve, Mar 3–24 2026) · Etherscan (USR ERC-20 Transfer events, hourly) | "
        "Liquidity = 24h rolling volume (both pools) | "
        "NAV = $50K ÷ USR price₀ → tokens compounded daily at stUSR APY × price"
    )


if __name__ == "__main__":
    main()
