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
from datetime import datetime, timezone
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
  /* Global font & background */
  html, body, [class*="css"]  { font-family: 'Inter', 'Helvetica Neue', sans-serif; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* KPI card */
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

  /* Section header */
  .section-header {
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #374151;
    border-bottom: 2px solid #f3f4f6;
    padding-bottom: 6px;
    margin-top: 10px;
    margin-bottom: 2px;
  }

  /* Divider */
  hr { border: none; border-top: 1px solid #f3f4f6; margin: 8px 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Constants ────────────────────────────────────────────────────────────────
INITIAL_INVESTMENT = 50_000.0
DEPEG_THRESHOLD    = 0.005      # ±0.5%
EXPLOIT_DATE       = pd.Timestamp("2026-03-22", tz="UTC")

C_USR        = "#ea580c"   # orange
C_NAV        = "#16a34a"   # green
C_TVL        = "#2563eb"   # blue
C_MINT       = "#059669"   # emerald
C_BURN       = "#dc2626"   # red
C_THRESHOLD  = "#f59e0b"   # amber
C_EXPLOIT    = "#7c3aed"   # violet
C_BAND       = "rgba(234,88,12,0.10)"

DATA_DIR = Path("data/cached")


# ─── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_all() -> dict[str, pd.DataFrame]:
    def _read(name: str) -> pd.DataFrame:
        p = DATA_DIR / name
        if not p.exists():
            st.error(f"Missing {name}. Run: python data/fetch_dashboard_data.py")
            st.stop()
        df = pd.read_csv(p, parse_dates=["date"])
        # Normalise to UTC — some CSVs are already tz-aware, some are naive
        col = pd.to_datetime(df["date"], utc=True)
        df["date"] = col.dt.normalize()  # floor to midnight UTC
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df

    tvl   = _read("resolv_tvl.csv")
    apy   = _read("usr_apy.csv")
    price = _read("usr_price_daily.csv")
    mb    = _read("usr_mint_burn.csv")
    return {"tvl": tvl, "apy": apy, "price": price, "mb": mb}


@st.cache_data
def compute_nav(price_df: pd.DataFrame, apy_df: pd.DataFrame) -> pd.DataFrame:
    """Simulate $50K stUSR portfolio: tokens compound daily with APY; NAV = tokens × price."""
    # Merge on date — price may have gaps; forward-fill price
    full_dates = pd.DataFrame({"date": pd.date_range(
        price_df["date"].min(), price_df["date"].max(), freq="D", tz="UTC"
    )})
    price_full = full_dates.merge(price_df[["date", "close", "volume_usd"]], on="date", how="left")
    price_full["close"] = price_full["close"].ffill().bfill()
    price_full = price_full.merge(apy_df[["date", "apy"]], on="date", how="left")
    price_full["apy"] = price_full["apy"].ffill().bfill().fillna(5.0)

    # Initial buy
    p0 = price_full["close"].iloc[0]
    if p0 <= 0 or np.isnan(p0):
        p0 = 1.0
    tokens = INITIAL_INVESTMENT / p0

    nav_rows = []
    for _, row in price_full.iterrows():
        daily_rate = row["apy"] / 100.0 / 365.0
        tokens *= (1.0 + daily_rate)
        nav = tokens * row["close"]
        nav_rows.append({
            "date": row["date"],
            "nav":  nav,
            "tokens": tokens,
            "price": row["close"],
            "apy": row["apy"],
        })

    df = pd.DataFrame(nav_rows)
    df["drawdown"] = (df["nav"] / df["nav"].cummax() - 1.0) * 100.0
    return df


# ─── Plotly helpers ───────────────────────────────────────────────────────────

_LAYOUT_BASE = dict(
    plot_bgcolor  = "#ffffff",
    paper_bgcolor = "#ffffff",
    font          = dict(family="Inter, Helvetica Neue, sans-serif", size=11, color="#374151"),
    margin        = dict(l=10, r=10, t=36, b=10),
    hovermode     = "x unified",
    legend        = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                         bgcolor="rgba(255,255,255,0.8)", bordercolor="#e5e7eb", borderwidth=1),
    xaxis         = dict(showgrid=False, zeroline=False, tickfont=dict(size=10)),
    yaxis         = dict(gridcolor="#f3f4f6", zeroline=False, tickfont=dict(size=10)),
)


def _exploit_line() -> go.Scatter:
    return go.Scatter(
        x=[EXPLOIT_DATE, EXPLOIT_DATE],
        y=[0, 1],
        yaxis="y",
        mode="lines",
        line=dict(color=C_EXPLOIT, width=1.2, dash="dot"),
        name="USR exploit (Mar 22)",
        showlegend=True,
    )


def apply_layout(fig: go.Figure, title: str, yformat: str | None = None,
                 height: int = 300) -> go.Figure:
    kw = dict(**_LAYOUT_BASE)
    kw["title"] = dict(text=title, font=dict(size=13, color="#111827"), x=0, xanchor="left")
    kw["height"] = height
    if yformat:
        kw["yaxis"]["tickformat"] = yformat
    fig.update_layout(**kw)
    return fig


# ─── Chart builders ───────────────────────────────────────────────────────────

def chart_nav(nav_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    # NAV line
    fig.add_trace(go.Scatter(
        x=nav_df["date"], y=nav_df["nav"],
        name="Portfolio NAV",
        line=dict(color=C_NAV, width=2),
        fill="tozeroy",
        fillcolor="rgba(22,163,74,0.08)",
        hovertemplate="<b>NAV: $%{y:,.0f}</b><extra></extra>",
    ))
    # Initial investment reference
    fig.add_hline(
        y=INITIAL_INVESTMENT, line_dash="dash", line_color="#9ca3af",
        line_width=1.2, annotation_text="$50K entry",
        annotation_font_size=9, annotation_font_color="#9ca3af",
    )
    # Exploit marker
    fig.add_vline(
        x=EXPLOIT_DATE.timestamp() * 1000,
        line_dash="dot", line_color=C_EXPLOIT, line_width=1.2,
        annotation_text="  USR exploit",
        annotation_font_size=9, annotation_font_color=C_EXPLOIT,
    )
    apply_layout(fig, "Portfolio NAV — $50K initial · stUSR daily compounding",
                 yformat="$,.0f", height=300)
    fig.update_yaxes(tickprefix="$")
    return fig


def chart_tvl(tvl_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tvl_df["date"], y=tvl_df["tvl_usd"] / 1e6,
        name="Resolv TVL",
        line=dict(color=C_TVL, width=2),
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.08)",
        hovertemplate="<b>TVL: $%{y:.1f}M</b><extra></extra>",
    ))
    fig.add_vline(
        x=EXPLOIT_DATE.timestamp() * 1000,
        line_dash="dot", line_color=C_EXPLOIT, line_width=1.2,
    )
    apply_layout(fig, "Resolv Protocol TVL (DeFiLlama)", height=260)
    fig.update_yaxes(ticksuffix="M", tickprefix="$")
    return fig


def chart_depeg(price_df: pd.DataFrame) -> go.Figure:
    df = price_df.copy()
    upper = 1.0 + DEPEG_THRESHOLD   # 1.005
    lower = 1.0 - DEPEG_THRESHOLD   # 0.995

    fig = go.Figure()
    # Price line
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["close"],
        name="USR / USDC",
        line=dict(color=C_USR, width=1.8),
        hovertemplate="<b>$%{y:.4f}</b><extra></extra>",
    ))
    # ±0.5% bands
    fig.add_hline(y=upper, line_dash="dash", line_color=C_THRESHOLD, line_width=1.2,
                  annotation_text=f"+0.5%  ({upper:.3f})",
                  annotation_font_size=9, annotation_font_color=C_THRESHOLD,
                  annotation_position="top right")
    fig.add_hline(y=lower, line_dash="dash", line_color=C_THRESHOLD, line_width=1.2,
                  annotation_text=f"−0.5%  ({lower:.3f})",
                  annotation_font_size=9, annotation_font_color=C_THRESHOLD,
                  annotation_position="bottom right")
    fig.add_hline(y=1.0, line_dash="dot", line_color="#9ca3af", line_width=0.8)
    # Depeg fill (below lower band)
    depeg_mask = df["close"] < lower
    if depeg_mask.any():
        fig.add_trace(go.Scatter(
            x=pd.concat([df["date"], df["date"][::-1]]),
            y=pd.concat([df["close"].clip(upper=lower), pd.Series([lower]*len(df), index=df.index)[::-1]]),
            fill="toself",
            fillcolor="rgba(220,38,38,0.12)",
            line=dict(width=0),
            name="Depeg zone",
            hoverinfo="skip",
        ))
    # Exploit line
    fig.add_vline(
        x=EXPLOIT_DATE.timestamp() * 1000,
        line_dash="dot", line_color=C_EXPLOIT, line_width=1.2,
        annotation_text="  USR exploit",
        annotation_font_size=9, annotation_font_color=C_EXPLOIT,
    )
    apply_layout(fig, "USR / USDC Price — ±0.5% withdrawal trigger bands", height=300)
    fig.update_yaxes(tickprefix="$")
    return fig


def chart_liquidity(price_df: pd.DataFrame) -> go.Figure:
    df = price_df.copy().set_index("date").sort_index()
    # 7-day rolling volume as liquidity proxy
    rolling_vol = df["volume_usd"].fillna(0).rolling("7D", min_periods=1).sum() / 1e6

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_vol.index, y=rolling_vol.values,
        name="7-day rolling volume",
        line=dict(color=C_USR, width=1.8),
        fill="tozeroy",
        fillcolor="rgba(234,88,12,0.10)",
        hovertemplate="<b>$%{y:.2f}M (7d)</b><extra></extra>",
    ))
    fig.add_vline(
        x=EXPLOIT_DATE.timestamp() * 1000,
        line_dash="dot", line_color=C_EXPLOIT, line_width=1.2,
    )
    apply_layout(fig, "USR Pool Liquidity — 7-Day Rolling Volume (Uniswap V3)", height=260)
    fig.update_yaxes(ticksuffix="M", tickprefix="$")
    return fig


def chart_mint_burn(mb_df: pd.DataFrame) -> go.Figure:
    df = mb_df.copy()
    # Threshold: 3-sigma above mean for net outflows (burns > mints)
    net_neg = df["net"][df["net"] < 0]
    threshold = -(net_neg.abs().mean() + 2 * net_neg.abs().std()) if len(net_neg) > 2 else -10e6

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["date"], y=df["minted"] / 1e6,
        name="Minted", marker_color=C_MINT, opacity=0.8,
        hovertemplate="<b>Minted: %{y:.2f}M USR</b><extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=df["date"], y=-df["burned"] / 1e6,
        name="Burned", marker_color=C_BURN, opacity=0.8,
        hovertemplate="<b>Burned: %{y:.2f}M USR</b><extra></extra>",
    ))
    # Net line
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["net"] / 1e6,
        name="Net flow",
        line=dict(color="#111827", width=1.5),
        hovertemplate="<b>Net: %{y:.2f}M USR</b><extra></extra>",
    ))
    # Threshold
    fig.add_hline(
        y=threshold / 1e6, line_dash="dash", line_color=C_THRESHOLD, line_width=1.2,
        annotation_text=f"Outflow alert ({threshold/1e6:.0f}M)",
        annotation_font_size=9, annotation_font_color=C_THRESHOLD,
        annotation_position="bottom right",
    )
    fig.add_vline(
        x=EXPLOIT_DATE.timestamp() * 1000,
        line_dash="dot", line_color=C_EXPLOIT, line_width=1.2,
    )
    apply_layout(fig, "USR Mint / Burn — Daily Net Flow (Etherscan Transfer Events)", height=280)
    fig.update_yaxes(ticksuffix="M USR")
    fig.update_layout(barmode="overlay")
    return fig


# ─── Date filter ──────────────────────────────────────────────────────────────

def filter_by_window(df: pd.DataFrame, window: str,
                     end_date: pd.Timestamp) -> pd.DataFrame:
    deltas = {"1M": 30, "3M": 91, "6M": 182}
    days   = deltas.get(window, 182)
    start  = end_date - pd.Timedelta(days=days)
    mask   = (df["date"] >= start) & (df["date"] <= end_date)
    return df[mask].copy()


# ─── Main app ────────────────────────────────────────────────────────────────

def main() -> None:
    # Header
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:8px;">
          <div style="width:44px;height:44px;background:#ea580c;border-radius:10px;
                      display:flex;align-items:center;justify-content:center;
                      font-size:22px;">🔷</div>
          <div>
            <div style="font-size:1.4rem;font-weight:800;color:#111827;line-height:1.1;">
              Resolv USR Strategy Monitor
            </div>
            <div style="font-size:0.78rem;color:#6b7280;margin-top:2px;">
              $50K allocation · stUSR yield compounding · Oct 2025 – Apr 2026
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load data
    data = load_all()
    nav_df = compute_nav(data["price"], data["apy"])

    # Window selector
    end_date = pd.Timestamp("2026-04-01", tz="UTC")
    col_sel, *_ = st.columns([3, 7])
    with col_sel:
        window = st.radio(
            "Time window",
            ["1M", "3M", "6M"],
            index=2,
            horizontal=True,
            label_visibility="collapsed",
        )

    # Apply date filter
    nav_w   = filter_by_window(nav_df,       window, end_date)
    tvl_w   = filter_by_window(data["tvl"],  window, end_date)
    price_w = filter_by_window(data["price"], window, end_date)
    mb_w    = filter_by_window(data["mb"],   window, end_date)
    apy_w   = filter_by_window(data["apy"],  window, end_date)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    cur_nav   = nav_w["nav"].iloc[-1] if len(nav_w) else INITIAL_INVESTMENT
    peak_nav  = nav_w["nav"].max() if len(nav_w) else INITIAL_INVESTMENT
    max_dd    = nav_w["drawdown"].min() if len(nav_w) else 0.0
    min_price = price_w["close"].min() if len(price_w) else 1.0
    avg_apy   = apy_w["apy"].mean() if len(apy_w) else 0.0
    cur_tvl   = tvl_w["tvl_usd"].iloc[-1] / 1e6 if len(tvl_w) else 0.0

    pnl       = cur_nav - INITIAL_INVESTMENT
    pnl_pct   = pnl / INITIAL_INVESTMENT * 100

    def kpi(label: str, value: str, delta: str = "", delta_type: str = "neu") -> str:
        delta_class = f"kpi-delta-{delta_type}"
        delta_html = f'<div class="{delta_class}">{delta}</div>' if delta else ""
        return (
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'{delta_html}'
            f'</div>'
        )

    pnl_type = "pos" if pnl >= 0 else "neg"
    dd_type  = "neg" if max_dd < -5 else "neu"

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(kpi("Current NAV", f"${cur_nav:,.0f}",
                        f"{'▲' if pnl>=0 else '▼'} ${abs(pnl):,.0f} ({pnl_pct:+.1f}%)",
                        pnl_type), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("Peak NAV", f"${peak_nav:,.0f}"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("Max Drawdown", f"{max_dd:.1f}%",
                        "from peak", dd_type), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("Min USR Price", f"${min_price:.4f}",
                        f"{'⚠ DEPEGGED' if min_price < 0.995 else '✓ In range'}",
                        "neg" if min_price < 0.995 else "pos"), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi("Avg APY (stUSR)", f"{avg_apy:.1f}%",
                        f"Annualised yield", "neu"), unsafe_allow_html=True)
    with c6:
        st.markdown(kpi("Latest TVL", f"${cur_tvl:.1f}M",
                        "Resolv protocol", "neu"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    st.plotly_chart(chart_nav(nav_w), use_container_width=True, config={"displayModeBar": False})

    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(chart_tvl(tvl_w), use_container_width=True, config={"displayModeBar": False})
    with col_r:
        st.plotly_chart(chart_liquidity(price_w), use_container_width=True, config={"displayModeBar": False})

    st.plotly_chart(chart_depeg(price_w), use_container_width=True, config={"displayModeBar": False})
    st.plotly_chart(chart_mint_burn(mb_w), use_container_width=True, config={"displayModeBar": False})

    # Footer
    st.markdown(
        """
        <hr>
        <div style="font-size:0.68rem;color:#9ca3af;padding:4px 0 8px;">
          Data sources: DeFiLlama (TVL, stUSR APY) · GeckoTerminal (USR/USDC daily OHLCV, Uniswap V3)
          · Etherscan (USR ERC-20 Transfer events) &nbsp;|&nbsp;
          Liquidity = 7-day rolling volume (pool TVL unavailable without archive RPC) &nbsp;|&nbsp;
          NAV = $50K initial ÷ USR price₀ → tokens compounded daily at stUSR APY × price
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
