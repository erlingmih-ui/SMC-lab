import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from smc_core import analyse, backtest, build_setups, summarize

st.set_page_config(page_title="SMC Signal Lab", layout="wide")


@st.cache_data(ttl=900, show_spinner=False)
def load_yahoo(ticker: str, interval: str) -> pd.DataFrame:
    period = "60d" if interval in ("15m", "30m", "1h") else "2y"
    raw = yf.download(ticker, period=period, interval=interval,
                      auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw[["Open", "High", "Low", "Close"]].dropna()


@st.cache_data(show_spinner=False)
def load_csv(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content), index_col=0, parse_dates=True)
    df.columns = [str(col).strip().capitalize() for col in df.columns]
    return df[["Open", "High", "Low", "Close"]].dropna()


@st.cache_data(show_spinner="Calcul SMC et backtest...")
def run_engine(df, n, ob_lookback, require_fvg, rr, buffer_atr, max_wait, spread):
    events, zones = analyse(df, n=n, ob_lookback=ob_lookback, require_fvg=require_fvg)
    setups = build_setups(df, events, zones, rr=rr, buffer_atr=buffer_atr, max_wait=max_wait)
    trades, setups = backtest(df, setups, spread=spread)
    return events, zones, trades, setups


def build_figure(df, zones, events, setups, current, lo, hi):
    w = df.iloc[lo:hi]
    fig = go.Figure(go.Candlestick(
        x=w.index, open=w["Open"], high=w["High"], low=w["Low"], close=w["Close"],
        name="Prix",
        increasing_line_color="#1fa35b", decreasing_line_color="#d64545",
    ))

    last_t = df.index[hi - 1]

    def t(k):
        return df.index[min(max(k, lo), hi - 1)]

    for z in zones:
        if z.start >= hi or (z.end is not None and z.end < lo):
            continue
        base = "0,170,90" if z.side == 1 else "230,60,60"
        alpha = 0.30 if z.kind == "OB" else 0.12
        fig.add_shape(
            type="rect",
            x0=t(z.start),
            x1=t(z.end) if z.end is not None else last_t,
            y0=z.bottom, y1=z.top,
            fillcolor=f"rgba({base},{alpha})",
            line=dict(width=0), layer="below",
        )

    for ev in events:
        if lo <= ev.index < hi:
            fig.add_annotation(
                x=df.index[ev.index], y=float(df["High"].iloc[ev.index]),
                text=ev.label, showarrow=True, arrowhead=2, ay=-30,
                font=dict(color="#1fa35b" if ev.side == 1 else "#ff6b6b"),
            )

    if current is not None and current.created < hi:
        x0 = t(current.created)
        for y, name, color in [
            (current.entry, "Entrée", "#4da3ff"),
            (current.stop, "Stop", "#ff5555"),
            (current.target, "Objectif", "#1fa35b"),
        ]:
            fig.add_shape(type="line", x0=x0, x1=last_t, y0=y, y1=y,
                          line=dict(color=color, dash="dot"))
            fig.add_annotation(x=last_t, y=y, text=f"{name} {y:.5f}",
                               showarrow=False, xanchor="right",
                               font=dict(color=color))

    fills = [s for s in setups if s.fill_index is not None and lo <= s.fill_index < hi]
    if fills:
        fig.add_trace(go.Scatter(
            x=[df.index[s.fill_index] for s in fills],
            y=[s.fill_price for s in fills],
            mode="markers", name="Entrée exécutée",
            marker=dict(symbol=["triangle-up" if s.side == 1 else "triangle-down" for s in fills],
                        size=12, color="#4da3ff"),
        ))

    exits = [s for s in setups if s.exit_index is not None and lo <= s.exit_index < hi]
    if exits:
        fig.add_trace(go.Scatter(
            x=[df.index[s.exit_index] for s in exits],
            y=[s.exit_price for s in exits],
            mode="markers", name="Sortie",
            marker=dict(symbol="x", size=11,
                        color=["#1fa35b" if s.status == "objectif" else "#ff5555" for s in exits]),
        ))

    fig.update_layout(height=640, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h"))
    return fig


# ---------- Barre latérale ----------
with st.sidebar:
    st.header("Données")
    source = st.radio("Source", ["Yahoo Finance", "Fichier CSV"], horizontal=True)
    if source == "Yahoo Finance":
        ticker = st.text_input("Ticker", "EURUSD=X")
        interval = st.selectbox("Unité de temps", ["15m", "30m", "1h", "1d"], index=2)
        df = load_yahoo(ticker, interval)
    else:
        upload = st.file_uploader("CSV : date en 1re colonne, Open/High/Low/Close",
                                  type="csv")
        df = load_csv(upload.getvalue()) if upload else pd.DataFrame()

    st.header("Structure")
    n = st.slider("Pivot N", 2, 10, 3)
    ob_lookback = st.slider("Recherche d'Order Block (bougies)", 2, 10, 5)
    require_fvg = st.checkbox("Exiger un FVG après l'OB", value=False)

    st.header("Trade")
    rr = st.slider("Objectif (ratio R)", 1.0, 5.0, 2.0, 0.5)
    buffer_atr = st.slider("Marge de stop (x ATR)", 0.0, 1.0, 0.2, 0.05)
    max_wait = st.slider("Validité du setup (bougies)", 5, 100, 30)
    spread = st.number_input("Coût par trade (unité de prix)", min_value=0.0,
                             value=0.0, step=0.0001, format="%.5f")
    bars = st.slider("Bougies affichées", 100, 1000, 300, 50)

if df.empty:
    st.info("Chargez des données pour commencer (ticker valide ou fichier CSV).")
    st.stop()

events, zones, trades, setups = run_engine(
    df, n, ob_lookback, require_fvg, rr, buffer_atr, max_wait, spread
)

live = [s for s in setups if s.status in ("en attente", "en cours")]
current = live[-1] if live else (setups[-1] if setups else None)
lo, hi = max(0, len(df) - bars), len(df)

tab_sig, tab_chart, tab_bt, tab_method = st.tabs(
    ["Signal", "Graphique", "Backtest", "Méthodologie"]
)

# ---------- Signal ----------
with tab_sig:
    st.caption(f"Dernière bougie : {df.index[-1]} · clôture {df['Close'].iloc[-1]:.5f}")
    if current is None:
        st.info("Aucun setup détecté sur la période.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sens", "ACHAT" if current.side == 1 else "VENTE")
        c2.metric("Entrée", f"{current.entry:.5f}")
        c3.metric("Stop loss", f"{current.stop:.5f}")
        c4.metric("Objectif", f"{current.target:.5f}")

        risk = abs(current.entry - current.stop)
        st.write(
            f"**Statut :** {current.status} · **Risque :** {risk:.5f} · "
            f"**Ratio :** 1:{(abs(current.target - current.entry) / risk):.1f} · "
            f"**Créé :** {df.index[current.created]} · "
            f"**Valide jusqu'à :** {df.index[min(current.expiry, len(df) - 1)]}"
        )
        if current.fill_index is not None:
            st.write(f"Entrée exécutée à {current.fill_price:.5f} le {df.index[current.fill_index]}.")

        st.markdown(
            "**Règle d'exécution :** ordre limite à l'entrée, stop fixe, sortie au stop ou à "
            "l'objectif. Ne déplacez pas le stop, n'ajoutez pas de position."
        )
        st.warning(
            "Analyse algorithmique à but éducatif. Aucun résultat n'est garanti. "
            "Validez la méthode en backtest hors échantillon puis sur compte démo "
            "avant toute décision réelle."
        )

# ---------- Graphique ----------
with tab_chart:
    fig = build_figure(df, zones, events, setups, current, lo, hi)
    st.plotly_chart(fig, use_container_width=True)

# ---------- Backtest ----------
with tab_bt:
    stats = summarize(trades)
    if not stats:
        st.info("Aucun trade clôturé avec ces paramètres.")
    else:
        pf = stats["profit_factor"]
        m = st.columns(6)
        m[0].metric("Trades", stats["n"])
        m[1].metric("Taux de réussite", f"{stats['win_rate']:.0%}")
        m[2].metric("Espérance (R)", f"{stats['expectancy']:.2f}")
        m[3].metric("Profit factor", "∞" if pf == float("inf") else f"{pf:.2f}")
        m[4].metric("R total", f"{stats['total_r']:.1f}")
        m[5].metric("Drawdown max (R)", f"{stats['max_dd_r']:.1f}")

        st.line_chart(trades["r"].cumsum().rename("Courbe de capital (R)"))
        st.dataframe(trades[["side", "fill_index", "fill_price", "stop", "target",
                             "exit_index", "exit_price", "status", "r"]])
        st.download_button("Exporter les trades (CSV)",
                           trades.to_csv(index=False), "trades.csv", "text/csv")
        st.caption("Hypothèse prudente : si stop et objectif sont touchés dans la même "
                   "bougie, le stop est retenu. Les résultats passés ne garantissent rien.")

# ---------- Méthodologie ----------
with tab_method:
    st.markdown("""
**Pivots** : un sommet (creux) est une bougie plus haute (basse) que les N bougies de chaque côté.
Il n'est confirmé qu'après N bougies, donc aucun résultat ne dépend du futur.

**BOS** : clôture au-delà du dernier swing, dans le sens de la tendance.
**CHoCH** : clôture au-delà du dernier swing opposé, qui change la tendance.

**Order Block** : dernière bougie de sens opposé avant la cassure. Option : exiger un FVG après.

**FVG** : déséquilibre sur trois bougies. La zone est mitigée au premier contact.

**Setup** : entrée limite au bord de l'OB, stop au-delà avec une marge en ATR,
objectif à ratio R fixe. Un seul trade à la fois, validité limitée dans le temps.

**Backtest** : remplissage à l'entrée, sortie au stop ou à l'objectif, coût fixe par trade.
Pas de slippage variable, pas de swap, pas d'optimisation automatique.
""")
