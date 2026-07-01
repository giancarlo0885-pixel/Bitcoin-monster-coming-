import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Garibaldi Crypto Prediction Bot v3", page_icon="₿", layout="wide")

DEFAULT_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "AVAX-USD", "LINK-USD"]
BINANCE_MAP = {
    "BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT", "SOL-USD": "SOLUSDT", "XRP-USD": "XRPUSDT",
    "DOGE-USD": "DOGEUSDT", "ADA-USD": "ADAUSDT", "AVAX-USD": "AVAXUSDT", "LINK-USD": "LINKUSDT"
}

st.markdown("""
<style>
.stApp {background: #080b12; color: #f5f7fb;}
.big-title {font-size: 50px; font-weight: 900; letter-spacing: 1px;}
.card {padding: 18px; border-radius: 18px; background: #111827; border: 1px solid #263244;}
.good {color:#5df28b; font-weight:800;} .bad {color:#ff5c7a; font-weight:800;} .warn {color:#f7c948; font-weight:800;}
.small {color:#aeb7c2; font-size: 13px;}
</style>
""", unsafe_allow_html=True)

# ---------- DATA ----------
@st.cache_data(ttl=20, show_spinner=False)
def fetch_binance_klines(symbol: str, interval="1m", limit=240) -> pd.DataFrame:
    pair = BINANCE_MAP.get(symbol)
    if not pair:
        return pd.DataFrame()
    url = "https://api.binance.com/api/v3/klines"
    r = requests.get(url, params={"symbol": pair, "interval": interval, "limit": limit}, timeout=10)
    r.raise_for_status()
    rows = r.json()
    df = pd.DataFrame(rows, columns=["time","Open","High","Low","Close","Volume","close_time","qav","trades","tbav","tqav","ignore"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["time","Open","High","Low","Close","Volume"]].dropna()

@st.cache_data(ttl=45, show_spinner=False)
def fetch_yahoo(symbol: str, period="2d", interval="5m") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    # yfinance may return MultiIndex columns when multiple ticker metadata is present.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index()
    time_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={time_col: "time"})
    keep = ["time", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        return pd.DataFrame()
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[keep].dropna()

@st.cache_data(ttl=20, show_spinner=False)
def get_market(symbol: str, source="Binance Live") -> pd.DataFrame:
    try:
        if source.startswith("Binance"):
            df = fetch_binance_klines(symbol)
            if not df.empty:
                return df
    except Exception:
        pass
    try:
        return fetch_yahoo(symbol)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=120, show_spinner=False)
def fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8).json()["data"][0]
        return int(r["value"]), r["value_classification"]
    except Exception:
        return None, "Unavailable"

# ---------- INDICATORS ----------
def indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["ret"] = x["Close"].pct_change()
    x["ema9"] = x["Close"].ewm(span=9).mean()
    x["ema21"] = x["Close"].ewm(span=21).mean()
    delta = x["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = 100 - (100 / (1 + rs))
    x["vol_ma"] = x["Volume"].rolling(20).mean()
    x["vol_spike"] = x["Volume"] / x["vol_ma"]
    ma20 = x["Close"].rolling(20).mean()
    sd20 = x["Close"].rolling(20).std()
    x["bb_hi"] = ma20 + 2 * sd20
    x["bb_lo"] = ma20 - 2 * sd20
    return x

def signal_engine(df: pd.DataFrame):
    if df.empty or len(df) < 35:
        return {"signal":"WAIT", "score":0, "reason":"Not enough live candles yet.", "target":None, "stop":None}
    x = indicators(df).dropna()
    if x.empty:
        return {"signal":"WAIT", "score":0, "reason":"Indicators warming up.", "target":None, "stop":None}
    last = x.iloc[-1]
    prev = x.iloc[-2]
    score = 0
    reasons = []
    if last.ema9 > last.ema21 and prev.ema9 <= prev.ema21:
        score += 30; reasons.append("fresh EMA bullish cross")
    elif last.ema9 > last.ema21:
        score += 15; reasons.append("EMA trend bullish")
    if 45 <= last.rsi <= 68:
        score += 20; reasons.append("RSI healthy")
    elif last.rsi < 32:
        score += 10; reasons.append("RSI oversold bounce zone")
    elif last.rsi > 75:
        score -= 20; reasons.append("RSI overheated")
    if last.vol_spike > 1.5 and last.Close > last.Open:
        score += 25; reasons.append("bull volume spike")
    if last.Close > last.bb_hi:
        score += 10; reasons.append("breakout above Bollinger band")
    if last.Close < last.bb_lo:
        score -= 20; reasons.append("breakdown below Bollinger band")
    if x.Close.iloc[-1] > x.Close.iloc[-20]:
        score += 15; reasons.append("20-candle momentum up")
    else:
        score -= 10; reasons.append("20-candle momentum weak")

    if score >= 55:
        sig = "BUY / LONG"
    elif score <= -20:
        sig = "SELL / AVOID"
    else:
        sig = "WAIT"
    price = float(last.Close)
    vol = float(x.ret.tail(60).std() or 0.01)
    target = price * (1 + max(0.008, vol * 3))
    stop = price * (1 - max(0.006, vol * 2))
    return {"signal":sig, "score":int(score), "reason":"; ".join(reasons), "target":target, "stop":stop}

def forecast(df, steps=30):
    if df.empty or len(df) < 20:
        return None
    close = df["Close"].astype(float)
    rets = np.log(close / close.shift(1)).dropna()
    drift = rets.mean() - 0.5 * rets.var()
    vol = rets.std()
    spot = close.iloc[-1]
    target = spot * np.exp(drift * steps)
    ceiling = target * np.exp(2 * vol * np.sqrt(steps))
    floor = target * np.exp(-2 * vol * np.sqrt(steps))
    bull_prob = float((rets.tail(50) > 0).mean() * 100)
    return spot, target, ceiling, floor, bull_prob

# ---------- PAPER BOT ----------
def init_state():
    if "cash" not in st.session_state:
        st.session_state.cash = 10000.0
        st.session_state.position = 0.0
        st.session_state.avg_price = 0.0
        st.session_state.trades = []
        st.session_state.last_bot_signal = None
        st.session_state.last_bot_trade_ts = 0.0
init_state()

def bot_trade(symbol, price, signal, qty_usd, force=False):
    # Prevent the paper bot from buying every refresh on the same repeated signal.
    now_ts = time.time()
    action = "BUY" if signal.startswith("BUY") else "SELL" if signal.startswith("SELL") else "WAIT"
    signal_key = f"{symbol}:{action}"
    if not force and (action == "WAIT" or (st.session_state.last_bot_signal == signal_key and now_ts - st.session_state.last_bot_trade_ts < 180)):
        return
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    if action == "BUY" and st.session_state.cash >= qty_usd:
        qty = qty_usd / price
        old_val = st.session_state.position * st.session_state.avg_price
        st.session_state.position += qty
        st.session_state.avg_price = (old_val + qty_usd) / st.session_state.position
        st.session_state.cash -= qty_usd
        st.session_state.trades.append({"time":now,"symbol":symbol,"side":"PAPER BUY","price":price,"usd":qty_usd})
        st.session_state.last_bot_signal = signal_key
        st.session_state.last_bot_trade_ts = now_ts
    elif action == "SELL" and st.session_state.position > 0:
        qty = st.session_state.position
        usd = qty * price
        st.session_state.cash += usd
        st.session_state.position = 0.0
        st.session_state.avg_price = 0.0
        st.session_state.trades.append({"time":now,"symbol":symbol,"side":"PAPER SELL","price":price,"usd":usd})
        st.session_state.last_bot_signal = signal_key
        st.session_state.last_bot_trade_ts = now_ts

# ---------- UI ----------
st.markdown('<div class="big-title">GARIBALDI CRYPTO PREDICTION BOT™ v3</div>', unsafe_allow_html=True)
st.caption("Live market reports + AI signal engine + paper-trading bot. Educational tool only — not financial advice.")

with st.sidebar:
    st.header("Live Controls")
    symbols = st.multiselect("Coins", DEFAULT_SYMBOLS, default=DEFAULT_SYMBOLS[:6])
    focus = st.selectbox("Focus chart", symbols or DEFAULT_SYMBOLS, index=0)
    source = st.selectbox("Market source", ["Binance Live", "Yahoo fallback"])
    refresh_sec = st.slider("Auto-refresh seconds", 5, 120, 15)
    paper_enabled = st.toggle("AI paper bot live mode", value=True)
    trade_size = st.number_input("Paper trade size ($)", min_value=10.0, max_value=5000.0, value=250.0, step=10.0)
    st.warning("Real-money trading is disabled by default. Use paper mode until API keys, exchange permissions, and risk limits are fully tested.")

st_autorefresh(interval=refresh_sec * 1000, key="live_refresh")

fg_value, fg_class = fear_greed()

# Live report table
rows = []
for sym in symbols:
    df = get_market(sym, source)
    if df.empty:
        rows.append({"Symbol": sym, "Price": np.nan, "Change %": np.nan, "AI Signal":"NO DATA", "Score":0, "Target":np.nan, "Stop":np.nan})
        continue
    sig = signal_engine(df)
    price = float(df["Close"].iloc[-1])
    chg = (price / float(df["Close"].iloc[0]) - 1) * 100
    fc = forecast(df)
    target = sig["target"] if sig["target"] else (fc[1] if fc else np.nan)
    stop = sig["stop"] if sig["stop"] else np.nan
    rows.append({"Symbol": sym, "Price": price, "Change %": chg, "AI Signal": sig["signal"], "Score": sig["score"], "Target": target, "Stop": stop})

live = pd.DataFrame(rows)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Market Feed", "LIVE", f"refresh {refresh_sec}s")
with c2:
    st.metric("Fear & Greed", fg_class, fg_value if fg_value is not None else "--")
with c3:
    best = live.sort_values("Change %", ascending=False).iloc[0] if not live.empty else None
    st.metric("Top Mover", best["Symbol"] if best is not None else "--", f"{best['Change %']:.2f}%" if best is not None and pd.notna(best['Change %']) else "--")
with c4:
    st.metric("Paper Cash", f"${st.session_state.cash:,.2f}")

st.subheader("Live Market Report")
st.dataframe(live, use_container_width=True, hide_index=True)

# Focus chart and signal
st.subheader(f"Live AI Chart: {focus}")
df_focus = get_market(focus, source)
if df_focus.empty:
    st.error("No market data loaded. Try another symbol or wait for refresh.")
else:
    x = indicators(df_focus)
    sig = signal_engine(df_focus)
    price = float(df_focus["Close"].iloc[-1])
    if paper_enabled:
        bot_trade(focus, price, sig["signal"], float(trade_size))

    left, right = st.columns([2, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=x["time"], open=x["Open"], high=x["High"], low=x["Low"], close=x["Close"], name="Price"))
        fig.add_trace(go.Scatter(x=x["time"], y=x["ema9"], name="EMA 9"))
        fig.add_trace(go.Scatter(x=x["time"], y=x["ema21"], name="EMA 21"))
        fig.update_layout(height=520, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=25,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        klass = "good" if sig["signal"].startswith("BUY") else "bad" if sig["signal"].startswith("SELL") else "warn"
        target_txt = f"${sig['target']:,.4f}" if sig.get("target") is not None else "Warming up"
        stop_txt = f"${sig['stop']:,.4f}" if sig.get("stop") is not None else "Warming up"
        st.markdown(f'<div class="card"><h2 class="{klass}">{sig["signal"]}</h2><p>AI score: <b>{sig["score"]}</b></p><p>{sig["reason"]}</p><hr><p>Price: ${price:,.4f}</p><p>Target: {target_txt}</p><p>Stop: {stop_txt}</p></div>', unsafe_allow_html=True)

        fc = forecast(df_focus)
        if fc:
            spot, target, ceiling, floor, bull = fc
            st.metric("Forecast target", f"${target:,.4f}")
            st.metric("Bull probability", f"{bull:.0f}%")

    tabs = st.tabs(["AI Bot", "Alerts", "Portfolio", "Developer / Real Trading Hook"])
    with tabs[0]:
        st.write("The bot reacts every refresh using trend, momentum, RSI, Bollinger breakout, and volume-spike logic.")
        st.write("Mode:", "🟢 Paper bot running" if paper_enabled else "Paused")
        if st.button("Manual PAPER BUY"):
            bot_trade(focus, price, "BUY / LONG", float(trade_size), force=True)
        if st.button("Manual PAPER SELL"):
            bot_trade(focus, price, "SELL / AVOID", float(trade_size), force=True)
    with tabs[1]:
        alerts = []
        last = x.dropna().iloc[-1]
        if last.rsi > 70: alerts.append("RSI overbought: possible pullback risk")
        if last.rsi < 30: alerts.append("RSI oversold: possible bounce zone")
        if last.vol_spike > 1.8: alerts.append("Unusual volume spike detected")
        if last.Close > last.bb_hi: alerts.append("Breakout above upper Bollinger band")
        if last.Close < last.bb_lo: alerts.append("Breakdown below lower Bollinger band")
        if not alerts: alerts = ["No major alerts right now."]
        for a in alerts:
            st.info(a)
    with tabs[2]:
        value = st.session_state.cash + st.session_state.position * price
        pnl = value - 10000
        st.metric("Paper portfolio value", f"${value:,.2f}", f"${pnl:,.2f}")
        st.metric("Position", f"{st.session_state.position:.6f} {focus.replace('-USD','')}")
        if st.session_state.trades:
            st.dataframe(pd.DataFrame(st.session_state.trades), use_container_width=True, hide_index=True)
        if st.button("Reset paper portfolio"):
            st.session_state.cash = 10000.0; st.session_state.position = 0.0; st.session_state.avg_price = 0.0; st.session_state.trades = []; st.session_state.last_bot_signal = None; st.session_state.last_bot_trade_ts = 0.0
            st.rerun()
    with tabs[3]:
        st.code('''# Real trading is intentionally disabled.\n# To add it later:\n# 1) Create exchange account API keys with withdrawal disabled.\n# 2) Store keys in Railway Variables, never in GitHub.\n# 3) Require max daily loss, max trade size, and manual kill switch.\n# 4) Paper trade for at least 30 days before live execution.\n\nEXCHANGE_API_KEY = os.getenv("EXCHANGE_API_KEY")\nEXCHANGE_API_SECRET = os.getenv("EXCHANGE_API_SECRET")\nREAL_TRADING_ENABLED = os.getenv("REAL_TRADING_ENABLED") == "true"\n''', language="python")

st.divider()
st.caption(f"Last run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} | Educational tool only, not financial advice. Live trading requires separate exchange setup and risk controls.")
