import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
import contextlib
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

APP_NAME = "GARIBALDI MARKET ORACLE™ v4.1"
STARTING_CASH = 10_000.0
DEFAULT_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "AVAX-USD", "LINK-USD"]
STOCK_SYMBOLS = ["NVDA", "TSLA", "AAPL", "MSFT", "GLW", "RKLB", "PLTR", "SOFI", "SPY", "QQQ"]
BINANCE_MAP = {
    "BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT", "SOL-USD": "SOLUSDT", "XRP-USD": "XRPUSDT",
    "DOGE-USD": "DOGEUSDT", "ADA-USD": "ADAUSDT", "AVAX-USD": "AVAXUSDT", "LINK-USD": "LINKUSDT"
}
COINBASE_MAP = {
    "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD", "SOL-USD": "SOL-USD", "XRP-USD": "XRP-USD",
    "DOGE-USD": "DOGE-USD", "ADA-USD": "ADA-USD", "AVAX-USD": "AVAX-USD", "LINK-USD": "LINK-USD"
}
COINGECKO_MAP = {
    "BTC-USD": "bitcoin", "ETH-USD": "ethereum", "SOL-USD": "solana", "XRP-USD": "ripple",
    "DOGE-USD": "dogecoin", "ADA-USD": "cardano", "AVAX-USD": "avalanche-2", "LINK-USD": "chainlink"
}

st.set_page_config(page_title=APP_NAME, page_icon="₿", layout="wide")

st.markdown("""
<style>
.stApp {background: #070a12; color: #f7f8fb;}
.big-title {font-size: 44px; font-weight: 900; letter-spacing: 1px; margin-bottom: 2px;}
.subtle {color:#aeb7c2; font-size: 14px;}
.card {padding: 18px; border-radius: 18px; background: #101827; border: 1px solid #263244;}
.good {color:#5df28b; font-weight:800;} .bad {color:#ff5c7a; font-weight:800;} .warn {color:#f7c948; font-weight:800;}
.small {color:#aeb7c2; font-size: 13px;}
</style>
""", unsafe_allow_html=True)


def safe_float(value, default=np.nan) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if np.isfinite(value):
            return value
        return default
    except Exception:
        return default


@st.cache_data(ttl=10, show_spinner=False)
def fetch_binance_klines(symbol: str, interval: str = "1m", limit: int = 240) -> pd.DataFrame:
    pair = BINANCE_MAP.get(symbol)
    if not pair:
        return pd.DataFrame()
    url = "https://api.binance.com/api/v3/klines"
    try:
        r = requests.get(url, params={"symbol": pair, "interval": interval, "limit": limit}, timeout=8)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["time", "Open", "High", "Low", "Close", "Volume", "close_time", "qav", "trades", "tbav", "tqav", "ignore"])
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[["time", "Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=15, show_spinner=False)
def fetch_coinbase_candles(symbol: str, granularity: int = 60) -> pd.DataFrame:
    pair = COINBASE_MAP.get(symbol)
    if not pair:
        return pd.DataFrame()
    url = f"https://api.exchange.coinbase.com/products/{pair}/candles"
    try:
        r = requests.get(url, params={"granularity": granularity}, timeout=8, headers={"User-Agent": "garibaldi-market-oracle"})
        r.raise_for_status()
        rows = r.json()
        if not isinstance(rows, list) or not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["time", "Low", "High", "Open", "Close", "Volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[["time", "Open", "High", "Low", "Close", "Volume"]].dropna().sort_values("time")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=10, show_spinner=False)
def fetch_coingecko_market(symbol: str, days: str = "1") -> pd.DataFrame:
    coin_id = COINGECKO_MAP.get(symbol)
    if not coin_id:
        return pd.DataFrame()
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    try:
        r = requests.get(url, params={"vs_currency": "usd", "days": days}, timeout=10, headers={"User-Agent": "garibaldi-market-oracle"})
        r.raise_for_status()
        data = r.json()
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        if not prices:
            return pd.DataFrame()
        df = pd.DataFrame(prices, columns=["time", "price"])
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        if volumes:
            v = pd.DataFrame(volumes, columns=["time", "Volume"])
            v["time"] = pd.to_datetime(v["time"], unit="ms", utc=True)
            v["Volume"] = pd.to_numeric(v["Volume"], errors="coerce")
            df = pd.merge_asof(df.sort_values("time"), v.sort_values("time"), on="time", direction="nearest")
        else:
            df["Volume"] = 0.0
        ohlc = df.set_index("time").resample("5min").agg(Open=("price", "first"), High=("price", "max"), Low=("price", "min"), Close=("price", "last"), Volume=("Volume", "mean")).dropna().reset_index()
        return ohlc
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30, show_spinner=False)
def fetch_yahoo(symbol: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    try:
        # Railway logs were showing noisy Yahoo/yfinance "1 Failed download" messages.
        # Suppress that noise and let the app cleanly fall back to other data sources.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.reset_index()
        time_col = "Datetime" if "Datetime" in df.columns else "Date"
        df = df.rename(columns={time_col: "time"})
        keep = ["time", "Open", "High", "Low", "Close", "Volume"]
        if any(col not in df.columns for col in keep):
            return pd.DataFrame()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        return df[keep].dropna()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=10, show_spinner=False)
def get_market(symbol: str, source: str) -> pd.DataFrame:
    symbol = symbol.strip().upper()
    if source == "Crypto live: Binance -> Coinbase -> CoinGecko -> Yahoo":
        for fetcher in (fetch_binance_klines, fetch_coinbase_candles, fetch_coingecko_market, fetch_yahoo):
            df = fetcher(symbol)
            if not df.empty:
                return df
    elif source == "CoinGecko live -> Yahoo":
        for fetcher in (fetch_coingecko_market, fetch_yahoo):
            df = fetcher(symbol)
            if not df.empty:
                return df
    elif source == "Yahoo only":
        return fetch_yahoo(symbol)
    else:
        for fetcher in (fetch_yahoo, fetch_coingecko_market):
            df = fetcher(symbol)
            if not df.empty:
                return df
    return pd.DataFrame()


@st.cache_data(ttl=120, show_spinner=False)
def fear_greed() -> tuple[Optional[int], str]:
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        r.raise_for_status()
        row = r.json()["data"][0]
        return int(row["value"]), row["value_classification"]
    except Exception:
        return None, "Unavailable"


@st.cache_data(ttl=180, show_spinner=False)
def market_news() -> List[Dict[str, str]]:
    # Public Yahoo RSS feed. If it fails, app continues.
    try:
        import xml.etree.ElementTree as ET
        url = "https://finance.yahoo.com/news/rssindex"
        r = requests.get(url, timeout=8, headers={"User-Agent": "garibaldi-market-oracle"})
        r.raise_for_status()
        root = ET.fromstring(r.text)
        items = []
        for item in root.findall("./channel/item")[:8]:
            title = item.findtext("title", default="Market headline")
            link = item.findtext("link", default="")
            pub = item.findtext("pubDate", default="")
            items.append({"title": title, "link": link, "published": pub})
        return items
    except Exception:
        return []


def indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_values("time")
    x["ret"] = x["Close"].pct_change()
    x["ema9"] = x["Close"].ewm(span=9, adjust=False).mean()
    x["ema21"] = x["Close"].ewm(span=21, adjust=False).mean()
    x["ema50"] = x["Close"].ewm(span=50, adjust=False).mean()
    delta = x["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = 100 - (100 / (1 + rs))
    ema12 = x["Close"].ewm(span=12, adjust=False).mean()
    ema26 = x["Close"].ewm(span=26, adjust=False).mean()
    x["macd"] = ema12 - ema26
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()
    x["vol_ma"] = x["Volume"].rolling(20).mean()
    x["vol_spike"] = x["Volume"] / x["vol_ma"].replace(0, np.nan)
    ma20 = x["Close"].rolling(20).mean()
    sd20 = x["Close"].rolling(20).std()
    x["bb_hi"] = ma20 + 2 * sd20
    x["bb_lo"] = ma20 - 2 * sd20
    return x


def signal_engine(df: pd.DataFrame) -> Dict[str, object]:
    if df.empty or len(df) < 35:
        return {"signal": "WAIT", "score": 0, "reason": "Not enough live candles yet.", "target": None, "stop": None, "risk": "High"}
    x = indicators(df).dropna()
    if len(x) < 5:
        return {"signal": "WAIT", "score": 0, "reason": "Indicators warming up.", "target": None, "stop": None, "risk": "High"}

    last, prev = x.iloc[-1], x.iloc[-2]
    score = 0
    reasons = []

    if last.ema9 > last.ema21 and prev.ema9 <= prev.ema21:
        score += 30; reasons.append("fresh EMA bullish cross")
    elif last.ema9 > last.ema21:
        score += 15; reasons.append("EMA trend bullish")
    else:
        score -= 10; reasons.append("EMA trend weak")

    if last.macd > last.macd_signal:
        score += 15; reasons.append("MACD positive")
    else:
        score -= 8; reasons.append("MACD negative")

    if 45 <= last.rsi <= 68:
        score += 20; reasons.append("RSI healthy")
    elif last.rsi < 32:
        score += 8; reasons.append("RSI oversold bounce zone")
    elif last.rsi > 75:
        score -= 25; reasons.append("RSI overheated")

    if safe_float(last.vol_spike, 0) > 1.5 and last.Close > last.Open:
        score += 25; reasons.append("bull volume spike")
    elif safe_float(last.vol_spike, 0) > 1.5 and last.Close < last.Open:
        score -= 20; reasons.append("bear volume spike")

    if last.Close > last.bb_hi:
        score += 10; reasons.append("breakout above Bollinger band")
    elif last.Close < last.bb_lo:
        score -= 20; reasons.append("breakdown below Bollinger band")

    momentum_20 = (last.Close / x.Close.iloc[-20] - 1) if len(x) >= 20 else 0
    if momentum_20 > 0:
        score += 15; reasons.append("20-candle momentum up")
    else:
        score -= 10; reasons.append("20-candle momentum weak")

    if score >= 60:
        signal = "BUY / LONG"
    elif score <= -25:
        signal = "SELL / AVOID"
    else:
        signal = "WAIT"

    price = safe_float(last.Close, 0)
    vol = safe_float(x.ret.tail(60).std(), 0.01)
    target = price * (1 + max(0.008, vol * 3)) if price else None
    stop = price * (1 - max(0.006, vol * 2)) if price else None
    risk = "Low" if abs(score) >= 75 else "Medium" if abs(score) >= 45 else "High"
    return {"signal": signal, "score": int(score), "reason": "; ".join(reasons), "target": target, "stop": stop, "risk": risk}


def forecast(df: pd.DataFrame, steps: int = 30):
    if df.empty or len(df) < 20:
        return None
    close = df["Close"].astype(float)
    rets = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    if rets.empty:
        return None
    drift = rets.mean() - 0.5 * rets.var()
    vol = safe_float(rets.std(), 0.01)
    spot = safe_float(close.iloc[-1])
    target = spot * np.exp(drift * steps)
    ceiling = target * np.exp(2 * vol * np.sqrt(steps))
    floor = target * np.exp(-2 * vol * np.sqrt(steps))
    bull_prob = float((rets.tail(50) > 0).mean() * 100)
    return spot, target, ceiling, floor, bull_prob


def init_state():
    if "cash" not in st.session_state:
        st.session_state.cash = STARTING_CASH
        st.session_state.positions = {}
        st.session_state.avg_prices = {}
        st.session_state.trades = []
        st.session_state.last_bot_signal = {}
        st.session_state.last_bot_trade_ts = {}


def bot_trade(symbol: str, price: float, signal: str, qty_usd: float, force: bool = False):
    if price <= 0:
        return
    now_ts = time.time()
    action = "BUY" if signal.startswith("BUY") else "SELL" if signal.startswith("SELL") else "WAIT"
    signal_key = f"{symbol}:{action}"
    if not force:
        repeated = st.session_state.last_bot_signal.get(symbol) == signal_key
        recent = now_ts - st.session_state.last_bot_trade_ts.get(symbol, 0) < 180
        if action == "WAIT" or (repeated and recent):
            return
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pos = float(st.session_state.positions.get(symbol, 0.0))
    avg = float(st.session_state.avg_prices.get(symbol, 0.0))
    if action == "BUY" and st.session_state.cash >= qty_usd:
        qty = qty_usd / price
        old_value = pos * avg
        new_pos = pos + qty
        st.session_state.positions[symbol] = new_pos
        st.session_state.avg_prices[symbol] = (old_value + qty_usd) / new_pos
        st.session_state.cash -= qty_usd
        st.session_state.trades.append({"time": now, "symbol": symbol, "side": "PAPER BUY", "price": price, "usd": qty_usd})
    elif action == "SELL" and pos > 0:
        usd = pos * price
        st.session_state.cash += usd
        st.session_state.positions[symbol] = 0.0
        st.session_state.avg_prices[symbol] = 0.0
        st.session_state.trades.append({"time": now, "symbol": symbol, "side": "PAPER SELL", "price": price, "usd": usd})
    st.session_state.last_bot_signal[symbol] = signal_key
    st.session_state.last_bot_trade_ts[symbol] = now_ts


def portfolio_value(prices: Dict[str, float]) -> float:
    total = float(st.session_state.cash)
    for sym, qty in st.session_state.positions.items():
        total += float(qty) * safe_float(prices.get(sym), 0)
    return total


init_state()

st.markdown(f'<div class="big-title">{APP_NAME}</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Live market reports + AI signal engine + paper-trading bot. Educational tool only — not financial advice.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Live Controls")
    universe = st.radio("Market universe", ["Crypto", "Stocks/ETFs", "Mixed"], horizontal=True)
    base_symbols = DEFAULT_SYMBOLS if universe == "Crypto" else STOCK_SYMBOLS if universe == "Stocks/ETFs" else DEFAULT_SYMBOLS[:6] + STOCK_SYMBOLS[:6]
    symbols = st.multiselect("Watchlist", base_symbols, default=base_symbols[:6])
    focus = st.selectbox("Focus chart", symbols or base_symbols, index=0)
    source = st.selectbox("Data source", ["Crypto live: Binance -> Coinbase -> CoinGecko -> Yahoo", "CoinGecko live -> Yahoo", "Yahoo only"])
    refresh_sec = st.slider("Auto-refresh seconds", 5, 120, 15)
    paper_enabled = st.toggle("AI paper bot live mode", value=True)
    trade_size = st.number_input("Paper trade size ($)", min_value=10.0, max_value=5000.0, value=250.0, step=10.0)
    max_risk = st.slider("Max trade risk score", 1, 100, 75, help="Lower number means the bot will be more conservative.")
    st.warning("Real-money trading is OFF. Paper mode only. Never put exchange API keys in GitHub.")

st_autorefresh(interval=refresh_sec * 1000, key="live_refresh")

fg_value, fg_class = fear_greed()

rows = []
latest_prices: Dict[str, float] = {}
for sym in symbols:
    df = get_market(sym, source)
    if df.empty:
        rows.append({"Symbol": sym, "Price": np.nan, "Change %": np.nan, "AI Signal": "NO DATA", "Score": 0, "Risk": "High", "Target": np.nan, "Stop": np.nan})
        continue
    sig = signal_engine(df)
    price = safe_float(df["Close"].iloc[-1])
    latest_prices[sym] = price
    first = safe_float(df["Close"].iloc[0])
    chg = (price / first - 1) * 100 if first else np.nan
    rows.append({"Symbol": sym, "Price": price, "Change %": chg, "AI Signal": sig["signal"], "Score": sig["score"], "Risk": sig["risk"], "Target": sig["target"], "Stop": sig["stop"]})

live = pd.DataFrame(rows)
valid_live = live.dropna(subset=["Price"])

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Market Feed", "LIVE", f"refresh {refresh_sec}s")
with c2:
    st.metric("Fear & Greed", fg_class, fg_value if fg_value is not None else "--")
with c3:
    if not valid_live.empty:
        best = valid_live.sort_values("Change %", ascending=False).iloc[0]
        st.metric("Top Mover", best["Symbol"], f"{best['Change %']:.2f}%")
    else:
        st.metric("Top Mover", "--", "--")
with c4:
    st.metric("Paper Equity", f"${portfolio_value(latest_prices):,.2f}", f"${portfolio_value(latest_prices) - STARTING_CASH:,.2f}")

st.subheader("Live Market Report")
st.dataframe(live, use_container_width=True, hide_index=True)

st.subheader(f"Live AI Chart: {focus}")
df_focus = get_market(focus, source)
if df_focus.empty:
    st.error("No market data loaded. Try the Crypto live source, another symbol, or wait for refresh. If stocks fail, Yahoo may be temporarily blocking Railway.")
else:
    x = indicators(df_focus)
    sig = signal_engine(df_focus)
    price = safe_float(df_focus["Close"].iloc[-1])
    if paper_enabled and sig["risk"] != "High" and abs(int(sig["score"])) <= max_risk:
        bot_trade(focus, price, str(sig["signal"]), float(trade_size))
    elif paper_enabled and str(sig["signal"]).startswith("BUY") and sig["risk"] in ("Low", "Medium"):
        bot_trade(focus, price, str(sig["signal"]), float(trade_size))

    left, right = st.columns([2, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=x["time"], open=x["Open"], high=x["High"], low=x["Low"], close=x["Close"], name="Price"))
        fig.add_trace(go.Scatter(x=x["time"], y=x["ema9"], name="EMA 9"))
        fig.add_trace(go.Scatter(x=x["time"], y=x["ema21"], name="EMA 21"))
        fig.add_trace(go.Scatter(x=x["time"], y=x["ema50"], name="EMA 50"))
        fig.update_layout(height=520, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        klass = "good" if str(sig["signal"]).startswith("BUY") else "bad" if str(sig["signal"]).startswith("SELL") else "warn"
        target_txt = f"${sig['target']:,.4f}" if sig.get("target") is not None else "Warming up"
        stop_txt = f"${sig['stop']:,.4f}" if sig.get("stop") is not None else "Warming up"
        st.markdown(f'<div class="card"><h2 class="{klass}">{sig["signal"]}</h2><p>AI score: <b>{sig["score"]}</b></p><p>Risk: <b>{sig["risk"]}</b></p><p>{sig["reason"]}</p><hr><p>Price: ${price:,.4f}</p><p>Target: {target_txt}</p><p>Stop: {stop_txt}</p></div>', unsafe_allow_html=True)
        fc = forecast(df_focus)
        if fc:
            spot, target, ceiling, floor, bull = fc
            st.metric("Forecast target", f"${target:,.4f}")
            st.metric("Bull probability", f"{bull:.0f}%")

    tabs = st.tabs(["AI Bot", "Alerts", "Portfolio", "News", "Real Trading Hook"])
    with tabs[0]:
        st.write("The bot reacts every refresh using EMA trend, RSI, MACD, Bollinger bands, momentum, and volume-spike logic.")
        st.write("Mode:", "🟢 Paper bot running" if paper_enabled else "Paused")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Manual PAPER BUY"):
                bot_trade(focus, price, "BUY / LONG", float(trade_size), force=True)
                st.rerun()
        with b2:
            if st.button("Manual PAPER SELL"):
                bot_trade(focus, price, "SELL / AVOID", float(trade_size), force=True)
                st.rerun()
    with tabs[1]:
        alerts = []
        clean = x.dropna()
        if clean.empty:
            alerts.append("Indicators warming up.")
        else:
            last = clean.iloc[-1]
            if last.rsi > 70: alerts.append("RSI overbought: possible pullback risk")
            if last.rsi < 30: alerts.append("RSI oversold: possible bounce zone")
            if safe_float(last.vol_spike, 0) > 1.8: alerts.append("Unusual volume spike detected")
            if last.Close > last.bb_hi: alerts.append("Breakout above upper Bollinger band")
            if last.Close < last.bb_lo: alerts.append("Breakdown below lower Bollinger band")
            if last.macd > last.macd_signal: alerts.append("MACD is bullish")
        if not alerts:
            alerts = ["No major alerts right now."]
        for alert in alerts:
            st.info(alert)
    with tabs[2]:
        value = portfolio_value(latest_prices | {focus: price})
        pnl = value - STARTING_CASH
        st.metric("Paper portfolio value", f"${value:,.2f}", f"${pnl:,.2f}")
        pos_rows = []
        for sym, qty in st.session_state.positions.items():
            if qty <= 0:
                continue
            px = safe_float((latest_prices | {focus: price}).get(sym), 0)
            avg = safe_float(st.session_state.avg_prices.get(sym), 0)
            pos_rows.append({"Symbol": sym, "Quantity": qty, "Avg Price": avg, "Live Price": px, "Value": qty * px, "PnL": qty * (px - avg)})
        if pos_rows:
            st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)
        if st.session_state.trades:
            st.dataframe(pd.DataFrame(st.session_state.trades), use_container_width=True, hide_index=True)
        if st.button("Reset paper portfolio"):
            st.session_state.cash = STARTING_CASH
            st.session_state.positions = {}
            st.session_state.avg_prices = {}
            st.session_state.trades = []
            st.session_state.last_bot_signal = {}
            st.session_state.last_bot_trade_ts = {}
            st.rerun()
    with tabs[3]:
        news = market_news()
        if not news:
            st.write("News feed unavailable right now. The app will keep running.")
        for item in news:
            title = item.get("title", "Market headline")
            link = item.get("link", "")
            pub = item.get("published", "")
            if link:
                st.markdown(f"- [{title}]({link})  ")
            else:
                st.markdown(f"- {title}")
            if pub:
                st.caption(pub)
    with tabs[4]:
        st.code('''# Real trading is intentionally disabled in this build.\n# To add it later:\n# 1) Create exchange API keys with withdrawal disabled.\n# 2) Store keys only in Railway Variables. Never commit keys to GitHub.\n# 3) Require max daily loss, max trade size, stop-losses, and manual kill switch.\n# 4) Paper trade first and review performance before enabling live execution.\n\nEXCHANGE_API_KEY = os.getenv("EXCHANGE_API_KEY")\nEXCHANGE_API_SECRET = os.getenv("EXCHANGE_API_SECRET")\nREAL_TRADING_ENABLED = os.getenv("REAL_TRADING_ENABLED") == "true"\n''', language="python")

st.divider()
st.caption(f"Last run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} | Educational tool only. No prediction is guaranteed. Real trading requires separate exchange setup, testing, and strict risk controls.")
