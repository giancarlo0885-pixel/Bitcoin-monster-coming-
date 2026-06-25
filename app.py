import os, sys, subprocess, time, math

# Railway sometimes starts Python files in "bare mode" instead of Streamlit.
# This guard forces the app to relaunch correctly as: streamlit run app.py
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    _ctx = get_script_run_ctx()
except Exception:
    _ctx = None

if _ctx is None and os.environ.get("BITCOIN_MONSTER_STREAMLIT") != "1":
    env = os.environ.copy()
    env["BITCOIN_MONSTER_STREAMLIT"] = "1"
    port = env.get("PORT", "8501")
    cmd = [
        sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__),
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    os.execvpe(sys.executable, cmd, env)

import requests
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Bitcoin Monster Oracle", page_icon="₿", layout="wide")

HEADERS = {"User-Agent": "Mozilla/5.0 BitcoinMonster/1.0"}
CRYPTO_IDS = {"BTC-USD":"bitcoin", "ETH-USD":"ethereum", "SOL-USD":"solana", "XRP-USD":"ripple", "DOGE-USD":"dogecoin", "BNB-USD":"binancecoin"}
BINANCE_SYMBOLS = {"BTC-USD":"BTCUSDT", "ETH-USD":"ETHUSDT", "SOL-USD":"SOLUSDT", "XRP-USD":"XRPUSDT", "DOGE-USD":"DOGEUSDT", "BNB-USD":"BNBUSDT"}

@st.cache_data(ttl=60, show_spinner=False)
def yahoo_chart(symbol, rng="5d", interval="1h"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    r = requests.get(url, params={"range": rng, "interval": interval}, headers=HEADERS, timeout=10)
    r.raise_for_status()
    result = r.json()["chart"]["result"][0]
    ts = result.get("timestamp") or []
    quote = result["indicators"]["quote"][0]
    close = quote.get("close") or []
    df = pd.DataFrame({"time": pd.to_datetime(ts, unit="s"), "close": close}).dropna()
    if df.empty: raise ValueError("Yahoo returned no rows")
    return df, "Yahoo Chart API"

@st.cache_data(ttl=60, show_spinner=False)
def coingecko_chart(symbol):
    cid = CRYPTO_IDS.get(symbol)
    if not cid: raise ValueError("No CoinGecko id")
    url = f"https://api.coingecko.com/api/v3/coins/{cid}/market_chart"
    r = requests.get(url, params={"vs_currency":"usd", "days":"5"}, headers=HEADERS, timeout=10)
    r.raise_for_status()
    prices = r.json().get("prices", [])
    df = pd.DataFrame(prices, columns=["ms", "close"])
    if df.empty: raise ValueError("CoinGecko returned no rows")
    df["time"] = pd.to_datetime(df["ms"], unit="ms")
    return df[["time","close"]].dropna(), "CoinGecko"

@st.cache_data(ttl=60, show_spinner=False)
def binance_chart(symbol):
    bs = BINANCE_SYMBOLS.get(symbol)
    if not bs: raise ValueError("No Binance symbol")
    url = "https://api.binance.us/api/v3/klines"
    r = requests.get(url, params={"symbol":bs, "interval":"1h", "limit":120}, headers=HEADERS, timeout=10)
    r.raise_for_status()
    rows = r.json()
    df = pd.DataFrame(rows)
    if df.empty: raise ValueError("Binance returned no rows")
    return pd.DataFrame({"time":pd.to_datetime(df[0], unit="ms"), "close":pd.to_numeric(df[4], errors="coerce")}).dropna(), "Binance.US"

@st.cache_data(ttl=60, show_spinner=False)
def stooq_chart(symbol):
    s = symbol.lower().replace("-usd", "usd")
    url = f"https://stooq.com/q/d/l/?s={s}&i=d"
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(r.text))
    if df.empty or "Close" not in df: raise ValueError("Stooq returned no rows")
    return pd.DataFrame({"time":pd.to_datetime(df["Date"]), "close":pd.to_numeric(df["Close"], errors="coerce")}).dropna().tail(120), "Stooq"

@st.cache_data(ttl=60, show_spinner=False)
def yfinance_chart(symbol):
    import yfinance as yf
    df = yf.download(symbol, period="5d", interval="1h", progress=False, auto_adjust=True, threads=False)
    if df is None or df.empty: raise ValueError("yfinance returned no rows")
    close = df["Close"]
    if hasattr(close, "columns"): close = close.iloc[:,0]
    out = pd.DataFrame({"time": pd.to_datetime(df.index), "close": pd.to_numeric(close, errors="coerce")}).dropna()
    if out.empty: raise ValueError("yfinance close empty")
    return out, "Yahoo/yfinance"

def get_market(symbol):
    errors=[]
    for fn in (yfinance_chart, yahoo_chart, binance_chart, coingecko_chart, stooq_chart):
        try:
            return fn(symbol)
        except Exception as e:
            errors.append(f"{fn.__name__}: {str(e)[:90]}")
    return pd.DataFrame(), "No source available: " + " | ".join(errors)

def oracle(df, days=5):
    prices = df["close"].astype(float).values
    spot = float(prices[-1])
    if len(prices) < 3:
        return spot, spot, spot
    ret = np.diff(np.log(prices))
    mu = float(np.nanmean(ret)); sig = float(np.nanstd(ret))
    tgt = spot * math.exp((mu - 0.5 * sig**2) * days)
    ceil = tgt * math.exp(2 * sig * math.sqrt(days))
    return spot, tgt, ceil

st.title("GARIBALDI MARKET ORACLE™")
st.caption("Multi-source crypto/stock dashboard with Yahoo, Yahoo Chart API, Binance.US, CoinGecko, and Stooq fallbacks.")

symbols = st.multiselect("Assets", ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","DOGE-USD","AAPL","NVDA","TSLA","GLW","RKLB"], default=["BTC-USD","ETH-USD","SOL-USD","DOGE-USD"])

cols = st.columns(4)
for i, sym in enumerate(symbols):
    df, source = get_market(sym)
    with cols[i % 4]:
        if df.empty:
            st.error(sym)
            st.caption(source)
        else:
            spot, tgt, ceil = oracle(df)
            st.metric(sym, f"${spot:,.4f}" if spot < 10 else f"${spot:,.2f}")
            st.caption(f"Source: {source} | AI target: ${tgt:,.2f} | ceiling: ${ceil:,.2f}")

chart_symbol = st.selectbox("Chart asset", symbols if symbols else ["BTC-USD"])
df, source = get_market(chart_symbol)
if not df.empty:
    st.subheader(f"{chart_symbol} chart — {source}")
    st.line_chart(df.set_index("time")["close"])
else:
    st.warning(source)

st.info("For education/research only. This app does not guarantee profits or place real trades.")
