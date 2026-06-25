import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

st.set_page_config(page_title="Bitcoin Monster 223 | Market Oracle", page_icon="🐋", layout="wide")

APP_NAME = "GARIBALDI MARKET ORACLE™"
DEFAULT_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]
DEFAULT_STOCKS = ["AAPL", "NVDA", "TSLA", "RKLB", "GLW", "PLTR"]
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 GaribaldiMarketOracle/1.0"})

CRYPTO_IDS = {
    "BTC-USD": "bitcoin", "ETH-USD": "ethereum", "SOL-USD": "solana",
    "BNB-USD": "binancecoin", "XRP-USD": "ripple", "DOGE-USD": "dogecoin",
}
BINANCE = {"BTC-USD":"BTCUSDT", "ETH-USD":"ETHUSDT", "SOL-USD":"SOLUSDT", "BNB-USD":"BNBUSDT", "XRP-USD":"XRPUSDT", "DOGE-USD":"DOGEUSDT"}

def is_crypto(symbol: str) -> bool:
    return symbol.upper() in CRYPTO_IDS or symbol.upper().endswith(("-USD", "USDT"))

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close"])
    return df[["Open", "High", "Low", "Close", "Volume"]]

def interval_to_binance(interval: str) -> str:
    return {"5m":"5m", "15m":"15m", "1h":"1h", "1d":"1d"}.get(interval, "1h")

def period_to_limit(period: str, interval: str) -> int:
    days = {"1d":1, "5d":5, "1mo":30, "3mo":90, "6mo":180, "1y":365}.get(period, 5)
    per_day = {"5m":288, "15m":96, "1h":24, "1d":1}.get(interval, 24)
    return max(24, min(1000, days * per_day))

def yahoo_chart(symbol: str, period: str, interval: str) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    r = SESSION.get(url, params={"range": period, "interval": interval}, timeout=10)
    r.raise_for_status()
    result = (r.json().get("chart", {}).get("result") or [])
    if not result:
        return pd.DataFrame()
    item = result[0]
    ts = item.get("timestamp") or []
    q = (item.get("indicators", {}).get("quote") or [{}])[0]
    df = pd.DataFrame({
        "Datetime": pd.to_datetime(ts, unit="s", utc=True),
        "Open": q.get("open") or [], "High": q.get("high") or [],
        "Low": q.get("low") or [], "Close": q.get("close") or [],
        "Volume": q.get("volume") or [],
    }).set_index("Datetime")
    return normalize(df)

def binance_klines(symbol: str, period: str, interval: str) -> pd.DataFrame:
    pair = BINANCE.get(symbol.upper())
    if not pair:
        return pd.DataFrame()
    url = "https://api.binance.us/api/v3/klines"
    params = {"symbol": pair, "interval": interval_to_binance(interval), "limit": period_to_limit(period, interval)}
    r = SESSION.get(url, params=params, timeout=10)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["open_time","Open","High","Low","Close","Volume","close_time","qv","trades","tb","tq","ignore"])
    df["Datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return normalize(df.set_index("Datetime"))

def coingecko_market_chart(symbol: str, period: str) -> pd.DataFrame:
    coin = CRYPTO_IDS.get(symbol.upper())
    if not coin:
        return pd.DataFrame()
    days = {"1d":1, "5d":5, "1mo":30, "3mo":90, "6mo":180, "1y":365}.get(period, 5)
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    r = SESSION.get(url, params={"vs_currency":"usd", "days":days}, timeout=10)
    r.raise_for_status()
    prices = r.json().get("prices") or []
    if not prices:
        return pd.DataFrame()
    df = pd.DataFrame(prices, columns=["ms", "Close"])
    df["Datetime"] = pd.to_datetime(df["ms"], unit="ms", utc=True)
    df["Open"] = df["High"] = df["Low"] = df["Close"]
    df["Volume"] = np.nan
    return normalize(df.set_index("Datetime"))

def stooq_daily(symbol: str) -> pd.DataFrame:
    # Free stock fallback; daily only. Not for crypto.
    s = symbol.lower().replace(".", "-")
    url = f"https://stooq.com/q/d/l/?s={s}.us&i=d"
    r = SESSION.get(url, timeout=10)
    r.raise_for_status()
    if "No data" in r.text or len(r.text) < 40:
        return pd.DataFrame()
    from io import StringIO
    df = pd.read_csv(StringIO(r.text))
    if "Date" not in df.columns:
        return pd.DataFrame()
    df["Datetime"] = pd.to_datetime(df["Date"], utc=True)
    df = df.rename(columns={"Open":"Open", "High":"High", "Low":"Low", "Close":"Close", "Volume":"Volume"})
    return normalize(df.set_index("Datetime").tail(370))

@st.cache_data(ttl=60, show_spinner=False)
def load_history(symbol: str, period: str = "5d", interval: str = "1h") -> Tuple[pd.DataFrame, str]:
    symbol = symbol.strip().upper()
    attempts = []
    providers = []
    if yf is not None:
        providers.append(("Yahoo yfinance", lambda: yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True, threads=False)))
    providers.append(("Yahoo chart API", lambda: yahoo_chart(symbol, period, interval)))
    if is_crypto(symbol):
        providers.extend([
            ("Binance.US public candles", lambda: binance_klines(symbol, period, interval)),
            ("CoinGecko market chart", lambda: coingecko_market_chart(symbol, period)),
        ])
    else:
        providers.append(("Stooq daily fallback", lambda: stooq_daily(symbol)))

    for name, fn in providers:
        try:
            df = normalize(fn())
            if not df.empty:
                return df, name
            attempts.append(f"{name}: empty")
        except Exception as exc:
            attempts.append(f"{name}: {str(exc)[:90]}")
        time.sleep(0.05)
    return pd.DataFrame(), " | ".join(attempts)

@st.cache_data(ttl=60, show_spinner=False)
def spot_board(symbols: Tuple[str, ...]) -> pd.DataFrame:
    rows = []
    crypto = [s for s in symbols if s in CRYPTO_IDS]
    if crypto:
        try:
            ids = ",".join(CRYPTO_IDS[s] for s in crypto)
            r = SESSION.get("https://api.coingecko.com/api/v3/simple/price", params={"ids":ids,"vs_currencies":"usd","include_24hr_change":"true","include_24hr_vol":"true"}, timeout=10)
            r.raise_for_status()
            data = r.json()
            rev = {v:k for k,v in CRYPTO_IDS.items()}
            for cid, p in data.items():
                rows.append({"Symbol": rev.get(cid, cid), "Price": p.get("usd"), "24h Change %": p.get("usd_24h_change"), "24h Volume": p.get("usd_24h_vol"), "Source": "CoinGecko"})
        except Exception:
            pass
    return pd.DataFrame(rows)

def oracle_metrics(df: pd.DataFrame, horizon_days: int = 5) -> Dict[str, Optional[float]]:
    df = normalize(df)
    if df.empty or len(df) < 3:
        return {"spot": None, "target": None, "ceiling": None, "floor": None, "vol": None}
    close = df["Close"].dropna()
    ret = np.log(close / close.shift(1)).dropna()
    spot = float(close.iloc[-1])
    if ret.empty:
        return {"spot": spot, "target": None, "ceiling": None, "floor": None, "vol": None}
    mu, sigma = float(ret.mean()), float(ret.std() or 0)
    target = spot * np.exp((mu - 0.5 * sigma**2) * horizon_days)
    radius = sigma * np.sqrt(horizon_days)
    return {"spot": spot, "target": target, "ceiling": target*np.exp(2*radius), "floor": target*np.exp(-2*radius), "vol": sigma}

def signal(m: Dict[str, Optional[float]]) -> str:
    if not m.get("spot") or not m.get("target"):
        return "NO DATA"
    edge = (m["target"] - m["spot"]) / m["spot"]
    return "BULL WATCH" if edge > .035 else "RISK WATCH" if edge < -.035 else "NEUTRAL"

def money(x):
    try:
        x = float(x)
        return f"${x:,.4f}" if abs(x) < 1 else f"${x:,.2f}"
    except Exception:
        return "—"

def pct(x):
    try: return f"{float(x):.2f}%"
    except Exception: return "—"

st.title(f"🐋 {APP_NAME}")
st.caption("Multi-source Railway-safe market app: Yahoo/yfinance + Yahoo Chart + CoinGecko + Binance.US + Stooq. Failed providers are skipped, not crashed.")

with st.sidebar:
    st.header("Controls")
    market_mode = st.radio("Market", ["Crypto", "Stocks", "Custom"], horizontal=True)
    period = st.selectbox("History", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=1)
    interval = st.selectbox("Interval", ["5m", "15m", "1h", "1d"], index=2)
    horizon = st.slider("Oracle horizon days", 1, 30, 5)
    if st.button("Refresh now"):
        st.cache_data.clear()
    st.caption("Research/paper-trading only. Not financial advice.")

if market_mode == "Crypto":
    symbols = DEFAULT_SYMBOLS
elif market_mode == "Stocks":
    symbols = DEFAULT_STOCKS
else:
    raw = st.text_input("Enter tickers separated by commas", "BTC-USD, ETH-USD, GLW, NVDA")
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]

feed = spot_board(tuple(symbols))
if not feed.empty:
    st.subheader("Live spot fallback feed")
    view = feed.copy()
    view["Price"] = view["Price"].map(money)
    view["24h Change %"] = view["24h Change %"].map(pct)
    st.dataframe(view, use_container_width=True, hide_index=True)

st.subheader("Oracle board")
rows, histories, sources = [], {}, {}
with st.spinner("Trying every available market source safely..."):
    for sym in symbols:
        df, src = load_history(sym, period, interval)
        histories[sym], sources[sym] = df, src
        m = oracle_metrics(df, horizon)
        rows.append({"Symbol": sym, "Spot": money(m["spot"]), "Target": money(m["target"]), "Floor": money(m["floor"]), "Ceiling": money(m["ceiling"]), "Signal": signal(m), "Provider Used / Status": src if not df.empty else "NO DATA: " + src})

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

left, right = st.columns([2, 1])
with left:
    selected = st.selectbox("Chart symbol", symbols)
    df = histories.get(selected, pd.DataFrame())
    if df.empty:
        st.warning(f"No chart data for {selected}. App is healthy; all providers failed or timed out.")
    else:
        st.line_chart(df["Close"], use_container_width=True)
with right:
    st.metric("App status", "Healthy")
    st.write("Last refresh UTC:", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    st.write("Selected source:", sources.get(selected, "—"))
    st.info("Deploy-safe: cached requests, multiple fallbacks, no startup crash, no broker live-trading without manual approval.")

st.subheader("Provider order")
st.code("Yahoo yfinance → Yahoo Chart API → crypto: Binance.US → crypto: CoinGecko → stocks: Stooq daily fallback", language="text")
