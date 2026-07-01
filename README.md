# GARIBALDI MARKET ORACLE™ v4

Railway-ready Streamlit dashboard with live market reporting, AI-style signal scoring, and a paper-trading bot.

## Fix included
The previous Railway build failed because Railway used Python 3.13 and pandas had to compile from source. This build includes `runtime.txt` pinned to Python 3.12.10 and updated dependency ranges so Railway can install prebuilt wheels.

## Features
- Live crypto feed with Binance -> Coinbase -> Yahoo fallback
- Yahoo support for stocks and ETFs
- Auto-refresh dashboard
- AI signal engine using EMA, RSI, MACD, Bollinger Bands, momentum, and volume spikes
- Paper trading bot with trade throttle
- Multi-symbol paper portfolio
- Fear & Greed Index
- Yahoo Finance news feed
- Railway deployment files included

## Railway start command
```bash
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

## Files
- `app.py` main Streamlit app
- `requirements.txt` Python dependencies
- `runtime.txt` forces Python 3.12.10
- `Procfile` Railway/Heroku-style start command
- `railway.json` Railway deployment settings

## Safety
Real-money trading is intentionally disabled. Use paper mode only unless you later add exchange API integrations with strict limits, withdrawal-disabled keys, and a kill switch.
