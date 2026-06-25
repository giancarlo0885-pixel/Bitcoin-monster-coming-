# Bitcoin Monster 223 — Railway Fixed

Railway-safe Streamlit market dashboard.

## Data provider fallback order

Yahoo yfinance → Yahoo Chart API → Binance.US crypto candles → CoinGecko crypto chart/spot → Stooq daily stock fallback.

The app does not crash if one provider blocks, times out, or returns empty data.

## Railway start command

```bash
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```
