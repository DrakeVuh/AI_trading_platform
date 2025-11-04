# AI_trading_platform

Collect OHLCV from CCXT exchanges (Binance by default) with:
- Incremental range (`--since`, `--to`)
- Retry/backoff on rate limits & network errors
- Deduplicated, sorted output
- Parquet/CSV, optional monthly partitioning
- Deterministic folder layout: `data/raw/crypto/<exchange>/<SYMBOL>/<TIMEFRAME>/...`


## Quickstart


```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env