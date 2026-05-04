# Binance Trade Database

This package records executed Binance orders in Postgres.

Default connection settings match `docker-postgres.yml`:

- host: `localhost`
- port: `5432`
- database: `brdasset`
- user: `assetuser`
- password: `assetpassword`
- table: `binance_trades`

Optional `.env` overrides:

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=brdasset
POSTGRES_USER=assetuser
POSTGRES_PASSWORD=assetpassword
POSTGRES_TRADES_TABLE=binance_trades
```

Install one Postgres Python driver before executing orders:

```powershell
pip install "psycopg[binary]"
```

The recorder creates the table automatically on startup. The SQL definition is also available in `database/schema.sql`.

## Reading Trades

```python
from database import PostgresTradeRecorder

recorder = PostgresTradeRecorder()

btc_spot_buys = recorder.get_trades(
    symbol="BTCUSDT",
    market_type="spot",
    side="BUY",
)

futures_trades = recorder.get_trades_by_market_type("futures")
one_order = recorder.get_trade_by_binance_order_id("123456789")

recent_btc = recorder.get_trades(
    symbol="BTCUSDT",
    created_from="2026-04-29 00:00:00-03",
    limit=100,
)
```

`get_trades()` supports filters for every table column:

- `id`
- `created_at`
- `symbol`
- `market_type`
- `order_type`
- `side`
- `quantity`
- `executed_price`
- `binance_order_id`
- `client_order_id`
- `raw_response`

It also supports `created_from`, `created_to`, `raw_response_contains`, `limit`, `offset`, `order_by`, and `ascending`.

For dynamic single-column filters:

```python
recorder.get_trades_by_column("symbol", "ETHUSDT")
recorder.get_trades_by_column("side", ["BUY", "SELL"])
```
