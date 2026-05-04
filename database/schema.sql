CREATE TABLE IF NOT EXISTS binance_trades (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol TEXT NOT NULL,
    market_type TEXT NOT NULL,
    order_type TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity NUMERIC NOT NULL,
    executed_price NUMERIC,
    binance_order_id TEXT,
    client_order_id TEXT,
    raw_response JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE UNIQUE INDEX IF NOT EXISTS binance_trades_order_uidx
ON binance_trades (market_type, symbol, binance_order_id)
WHERE binance_order_id IS NOT NULL;
