import json
import os
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


try:
    import psycopg
except ImportError:
    psycopg = None

try:
    import psycopg2
except ImportError:
    psycopg2 = None


class PostgresTradeRecorderError(Exception):
    """Raised when the trade recorder cannot connect to or write to Postgres."""


@dataclass(frozen=True)
class TradeRecord:
    symbol: str
    market_type: str
    order_type: str
    side: str
    quantity: Decimal
    executed_price: Optional[Decimal]
    binance_order_id: Optional[str]
    client_order_id: Optional[str]
    raw_response: Dict[str, Any]


class PostgresTradeRecorder:
    """
    Stores Binance order executions in Postgres.

    The docker-postgres.yml defaults are used unless POSTGRES_* environment
    variables override them.
    """

    FILTERABLE_COLUMNS = {
        "id",
        "created_at",
        "symbol",
        "market_type",
        "order_type",
        "side",
        "quantity",
        "executed_price",
        "binance_order_id",
        "client_order_id",
        "raw_response",
    }

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        table_name: Optional[str] = None,
    ):
        load_dotenv()
        self.host = host or os.getenv("POSTGRES_HOST", "localhost")
        self.port = int(port or os.getenv("POSTGRES_PORT", "5432"))
        self.database = database or os.getenv("POSTGRES_DB", "brdasset")
        self.user = user or os.getenv("POSTGRES_USER", "assetuser")
        self.password = password or os.getenv("POSTGRES_PASSWORD", "assetpassword")
        self.table_name = table_name or os.getenv("POSTGRES_TRADES_TABLE", "binance_trades")
        self._validate_identifier(self.table_name)
        self.driver = self._get_driver()
        self._ensure_schema()

    @staticmethod
    def _get_driver():
        if psycopg is not None:
            return psycopg
        if psycopg2 is not None:
            return psycopg2
        raise PostgresTradeRecorderError(
            "No Postgres Python driver found. Install one of: psycopg[binary] or psycopg2-binary."
        )

    @staticmethod
    def _validate_identifier(identifier: str) -> None:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier) is None:
            raise PostgresTradeRecorderError(f"Invalid Postgres identifier: {identifier}")

    @staticmethod
    def _to_decimal(value: Any) -> Optional[Decimal]:
        if value in (None, ""):
            return None
        try:
            return Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            return None

    @staticmethod
    def _first_present(source: Dict[str, Any], *keys: str) -> Any:
        for key in keys:
            value = source.get(key)
            if value not in (None, ""):
                return value
        return None

    def _connect(self):
        return self.driver.connect(
            host=self.host,
            port=self.port,
            dbname=self.database,
            user=self.user,
            password=self.password,
        )

    def _ensure_schema(self) -> None:
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
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
            raw_response JSONB NOT NULL DEFAULT '{{}}'::jsonb
        );
        """
        create_index_sql = f"""
        CREATE UNIQUE INDEX IF NOT EXISTS {self.table_name}_order_uidx
        ON {self.table_name} (market_type, symbol, binance_order_id)
        WHERE binance_order_id IS NOT NULL;
        """
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(create_table_sql)
                cursor.execute(create_index_sql)
            conn.commit()
        finally:
            conn.close()

    def build_trade_record(
        self,
        response: Dict[str, Any],
        market_type: str,
        fallback_order: Optional[Dict[str, Any]] = None,
    ) -> TradeRecord:
        fallback_order = fallback_order or {}

        symbol = self._first_present(response, "symbol") or fallback_order.get("symbol")
        order_type = self._first_present(response, "type") or fallback_order.get("type")
        side = self._first_present(response, "side") or fallback_order.get("side")
        quantity = self._to_decimal(self._first_present(response, "executedQty", "origQty") or fallback_order.get("quantity"))
        executed_price = self._extract_executed_price(response)

        if symbol is None or order_type is None or side is None or quantity is None:
            raise PostgresTradeRecorderError(f"Could not build a complete trade record from Binance response: {response}")

        return TradeRecord(
            symbol=str(symbol),
            market_type=market_type,
            order_type=str(order_type),
            side=str(side),
            quantity=quantity,
            executed_price=executed_price,
            binance_order_id=self._string_or_none(self._first_present(response, "orderId")),
            client_order_id=self._string_or_none(self._first_present(response, "clientOrderId", "clientOrderID")),
            raw_response=response,
        )

    def record_trade(
        self,
        response: Dict[str, Any],
        market_type: str,
        fallback_order: Optional[Dict[str, Any]] = None,
    ) -> TradeRecord:
        record = self.build_trade_record(response, market_type, fallback_order=fallback_order)
        insert_sql = f"""
        INSERT INTO {self.table_name} (
            symbol,
            market_type,
            order_type,
            side,
            quantity,
            executed_price,
            binance_order_id,
            client_order_id,
            raw_response
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT DO NOTHING;
        """
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    insert_sql,
                    (
                        record.symbol,
                        record.market_type,
                        record.order_type,
                        record.side,
                        str(record.quantity),
                        str(record.executed_price) if record.executed_price is not None else None,
                        record.binance_order_id,
                        record.client_order_id,
                        json.dumps(record.raw_response, default=str),
                    ),
                )
            conn.commit()
        finally:
            conn.close()
        return record

    def get_trades(
        self,
        id: Optional[int] = None,
        created_at: Optional[Any] = None,
        created_from: Optional[Any] = None,
        created_to: Optional[Any] = None,
        symbol: Optional[str | List[str]] = None,
        market_type: Optional[str | List[str]] = None,
        order_type: Optional[str | List[str]] = None,
        side: Optional[str | List[str]] = None,
        quantity: Optional[Any] = None,
        executed_price: Optional[Any] = None,
        binance_order_id: Optional[str | List[str]] = None,
        client_order_id: Optional[str | List[str]] = None,
        raw_response: Optional[Dict[str, Any]] = None,
        raw_response_contains: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = 500,
        offset: int = 0,
        order_by: str = "created_at",
        ascending: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Read trades using optional filters for every table column.

        Scalar values use equality, lists use SQL IN, and None means no filter
        unless passed through get_trades_by_column().
        """
        filters = {
            "id": id,
            "created_at": created_at,
            "symbol": symbol,
            "market_type": market_type,
            "order_type": order_type,
            "side": side,
            "quantity": quantity,
            "executed_price": executed_price,
            "binance_order_id": binance_order_id,
            "client_order_id": client_order_id,
            "raw_response": raw_response,
        }
        return self._select_trades(
            filters=filters,
            created_from=created_from,
            created_to=created_to,
            raw_response_contains=raw_response_contains,
            limit=limit,
            offset=offset,
            order_by=order_by,
            ascending=ascending,
        )

    def get_trades_by_column(
        self,
        column: str,
        value: Any,
        limit: Optional[int] = 500,
        offset: int = 0,
        order_by: str = "created_at",
        ascending: bool = False,
    ) -> List[Dict[str, Any]]:
        """Read trades filtered by any single table column."""
        self._validate_filter_column(column)
        return self._select_trades(
            filters={column: value},
            limit=limit,
            offset=offset,
            order_by=order_by,
            ascending=ascending,
            include_null_filters=True,
        )

    def get_trades_by_symbol(self, symbol: str | List[str], **kwargs) -> List[Dict[str, Any]]:
        return self.get_trades(symbol=symbol, **kwargs)

    def get_trade_by_id(self, trade_id: int, **kwargs) -> List[Dict[str, Any]]:
        return self.get_trades(id=trade_id, **kwargs)

    def get_trades_by_created_at(self, created_at: Any, **kwargs) -> List[Dict[str, Any]]:
        return self.get_trades(created_at=created_at, **kwargs)

    def get_trades_by_market_type(self, market_type: str | List[str], **kwargs) -> List[Dict[str, Any]]:
        return self.get_trades(market_type=market_type, **kwargs)

    def get_trades_by_order_type(self, order_type: str | List[str], **kwargs) -> List[Dict[str, Any]]:
        return self.get_trades(order_type=order_type, **kwargs)

    def get_trades_by_side(self, side: str | List[str], **kwargs) -> List[Dict[str, Any]]:
        return self.get_trades(side=side, **kwargs)

    def get_trades_by_quantity(self, quantity: Any, **kwargs) -> List[Dict[str, Any]]:
        return self.get_trades(quantity=quantity, **kwargs)

    def get_trades_by_executed_price(self, executed_price: Any, **kwargs) -> List[Dict[str, Any]]:
        return self.get_trades(executed_price=executed_price, **kwargs)

    def get_trade_by_binance_order_id(self, binance_order_id: str, **kwargs) -> List[Dict[str, Any]]:
        return self.get_trades(binance_order_id=binance_order_id, **kwargs)

    def get_trade_by_client_order_id(self, client_order_id: str, **kwargs) -> List[Dict[str, Any]]:
        return self.get_trades(client_order_id=client_order_id, **kwargs)

    def _select_trades(
        self,
        filters: Optional[Dict[str, Any]] = None,
        created_from: Optional[Any] = None,
        created_to: Optional[Any] = None,
        raw_response_contains: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = 500,
        offset: int = 0,
        order_by: str = "created_at",
        ascending: bool = False,
        include_null_filters: bool = False,
    ) -> List[Dict[str, Any]]:
        self._validate_filter_column(order_by)
        where_parts = []
        params = []

        for column, value in (filters or {}).items():
            self._validate_filter_column(column)
            if value is None and not include_null_filters:
                continue

            clause, clause_params = self._build_filter_clause(column, value)
            where_parts.append(clause)
            params.extend(clause_params)

        if created_from is not None:
            where_parts.append("created_at >= %s")
            params.append(created_from)
        if created_to is not None:
            where_parts.append("created_at <= %s")
            params.append(created_to)
        if raw_response_contains is not None:
            where_parts.append("raw_response @> %s::jsonb")
            params.append(json.dumps(raw_response_contains, default=str))

        where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        direction = "ASC" if ascending else "DESC"
        limit_sql = ""
        if limit is not None:
            if int(limit) < 0:
                raise PostgresTradeRecorderError("limit must be non-negative or None.")
            limit_sql = "LIMIT %s"
            params.append(int(limit))

        if int(offset) < 0:
            raise PostgresTradeRecorderError("offset must be non-negative.")
        offset_sql = "OFFSET %s"
        params.append(int(offset))

        select_sql = f"""
        SELECT
            id,
            created_at,
            symbol,
            market_type,
            order_type,
            side,
            quantity,
            executed_price,
            binance_order_id,
            client_order_id,
            raw_response
        FROM {self.table_name}
        {where_sql}
        ORDER BY {order_by} {direction}
        {limit_sql}
        {offset_sql};
        """
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(select_sql, tuple(params))
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
        finally:
            conn.close()

    def _build_filter_clause(self, column: str, value: Any) -> tuple[str, List[Any]]:
        if value is None:
            return f"{column} IS NULL", []

        if column == "raw_response":
            return f"{column} = %s::jsonb", [json.dumps(value, default=str)]

        if isinstance(value, (list, tuple, set)):
            values = list(value)
            if not values:
                return "FALSE", []
            return f"{column} IN ({', '.join(['%s'] * len(values))})", values

        return f"{column} = %s", [value]

    def _validate_filter_column(self, column: str) -> None:
        if column not in self.FILTERABLE_COLUMNS:
            raise PostgresTradeRecorderError(f"Invalid trade filter column: {column}")

    def _extract_executed_price(self, response: Dict[str, Any]) -> Optional[Decimal]:
        avg_price = self._to_decimal(response.get("avgPrice"))
        if avg_price is not None and avg_price != 0:
            return avg_price

        fills = response.get("fills")
        if isinstance(fills, list) and fills:
            total_qty = Decimal("0")
            total_quote = Decimal("0")
            for fill in fills:
                fill_price = self._to_decimal(fill.get("price"))
                fill_qty = self._to_decimal(fill.get("qty"))
                if fill_price is None or fill_qty is None:
                    continue
                total_qty += fill_qty
                total_quote += fill_price * fill_qty
            if total_qty != 0:
                return total_quote / total_qty

        executed_qty = self._to_decimal(response.get("executedQty"))
        quote_qty = self._to_decimal(response.get("cumQuote")) or self._to_decimal(response.get("cummulativeQuoteQty"))
        if executed_qty is not None and quote_qty is not None and executed_qty != 0:
            return quote_qty / executed_qty

        price = self._to_decimal(response.get("price"))
        if price is not None and price != 0:
            return price

        return None

    @staticmethod
    def _string_or_none(value: Any) -> Optional[str]:
        if value in (None, ""):
            return None
        return str(value)


if __name__ == "__main__":
    postgres = PostgresTradeRecorder()
    print(postgres.get_trades())