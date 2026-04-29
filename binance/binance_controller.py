# import logging
import concurrent.futures
import logging
import os
import threading
import requests
import ujson as json
from datetime import datetime, timedelta, time
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any, Dict, Iterator, List, Optional
from time import sleep

import pandas as pd
from binance.client import Client
from dotenv import load_dotenv

import sys
sys.path.append(".")
from singleton_meta import SingletonMeta
from utils import push_notification, setup_logger



WHITELISTED_API_ERROR_CODES: List[int] = [-4003, -4164]
DATE_FORMAT = '%Y%m%d'

FUTURES_MARKET = "futures"
SPOT_MARKET = "spot"
MARKET_ALIASES = {
    "future": FUTURES_MARKET,
    "futures": FUTURES_MARKET,
    "perp": FUTURES_MARKET,
    "perps": FUTURES_MARKET,
    "perpetual": FUTURES_MARKET,
    "perpetual_futures": FUTURES_MARKET,
    "um_futures": FUTURES_MARKET,
    "spot": SPOT_MARKET,
}
VALID_ORDER_SIDES = {"BUY", "SELL"}

DEFAULT_QUOTE_ASSET = "USDT"
QUOTE_ASSETS = ['USDT', 'USDC', 'FDUSD', 'BUSD', 'TUSD', 'BRL', 'ARS', 'BTC', 'ETH]', 'BNB', 'TRY', 'EUR']
UNDERLYING_CURRENCIES = ['USDC', 'USDT']
STABLE_PRICE_ASSETS = {'USDC', 'USDT', 'FDUSD', 'BUSD', 'TUSD'}
CASH_ASSETS = STABLE_PRICE_ASSETS | {'BRL', 'ARS', 'TRY', 'EUR'}
STOP_LIMIT_TOKENS = [] # []
STOP_PRICE_TOL = 0.0005 # 5 bps

ROUND_PRICE_PRECISION = {'BTC': 1, 'ETH': 2, 'SOL': 2}
OBSERVE_LIMIT_ORDER_TIME = 10 # seconds


class BinanceControllerError(Exception):
    """Base exception for Binance controller failures."""


class BinanceAPIResponseError(BinanceControllerError):
    """Raised when Binance returns an API error payload."""

    def __init__(self, error_code: int, error_message: str):
        self.error_code = error_code
        self.error_message = error_message
        super().__init__(f"Binance API error code: {error_code}, message: {error_message}")


class BinanceValidationError(BinanceControllerError, ValueError):
    """Raised when an order request is invalid before it reaches Binance."""


class BinanceOrderExecutionError(BinanceControllerError):
    """Raised when an order request cannot be sent or confirmed."""


def handle_error(
    error_code: int,
    error_message: str,
    logger: Optional[logging.Logger] = None,
    raise_exception: bool = True,
) -> None:
    message = f"Binance API error code: {error_code}, message: {error_message}"
    if logger:
        log_method = logger.warning if error_code in WHITELISTED_API_ERROR_CODES else logger.error
        log_method(message)
    else:
        print(message)

    if error_code not in WHITELISTED_API_ERROR_CODES:
        if error_code == 0:
            push_notification("Binance API Unexpected Error", f"{error_message}")
        push_notification(f"Binance API Error ({error_code})", f"{error_message}")

        if raise_exception:
            raise BinanceAPIResponseError(error_code, error_message)


def handle_single_response(tx_response: Any, logger: Optional[logging.Logger] = None) -> None:
    if not isinstance(tx_response, dict) or 'code' not in tx_response:
        return

    try:
        error_code = int(tx_response['code'])
    except (TypeError, ValueError) as exc:
        handle_error(0, f"Could not parse Binance error response {tx_response}: {exc}", logger=logger)
        return

    handle_error(error_code, tx_response.get('msg', str(tx_response)), logger=logger)


def handle_batch_response(response: Any, logger: Optional[logging.Logger] = None) -> None:
    if isinstance(response, list):
        if len(response) == 0:
            handle_error(0, "Check if the request has gone through", logger=logger)
        for res in response:
            handle_single_response(res, logger=logger)
        return

    handle_single_response(response, logger=logger)
            
def observe_limit_order(
    client: Client,
    symbol: str,
    order_id: int,
    symbol_qty_precision: int,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Checks if the limit order is filled after a certain time, cancels if not, and places a market order for the remainder.
    Improved for reliability, error handling, and logging.
    """
    try:
        order_status = client.futures_get_order(symbol=symbol, orderId=order_id)
        if not order_status or "status" not in order_status:
            push_notification("Limit Order Status Error", f"Could not retrieve status for order {order_id} on {symbol}. Response: {order_status}")
            if logger:
                logger.error(f"Could not retrieve status for order {order_id} on {symbol}. Response: {order_status}")
            return

        status = order_status.get("status")
        executed_qty = float(order_status.get("executedQty", 0))
        orig_qty = float(order_status.get("origQty", 0))
        side = order_status.get("side")

        if status == "FILLED":
            #print(f"Limit order {order_id} for {symbol} has been filled.")
            #push_notification("Limit Order Filled", f"Limit order {order_id} for {symbol} has been filled.")
            if logger:
                logger.info(f"Limit order {order_id} for {symbol} has been filled.")
            return

        if executed_qty < orig_qty:
            if logger:
                logger.warning(f"Limit order {order_id} for {symbol} is not filled. Attempting to cancel.")
            try:
                cancel_response = client.futures_cancel_order(symbol=symbol, orderId=order_id)
                handle_single_response(cancel_response, logger=logger)
            except Exception as cancel_exc:
                if logger:
                    logger.error(f"Exception during cancel attempt for order {order_id}: {cancel_exc}", exc_info=True)
                push_notification("Limit Order Cancel Failed", f"Exception during cancel attempt for order {order_id} on {symbol}: {cancel_exc}")
                return

            if cancel_response and cancel_response.get("status") == "CANCELED":
                cancel_executed_qty = float(cancel_response.get("executedQty", 0))
                cancel_orig_qty = float(cancel_response.get("origQty", 0))
                
                remaining_qty = round(cancel_orig_qty - cancel_executed_qty, symbol_qty_precision)
                #push_notification("Limit Order Canceled", f"Limit order {order_id} for {symbol} was not filled. {remaining_qty} will be sent as a market order.")
                try:
                    market_order_response = client.futures_create_order(
                        symbol=symbol,
                        type='MARKET',
                        side=side,
                        quantity=str(remaining_qty)
                    )
                    handle_single_response(market_order_response, logger=logger)
                    if logger:
                        logger.info(f"Market order placed for remaining quantity of limit order {order_id} for {symbol}.")
                    #push_notification("Market Order Placed", f"Market order placed for remaining quantity ({remaining_qty}) of limit order {order_id} for {symbol}.")
                except Exception as market_exc:
                    if logger:
                        logger.error(f"Error placing market order for remaining quantity of limit order {order_id} for {symbol}: {market_exc}", exc_info=True)
                    push_notification("Market Order Placement Failed", f"Could not place market order for remaining quantity of limit order {order_id} for {symbol}: {market_exc}")
            else:
                if logger:
                    logger.error(f"Failed to cancel limit order {order_id} for {symbol}. Response: {cancel_response}")
                push_notification("Limit Order Cancel Failed", f"Could not cancel limit order {order_id} for {symbol}. Response: {cancel_response}")
        else:
            if logger:
                logger.warning(f"Limit order {order_id} for {symbol} is in unexpected state. Status: {status}, executedQty: {executed_qty}, origQty: {orig_qty}")
            push_notification("Limit Order Unexpected State", f"Order {order_id} for {symbol} status: {status}, executedQty: {executed_qty}, origQty: {orig_qty}")

    except Exception as e:
        if logger:
            logger.error(f"Error checking status of limit order {order_id} for {symbol}: {e}", exc_info=True)
        push_notification("Error Observing Limit Order", f"Could not check status of limit order {order_id} for {symbol}: {e}")


class BinanceController(metaclass=SingletonMeta):

    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("APIKEYBINANCE")
        self.api_secret = os.getenv("APISECRETBINANCE")
        self.binance_client = Client(self.api_key, self.api_secret)
        self.lock = threading.Lock()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
        self.logger = setup_logger(name=__name__, stdout=True, log_file='binance_log.log', level=logging.INFO)
        self.spot_symbol_filters: Dict[str, Dict[str, Decimal]] = {}
        self.futures_symbol_filters: Dict[str, Dict[str, Decimal]] = {}
        self.futures_precision: Dict[str, int] = {}

    def _normalize_market_type(self, market_type: str = FUTURES_MARKET) -> str:
        normalized_market = MARKET_ALIASES.get(str(market_type).lower())
        if normalized_market is None:
            raise BinanceValidationError(
                f"Invalid market_type '{market_type}'. Use '{FUTURES_MARKET}' or '{SPOT_MARKET}'."
            )
        return normalized_market

    def _normalize_side(self, side: str) -> str:
        normalized_side = str(side).upper()
        if normalized_side not in VALID_ORDER_SIDES:
            raise BinanceValidationError(f"Invalid side '{side}'. Use 'BUY' or 'SELL'.")
        return normalized_side

    def _get_base_token(self, token_or_symbol: str) -> str:
        token_or_symbol = str(token_or_symbol).upper()
        for quote_asset in sorted(QUOTE_ASSETS, key=len, reverse=True):
            if token_or_symbol.endswith(quote_asset) and len(token_or_symbol) > len(quote_asset):
                return token_or_symbol[:-len(quote_asset)]
        return token_or_symbol

    def _get_order_symbol(self, token: str, market_type: str = FUTURES_MARKET, quote_asset: str = DEFAULT_QUOTE_ASSET) -> str:
        self._normalize_market_type(market_type)
        token = str(token).upper()
        base_token = self._get_base_token(token)
        if base_token != token:
            return token

        return f"{base_token}{str(quote_asset).upper()}"

    @staticmethod
    def _decimal_to_plain_string(value: Decimal) -> str:
        formatted = format(value.normalize(), "f")
        return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted

    def _get_spot_lot_size_filter(self, symbol: str) -> Dict[str, Decimal]:
        symbol = symbol.upper()
        if symbol in self.spot_symbol_filters:
            return self.spot_symbol_filters[symbol]

        symbol_info = self.binance_client.get_symbol_info(symbol)
        if not symbol_info:
            raise BinanceValidationError(f"Spot symbol {symbol} was not found on Binance.")

        lot_size_filter = next(
            (item for item in symbol_info.get("filters", []) if item.get("filterType") == "LOT_SIZE"),
            None,
        )
        if not lot_size_filter:
            raise BinanceValidationError(f"Spot symbol {symbol} does not expose a LOT_SIZE filter.")

        try:
            parsed_filter = {
                "minQty": Decimal(str(lot_size_filter["minQty"])),
                "maxQty": Decimal(str(lot_size_filter["maxQty"])),
                "stepSize": Decimal(str(lot_size_filter["stepSize"])),
            }
        except (InvalidOperation, KeyError) as exc:
            raise BinanceValidationError(f"Could not parse LOT_SIZE filter for {symbol}: {lot_size_filter}") from exc

        if parsed_filter["stepSize"] <= 0:
            raise BinanceValidationError(f"Spot symbol {symbol} returned an invalid stepSize.")

        self.spot_symbol_filters[symbol] = parsed_filter
        return parsed_filter

    def _get_futures_symbol_info(self, symbol: str) -> Dict[str, Any]:
        symbol = self._get_order_symbol(symbol, FUTURES_MARKET)
        exchange_info = self.binance_client.futures_exchange_info()
        symbol_info = next((item for item in exchange_info.get("symbols", []) if item.get("symbol") == symbol), None)
        if not symbol_info:
            raise BinanceValidationError(f"Futures symbol {symbol} was not found on Binance.")
        return symbol_info

    def _get_futures_lot_size_filter(self, symbol: str) -> Dict[str, Decimal]:
        symbol = self._get_order_symbol(symbol, FUTURES_MARKET)
        if symbol in self.futures_symbol_filters:
            return self.futures_symbol_filters[symbol]

        symbol_info = self._get_futures_symbol_info(symbol)
        lot_size_filter = next(
            (item for item in symbol_info.get("filters", []) if item.get("filterType") in {"MARKET_LOT_SIZE", "LOT_SIZE"}),
            None,
        )
        if not lot_size_filter:
            raise BinanceValidationError(f"Futures symbol {symbol} does not expose a lot size filter.")

        try:
            parsed_filter = {
                "minQty": Decimal(str(lot_size_filter["minQty"])),
                "maxQty": Decimal(str(lot_size_filter["maxQty"])),
                "stepSize": Decimal(str(lot_size_filter["stepSize"])),
            }
        except (InvalidOperation, KeyError) as exc:
            raise BinanceValidationError(f"Could not parse futures lot size filter for {symbol}: {lot_size_filter}") from exc

        if parsed_filter["stepSize"] <= 0:
            raise BinanceValidationError(f"Futures symbol {symbol} returned an invalid stepSize.")

        self.futures_symbol_filters[symbol] = parsed_filter
        return parsed_filter

    def _get_futures_quantity_precision(self, symbol: str) -> int:
        symbol = self._get_order_symbol(symbol, FUTURES_MARKET)
        if symbol in self.futures_precision:
            return self.futures_precision[symbol]

        symbol_info = self._get_futures_symbol_info(symbol)
        precision = int(symbol_info.get("quantityPrecision", 8))
        self.futures_precision[symbol] = precision
        return precision

    def _round_spot_quantity(self, quantity: float, symbol: str) -> str:
        try:
            quantity_decimal = Decimal(str(quantity))
        except InvalidOperation as exc:
            raise BinanceValidationError(f"Invalid spot quantity for {symbol}: {quantity}") from exc

        lot_size = self._get_spot_lot_size_filter(symbol)
        rounded_quantity = (quantity_decimal / lot_size["stepSize"]).to_integral_value(rounding=ROUND_DOWN) * lot_size["stepSize"]

        if rounded_quantity <= 0:
            raise BinanceValidationError(f"Spot quantity for {symbol} must be greater than zero after rounding.")
        if rounded_quantity < lot_size["minQty"]:
            raise BinanceValidationError(
                f"Spot quantity for {symbol} is below minQty {lot_size['minQty']} after rounding: {rounded_quantity}."
            )
        if rounded_quantity > lot_size["maxQty"]:
            raise BinanceValidationError(
                f"Spot quantity for {symbol} is above maxQty {lot_size['maxQty']} after rounding: {rounded_quantity}."
            )

        return self._decimal_to_plain_string(rounded_quantity)

    def _round_order_quantity(self, token: str, quantity: float, market_type: str, symbol: Optional[str] = None) -> str:
        market_type = self._normalize_market_type(market_type)
        if market_type == SPOT_MARKET:
            return self._round_spot_quantity(quantity, symbol or self._get_order_symbol(token, market_type))

        symbol = symbol or self._get_order_symbol(token, market_type)
        try:
            quantity_decimal = Decimal(str(quantity))
        except InvalidOperation as exc:
            raise BinanceValidationError(f"Invalid futures quantity for {symbol}: {quantity}") from exc

        lot_size = self._get_futures_lot_size_filter(symbol)
        rounded_quantity = (quantity_decimal / lot_size["stepSize"]).to_integral_value(rounding=ROUND_DOWN) * lot_size["stepSize"]
        if rounded_quantity <= 0:
            raise BinanceValidationError(f"Futures quantity for {symbol} must be greater than zero after rounding.")
        if rounded_quantity < lot_size["minQty"]:
            raise BinanceValidationError(
                f"Futures quantity for {symbol} is below minQty {lot_size['minQty']} after rounding: {rounded_quantity}."
            )
        if rounded_quantity > lot_size["maxQty"]:
            raise BinanceValidationError(
                f"Futures quantity for {symbol} is above maxQty {lot_size['maxQty']} after rounding: {rounded_quantity}."
            )
        return self._decimal_to_plain_string(rounded_quantity)


    def get_futures_ticker_price(self, token, spot=False) -> float:
        if token != "COREUSDT":
            try:
                if not spot:
                    response = requests.get(f"https://fapi.binance.com/fapi/v2/ticker/price?symbol={token}", timeout=10)
                    response.raise_for_status()
                    ticker_price = float(json.loads(response.text)["price"])
                else:
                    return self.get_spot_ticker_price(token)
                return ticker_price
            except Exception as e:
                self.logger.warning(f"Could not fetch futures ticker price for {token}, falling back to spot: {e}")
                return self.get_spot_ticker_price(token)
        else:
            return 0.0

    def get_spot_ticker_price(self, token) -> float:
        try:
            response = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={token}", timeout=10)
            response.raise_for_status()
            ticker_price = float(json.loads(response.text)["price"])
            return ticker_price
        except Exception as e:
            self.logger.error(f"Error fetching spot ticker price for {token}: {e}", exc_info=True)
            raise e

    def send_market_spot_order(self, token: str, quantity: float, side: str) -> Dict[str, Any]:
        return self.execute_market_individual_order(token, quantity, side, market_type=SPOT_MARKET)

    def execute_market_individual_order(
        self,
        token: str,
        quantity: float,
        side: str,
        market_type: str = FUTURES_MARKET,
    ) -> Dict[str, Any]:
        market_type = self._normalize_market_type(market_type)
        side = self._normalize_side(side)

        try:
            if market_type == SPOT_MARKET:
                order = self._build_spot_market_order(token, quantity, side)
                self.logger.info(f"Submitting spot order: {order}")
                response = self.binance_client.create_order(**order)
            else:
                order = self._build_futures_market_order(token, quantity, side)
                self.logger.info(f"Submitting futures order: {order}")
                response = self.binance_client.futures_create_order(**order)

            handle_single_response(response, logger=self.logger)
            self.logger.info(f"Binance {market_type} order response: {response}")
            return response

        except BinanceControllerError:
            raise
        except Exception as exc:
            self.logger.error(
                f"Error executing {market_type} order for token={token}, quantity={quantity}, side={side}: {exc}",
                exc_info=True,
            )
            raise BinanceOrderExecutionError(
                f"Could not execute {market_type} order for {token}: {exc}"
            ) from exc


    def _build_futures_stop_limit_order(self, token, quantity, side) -> Dict[str, str]:
        symbol = self._get_order_symbol(token, FUTURES_MARKET)
        order_quantity = self._round_order_quantity(token, quantity, FUTURES_MARKET, symbol=symbol)
        order = {
                "symbol": symbol,
                "type": "LIMIT",
                "side": self._normalize_side(side),
                "quantity": order_quantity,
                "priceMatch": "QUEUE",
                "timeInForce": "GTC"
            }
        return order
    
    def _build_futures_market_order(self, token, quantity, side) -> Dict[str, str]:
        symbol = self._get_order_symbol(token, FUTURES_MARKET)
        order_quantity = self._round_order_quantity(token, quantity, FUTURES_MARKET, symbol=symbol)
        order = {
                "symbol": symbol,
                "type": "MARKET",
                "side": self._normalize_side(side),
                "quantity": order_quantity,
            }
        return order

    def _build_spot_market_order(self, token, quantity, side) -> Dict[str, str]:
        symbol = self._get_order_symbol(token, SPOT_MARKET)
        order_quantity = self._round_order_quantity(token, quantity, SPOT_MARKET, symbol=symbol)
        order = {
                "symbol": symbol,
                "type": "MARKET",
                "side": self._normalize_side(side),
                "quantity": order_quantity,
            }
        return order
    
    def _build_futures_order(self, token, quantity, side) -> Dict[str, str]:
        if token in STOP_LIMIT_TOKENS:
            return self._build_futures_stop_limit_order(token, quantity, side)
        else:
            return self._build_futures_market_order(token, quantity, side)

    def _build_order(self, token, quantity, side, market_type: str = FUTURES_MARKET) -> Dict[str, str]:
        market_type = self._normalize_market_type(market_type)
        if market_type == SPOT_MARKET:
            return self._build_spot_market_order(token, quantity, side)
        return self._build_futures_order(token, quantity, side)

            
    def _observe_orders(self, responses: List[Dict[str, str]]) -> None:
        # observe limit orders
        for response in responses:
            try:
                if response.get("type") == "LIMIT":
                    order_id = response.get("orderId")
                    symbol = response.get("symbol")
                    if order_id is not None and symbol is not None and isinstance(order_id, (int, str)) and isinstance(symbol, str):
                        try:
                            symbol_qty_precision = self._get_futures_quantity_precision(symbol)
                            threading.Timer(
                                OBSERVE_LIMIT_ORDER_TIME,
                                observe_limit_order,
                                args=(self.binance_client, symbol, int(order_id), symbol_qty_precision, self.logger)
                            ).start()
                        except Exception as timer_exc:
                            self.logger.error(f"Error starting timer for observing limit order {order_id} on {symbol}: {timer_exc}", exc_info=True)
                    else:
                        self.logger.warning(f"Cannot observe limit order, missing or invalid orderId or symbol in response: {response}")
            except Exception as e:
                self.logger.error(f"Unexpected error in _observe_orders for response: {response} - {e}", exc_info=True)

    def execute_market_order(
        self,
        ticker: str,
        quantity: int,
        side: str,
        market_type: str = FUTURES_MARKET,
    ) -> List[Dict[str, Any]]:
        return [self.execute_market_individual_order(ticker, quantity, side, market_type=market_type)]

    def execute_futures_market_order(self, ticker: str, quantity: int, side: str) -> List[Dict[str, Any]]:
        return [self.execute_market_individual_order(ticker, quantity, side, market_type=FUTURES_MARKET)]

    def execute_perpetual_market_order(self, ticker: str, quantity: int, side: str) -> List[Dict[str, Any]]:
        return [self.execute_market_individual_order(ticker, quantity, side, market_type=FUTURES_MARKET)]

    def execute_spot_market_order(self, ticker: str, quantity: int, side: str) -> List[Dict[str, Any]]:
        return [self.execute_market_individual_order(ticker, quantity, side, market_type=SPOT_MARKET)]

    def get_positions2(self) -> pd.DataFrame:
        positions = self.binance_client.futures_account()["positions"]
        df = pd.DataFrame(positions)

        # Convert columns to float and filter out rows with zero notional
        df["notional"] = df["notional"].astype(float)
        df["positionAmt"] = df["positionAmt"].astype(float)
        df = df[df["notional"] != 0]

        # Calculate price
        df["price"] = df["notional"] / df["positionAmt"]

        # Select relevant columns
        df = df[["symbol", "notional", "positionAmt", "price"]]
        return df

    def get_account_trades(self, symbols: str | List[str], days_back=0, market_type: str = FUTURES_MARKET) -> pd.DataFrame:
        market_type = self._normalize_market_type(market_type)
        start_of_day = datetime.now().replace(hour=1, minute=0, second=0, microsecond=0)
        if days_back > 0:
            start_of_day = start_of_day - timedelta(days=days_back)

        start_of_day_ts = int(start_of_day.timestamp() * 1000)

        all_trades = []
        if isinstance(symbols, str):
            symbols = [symbols]

        for ticker in [self._get_order_symbol(symbol, market_type) for symbol in symbols]:
            try:
                if market_type == SPOT_MARKET:
                    trades = self.binance_client.get_all_orders(symbol=ticker, startTime=start_of_day_ts, limit=100)
                else:
                    trades = self.binance_client.futures_get_all_orders(symbol=ticker, startTime=start_of_day_ts, limit=100)
                ticker_trades_df = pd.DataFrame(trades)

                if ticker_trades_df.empty:
                    continue
                filled_ticker_trades_df = ticker_trades_df[ticker_trades_df['status'] == 'FILLED']

                if filled_ticker_trades_df.empty:
                    continue

                filled_ticker_trades_df = filled_ticker_trades_df.copy()
                filled_ticker_trades_df['market_type'] = market_type
                all_trades.append(filled_ticker_trades_df)
            except Exception as e:
                self.logger.error(f"Error fetching {market_type} account trades at symbol {ticker}: {e}", exc_info=True)
                continue

        if not all_trades:
            return pd.DataFrame()

        trades_df = pd.concat(all_trades, ignore_index=True)
        return trades_df


    def _fetch_trades_for_symbol(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        market_type: str = FUTURES_MARKET,
    ) -> pd.DataFrame:
        """Fetches filled orders for a single symbol within a time range."""
        market_type = self._normalize_market_type(market_type)
        self.logger.debug(f"Fetching {market_type} trades for {symbol} from {start_ts} to {end_ts}")
        try:
            # Use limit=1000 (API max) to reduce chance of missed trades
            # Note: Pagination is NOT implemented here. Assumes 1000 orders per symbol per period is sufficient.
            if market_type == SPOT_MARKET:
                trades = self.binance_client.get_all_orders(
                    symbol=symbol,
                    startTime=start_ts,
                    endTime=end_ts,
                    limit=1000
                )
            else:
                trades = self.binance_client.futures_get_all_orders(
                    symbol=symbol,
                    startTime=start_ts,
                    endTime=end_ts,
                    limit=1000
                )
            if not trades:
                return pd.DataFrame()

            df = pd.DataFrame(trades)
            df_filled = df[df['status'] == 'FILLED'].copy() # Filter for filled orders

            # Optional: Convert types for consistency
            if not df_filled.empty:
                for col in ['price', 'avgPrice', 'executedQty', 'cumQuote', 'cummulativeQuoteQty', 'origQty']:
                    if col in df_filled.columns:
                        df_filled[col] = pd.to_numeric(df_filled[col], errors='coerce')
                if 'time' in df_filled.columns:
                    df_filled['time'] = pd.to_datetime(df_filled['time'], unit='ms')
                if 'updateTime' in df_filled.columns:
                    df_filled['updateTime'] = pd.to_datetime(df_filled['updateTime'], unit='ms')
                df_filled['market_type'] = market_type

            return df_filled

        except Exception as e:
            self.logger.error(f"Unexpected error fetching {market_type} trades for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    def get_trades_for_period(
        self,
        symbols: str | List[str],
        start_ts: int,
        end_ts: int,
        market_type: str = FUTURES_MARKET,
    ) -> pd.DataFrame:
        """
        Fetches all filled account trades for all relevant symbols within a given millisecond timestamp range.
        Uses ThreadPoolExecutor for concurrent fetching per symbol.
        """
        market_type = self._normalize_market_type(market_type)
        all_trades_list = []
        futures = {}
        if isinstance(symbols, str):
            symbols = [symbols]
        relevant_symbols = [self._get_order_symbol(symbol, market_type) for symbol in symbols]

        if not relevant_symbols:
            raise BinanceValidationError("At least one symbol is required to fetch trades.")

        self.logger.info(f"Fetching {market_type} trades for {len(relevant_symbols)} symbols from {datetime.fromtimestamp(start_ts/1000)} to {datetime.fromtimestamp(end_ts/1000)}")

        # Submit tasks for each symbol
        for symbol in relevant_symbols:
             if "MATIC" in symbol: # MATIC has been deprecated for POL
                continue

             futures[self.executor.submit(self._fetch_trades_for_symbol, symbol, start_ts, end_ts, market_type)] = symbol
             sleep(1)

        # Collect results
        for future in concurrent.futures.as_completed(futures):
            symbol = futures[future]
            try:
                result_df = future.result()
                if not result_df.empty:
                    all_trades_list.append(result_df)
                    self.logger.debug(f"Successfully fetched {len(result_df)} trades for {symbol}")
            except Exception as e:
                # Errors during the fetch itself are logged in _fetch_trades_for_symbol
                # Log potential errors retrieving the future's result
                self.logger.error(f"Error retrieving trade fetch result for {symbol}: {e}", exc_info=True)

        if not all_trades_list:
            self.logger.info("No trades found for the specified period.")
            return pd.DataFrame()

        # Concatenate all results
        trades_df = pd.concat(all_trades_list, ignore_index=True)
        # Sort by time for chronological order
        if 'time' in trades_df.columns:
             trades_df = trades_df.sort_values(by='time').reset_index(drop=True)

        self.logger.info(f"Total filled {market_type} trades fetched across all symbols: {len(trades_df)}")
        return trades_df

    def get_trades_for_date(self, symbols: str | List[str], date_str: str, market_type: str = FUTURES_MARKET) -> pd.DataFrame:
        """
        Fetches all filled account trades for a specific date (YYYYMMDD).
        The date is interpreted in the system's local timezone.
        """
        try:
            target_date = datetime.strptime(date_str, DATE_FORMAT).date()
        except ValueError:
            self.logger.error(f"Invalid date format: {date_str}. Please use YYYYMMDD.")
            return pd.DataFrame()

        # Define start and end of the day (00:00:00 to 23:59:59.999)
        start_dt = datetime.combine(target_date, time.min)
        # End is exclusive in many range operations, but Binance endTime is inclusive.
        # So, end of day is just before the start of the next day.
        end_dt = datetime.combine(target_date, time.max).replace(microsecond=999999)

        # Convert to milliseconds timestamp
        start_ts = int(start_dt.timestamp() * 1000)
        end_ts = int(end_dt.timestamp() * 1000)

        self.logger.info(f"Fetching {market_type} trades for date: {date_str}")
        return self.get_trades_for_period(symbols, start_ts, end_ts, market_type=market_type)

    def get_trades_history(
        self,
        symbols: str | List[str],
        start_date_str: str,
        end_date_str: Optional[str] = None,
        market_type: str = FUTURES_MARKET,
    ) -> pd.DataFrame:
        """
        Fetches all filled account trades between a start date and an end date (inclusive),
        iterating one day at a time. If end_date is None, it defaults to the current day.
        Dates are interpreted in the system's local timezone.
        """
        try:
            start_date = datetime.strptime(start_date_str, DATE_FORMAT).date()
        except ValueError:
            self.logger.error(f"Invalid start date format: {start_date_str}. Please use YYYYMMDD.")
            return pd.DataFrame()

        if end_date_str:
            try:
                end_date = datetime.strptime(end_date_str, DATE_FORMAT).date()
            except ValueError:
                self.logger.error(f"Invalid end date format: {end_date_str}. Please use YYYYMMDD.")
                return pd.DataFrame()
        else:
            # Default to today's date (local timezone)
            end_date = datetime.now().date()

        if start_date > end_date:
            self.logger.error(f"Start date ({start_date_str}) cannot be after end date ({end_date_str or 'today'}).")
            return pd.DataFrame()

        market_type = self._normalize_market_type(market_type)
        self.logger.info(f"Starting {market_type} trade history fetch from {start_date} to {end_date} (fetching day by day)...")

        all_daily_trades = []
        current_date = start_date
        total_days = (end_date - start_date).days + 1
        day_count = 0

        while current_date <= end_date:
            day_count += 1
            current_date_str = current_date.strftime(DATE_FORMAT)
            self.logger.info(f"Fetching trades for day {day_count}/{total_days}: {current_date_str}")

            # Use the existing method to get trades for the single day
            try:
                daily_trades_df = self.get_trades_for_date(symbols, current_date_str, market_type=market_type)
                if not daily_trades_df.empty:
                    self.logger.debug(f"Found {len(daily_trades_df)} trades for {current_date_str}")
                    all_daily_trades.append(daily_trades_df)
                else:
                    self.logger.debug(f"No trades found for {current_date_str}")
            except Exception as e:
                 # Log error for the specific day but continue with other days
                 self.logger.error(f"Error fetching trades for {current_date_str}: {e}", exc_info=True)
                 # Optionally add a notification or specific handling here
            else:
                sleep(1)

            # Move to the next day
            current_date += timedelta(days=1)

        if not all_daily_trades:
            self.logger.info("No trades found for the entire specified period.")
            return pd.DataFrame()

        # Concatenate all daily results
        self.logger.info(f"Concatenating results from {len(all_daily_trades)} days...")
        try:
            final_trades_df = pd.concat(all_daily_trades, ignore_index=True)

            # Sort by time for chronological order across all days
            if 'time' in final_trades_df.columns:
                 # Ensure 'time' is datetime before sorting if not already done in get_trades_for_date
                 if not pd.api.types.is_datetime64_any_dtype(final_trades_df['time']):
                      final_trades_df['time'] = pd.to_datetime(final_trades_df['time'], errors='coerce')
                      # Handle potential NaT values if conversion failed for some reason
                      final_trades_df = final_trades_df.dropna(subset=['time'])

                 final_trades_df = final_trades_df.sort_values(by='time').reset_index(drop=True)
            else:
                 self.logger.warning("Could not sort final trades DataFrame: 'time' column not found.")


            self.logger.info(f"Finished {market_type} trade history fetch. Total filled trades found: {len(final_trades_df)}")
            return final_trades_df
        except Exception as e:
            self.logger.error(f"Error during final concatenation or sorting of trade history: {e}", exc_info=True)
            # Decide on return value: maybe return concatenated but unsorted? Or empty DF?
            # Returning empty DF for safety in case concatenation itself fails badly.
            return pd.DataFrame()

    def get_historical_klines(self, ticker, start_date="10 Mar, 2025") -> pd.DataFrame:
        candles = self.binance_client.get_historical_klines(ticker, Client.KLINE_INTERVAL_1HOUR, start_date)
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
        df["date"] = pd.to_datetime(df["timestamp"], unit='ms') - pd.Timedelta(hours=3)
        df = df[df["date"].dt.weekday < 5]  # Select weekdays
        df = df[df["date"].dt.hour == 17]  # Select 5 pm
        df["day"] = df["date"].dt.date
        return df

    def get_futures_acc_balance(self):
        acc_balance_data = self.binance_client.futures_account_balance()
        filtered_balance_data = list(filter(lambda data: data['asset'] in ['USDT', 'USDC'], acc_balance_data))
        return pd.DataFrame.from_records(filtered_balance_data)

    def get_spot_acc_balance(
        self,
        assets: Optional[List[str]] = None,
        non_zero: bool = True,
    ) -> pd.DataFrame:
        account_data = self.binance_client.get_account()
        balances = pd.DataFrame.from_records(account_data.get("balances", []))
        if balances.empty:
            return pd.DataFrame(columns=["asset", "free", "locked", "total"])

        balances["free"] = pd.to_numeric(balances["free"], errors="coerce").fillna(0.0)
        balances["locked"] = pd.to_numeric(balances["locked"], errors="coerce").fillna(0.0)
        balances["total"] = balances["free"] + balances["locked"]

        if assets is not None:
            asset_set = {asset.upper() for asset in assets}
            balances = balances[balances["asset"].isin(asset_set)]
        if non_zero:
            balances = balances[balances["total"] != 0.0]

        return balances.sort_values(by="total", ascending=False).reset_index(drop=True)

    def get_total_margin_balance_usd(self):
        account_summary = self.get_futures_acc_balance()
        if account_summary.empty:
            return 0.0
        return account_summary["balance"].astype(float).sum()

    def get_spot_positions(self, normalize_symbols=False) -> pd.DataFrame:
        balances = self.get_spot_acc_balance(non_zero=True)
        columns = ["symbol", "notional", "positionAmt", "price", "markPrice"]
        if balances.empty:
            return pd.DataFrame(columns=columns)

        positions = []
        for balance in balances.to_dict(orient="records"):
            asset = balance["asset"]
            amount = float(balance["total"])
            symbol = asset
            price = None
            notional = None

            if asset in STABLE_PRICE_ASSETS:
                price = 1.0
                notional = amount
            elif asset in CASH_ASSETS:
                self.logger.info(f"Skipping spot USD pricing for cash asset {asset}.")
            else:
                try:
                    symbol = self._get_order_symbol(asset, SPOT_MARKET)
                    price = self.get_spot_ticker_price(symbol)
                    notional = amount * price
                except Exception as exc:
                    self.logger.warning(f"Could not price spot balance for {asset}: {exc}")

            positions.append({
                "symbol": asset if normalize_symbols else symbol,
                "notional": round(notional, 2) if notional is not None else None,
                "positionAmt": round(amount, 8),
                "price": round(price, 8) if price is not None else None,
                "markPrice": round(price, 8) if price is not None else None,
            })

        df = pd.DataFrame.from_records(positions, columns=columns)
        if "notional" in df.columns:
            df["notional_abs"] = pd.to_numeric(df["notional"], errors="coerce").abs()
            df = df.sort_values(by="notional_abs", ascending=False, na_position="last").drop(columns=["notional_abs"])
        return df.reset_index(drop=True)

    def get_positions(self, normalize_symbols=False, kind='future'):
        market_type = self._normalize_market_type(kind)
        if market_type == SPOT_MARKET:
            return self.get_spot_positions(normalize_symbols=normalize_symbols)

        df = pd.DataFrame.from_records(self.binance_client.futures_position_information())

        if df.empty:
            df = pd.DataFrame(columns=["symbol", "notional", "positionAmt", "price", "markPrice"])
            return df

        df["positionAmt"] = df["positionAmt"].astype(float)
        df["markPrice"] = df["markPrice"].astype(float)
        df = df[df["positionAmt"] != 0.0]

        df["markPrice"] = df["markPrice"].apply(lambda x: round(x, 2))
        df["positionAmt"] = df["positionAmt"].apply(lambda x: round(x, 3))
        df["notional"] = df["notional"].apply(lambda x: round(float(x), 2))
        df["notional_abs"] = df["notional"].abs()
        df = df.sort_values(by="notional_abs", ascending=False)

        if normalize_symbols:
            df["symbol"] = df["symbol"].apply(lambda symbol: symbol[:-4])

        df[["symbol", "notional", "positionAmt", "markPrice"]]
        df['price'] = df['markPrice']
        num_cols = ["notional", "positionAmt", "price", "markPrice"]
        df[num_cols] = df[num_cols].astype(float)
        return df[["symbol", "notional", "positionAmt", "price", "markPrice"]]

    def get_subaccount_transfer_history(self, email: str, futures_type: int = 1, start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        params = {}
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        sub_account_futures_transfer_history = self.binance_client.get_sub_account_futures_transfer_history(email=email, futuresType=futures_type, **params)
        if sub_account_futures_transfer_history['success'] != 'true':
            raise Exception("Error getting subaccount futures transfer history")
        sub_account_futures_transfers = pd.DataFrame.from_records(sub_account_futures_transfer_history["transfers"])
        return sub_account_futures_transfers


if __name__ == "__main__":
    binance = BinanceController()

    # binance.get_etf_basket_tickers_prices()
    #execute_dict = binance.get_execute_dict("HASH11", 1)
    #print(execute_dict)

    #acc_transfer_history = binance.get_subaccount_transfer_history("charmander16_virtual@1vugntkunoemail.com")
    #print(acc_transfer_history)
    # binance.get_trades_history('20250314')

    #print(binance.get_positions())
    pos = binance.get_positions(normalize_symbols=True, kind="spot")
    print(pos)

    # binance.send_market_spot_order("BTC", 0.0004, side="SELL")

