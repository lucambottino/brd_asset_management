from typing import Any, List, Optional
import MetaTrader5 as mt5
import os
import pytz
import pandas as pd
from dotenv import load_dotenv
from dateutil.relativedelta import relativedelta
import concurrent.futures
from datetime import timedelta, datetime, time
import numpy as np


from singleton_meta import SingletonMeta
from utils import setup_logger

BID_INDEX = 2
ASK_INDEX = 1

class Broker(metaclass=SingletonMeta):

    def __init__(self):
        load_dotenv()
        # mt5.shutdown()

        # Define your MT5 account credentials
        self.server_name = str(os.getenv('MT5SERVER'))
        self.account_number = int(str(os.getenv('METATRADEREXECUTIONLOGIN')))#int(str(os.getenv('MT5SUMOUSER')))
        self.account_password = str(os.getenv('METATRADEREXECUTIONPASSWORD'))#str(os.getenv('MT5SUMOPASSWORD'))

        self.ticker_list = ["LFTS11"]

        # self.whitelisted_tags = str(os.getenv("BROKER_TAGS")).split(",")
        # self.blacklisted_tags = str(os.getenv("BROKER_TAGS_BLACKLIST")).split(",")
        self.whitelisted_tags = ["METATRADER5"]
        self.blacklisted_tags = []

        self.ticker_subscriptions = []

        self.logger = setup_logger(name=__name__, stdout=True, log_file='broker_log.log')

        # Initialize connection to MT5
        if not mt5.initialize(login=self.account_number, password=self.account_password, server=self.server_name):
            last_error = mt5.last_error()
            self.logger.error(f"initialize() failed, error code = {last_error}")
            quit()
        else:
            print("Successfully logged in!")

        '''
        account_info=mt5.account_info()
        print("Account info:")
        print(account_info)
        if account_info!=None:
            # display trading account data 'as is'
            print(account_info)
            # display trading account data in the form of a dictionary
            print("Show account_info()._asdict():")
            account_info_dict = mt5.account_info()._asdict()
            for prop in account_info_dict:
                print("  {}={}".format(prop, account_info_dict[prop]))
            print()
    
            # convert the dictionary into DataFrame and print
            df=pd.DataFrame(list(account_info_dict.items()),columns=['property','value'])
            print("account_info() as dataframe:")
            print(df)
        '''

    def get_info(self):
        # Retrieve some account info as a test
        account_info = mt5.account_info()
        print(account_info)
        if account_info is not None:
            print("Account balance:", account_info.balance)
        else:
            print("Failed to retrieve account info.")
        return account_info


    def get_acc_position(self):
        positions_total = mt5.positions_get()
        df=pd.DataFrame(list(positions_total),columns=positions_total[0]._asdict().keys())
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.drop(['time_update', 'time_msc', 'time_update_msc', 'external_id'], axis=1, inplace=True)
        df['total'] = df['price_current'] * df['volume']
        return df


    def convert_timestamp_to_sao_paulo_time(self, timestamp):
        try:
            utc_time = datetime.fromtimestamp(timestamp, pytz.utc)
            sao_paulo_tz = pytz.timezone('America/Sao_Paulo')
            sao_paulo_time = utc_time.astimezone(sao_paulo_tz)
            sao_paulo_time = sao_paulo_time + relativedelta(hours=3)
            return sao_paulo_time
        except Exception as e:
            # Handle errors and log them
            print(f"Error: {e}")


    def get_ticker_price(self, ticker) -> dict[str, Any]:
        try:
            tick = mt5.symbol_info_tick(ticker)

            d = {
                "ticker": ticker,
                "timestamp": str(tick.time),
                "date": self.convert_timestamp_to_sao_paulo_time(tick.time),
                "last": tick.last,
                "bid": tick.bid,
                "ask": tick.ask
            }
            return d

        except Exception as e:
            print(f"Error: {e} for ticker {ticker}")
            raise e
        

    def get_recent_prices(self):
        d = {}
        for ticker in self.ticker_list:
            tick = mt5.symbol_info_tick(ticker)
            last = tick.last
            d[ticker] = last
        return d


    def get_theoretical_price(self, ticker):
        tick = mt5.symbol_info(ticker)
        theoretical_price = float(tick.price_theoretical)
        return theoretical_price


    def send_market_order(self, symbol, side, volume, magic=0):
        info = mt5.symbol_info_tick(symbol)
        if side.upper() == "LONG" or side.upper() == "BUY" or side.upper() == "C":
            return self._send_order(symbol=symbol, side=side, volume=volume, price=float(info.ask)+0.1, magic=0)
        elif side.upper() == "SHORT" or side.upper() == "SELL" or side.upper() == "V":
            return self._send_order(symbol=symbol, side=side, volume=volume, price=float(info.bid)-0.1, magic=0)


    def get_order(self, order_id):
        return mt5.orders_get(
            ticket=order_id
        )


    def get_info(self, ticker):
        msg = mt5.symbol_info(ticker)
        return msg._asdict()


    def get_number_of_deals(self, ticker):
        info_dict = self.get_info(ticker)
        return info_dict["session_deals"]


    def _send_order(self, symbol, side, volume, price, magic=0):

        print(f"send order: symbol {symbol}, side {side}, volume {volume}")
        if side.upper() == "LONG" or side.upper() == "BUY" or side.upper() == "C":
            trade_type = mt5.ORDER_TYPE_BUY_LIMIT
        elif side.upper() == "SHORT" or side.upper() == "SELL" or side.upper() == "V":
            trade_type = mt5.ORDER_TYPE_SELL_LIMIT
        else:
            print("Invalid side")

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": float(volume),
            "type": trade_type,
            "price": float(price),
            "magic": magic,
            "comment": "METATRADER5",
            "type_time": mt5.ORDER_TIME_DAY,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }

        try:
            result = mt5.order_send(request)
            self.logger.debug(f"{result}")
            return result
        except Exception as e:
            self.logger.error(f"Error sending order: {e}")
            return None


    def _modify_order(self, order_id, magic, ticker, new_price):

        request = {
            "action": mt5.TRADE_ACTION_MODIFY,
            "order": int(order_id),
            "symbol": ticker,
            "price": float(new_price),
            "magic": int(magic),
            "comment": "METATRADER5",
            "type_time": mt5.ORDER_TIME_DAY,
            "type_filling": mt5.ORDER_FILLING_RETURN
        }

        try:
            result = mt5.order_send(request)
            return result
        except Exception as e:
            self.logger.error(f"Error modifying order: {e}")
            return None


    def cancel_all_orders(self, symbols: Optional[List[str]]=None):
        if isinstance(symbols, list):
            orders = mt5.orders_get(group=",".join([f"*{element}*" for element in symbols]))
        else:
            orders = mt5.orders_get()

        for order in orders:
            try:
                mt5.order_send({
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": order.ticket
                })
                print(f"Order {order.ticket} cancelled.")
            except Exception as e:
                self.logger.error(f"Error in order cancellation: {e}")
        print("All orders cancelled.")


    def cancel_order(self, order_id):
        try:
            mt5.order_send({
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": order_id
            })
            print(f"Order {order_id} cancelled.")
        except Exception as e:
            self.logger.error(f"Error in order cancellation: {e}")


    def get_top_of_book(self, ticker):
        if mt5.market_book_add(ticker):
            book = mt5.market_book_get(ticker)
            bids, asks = [], []
            for i in book:
                if i.type == 1:
                    asks.append(i)
                else:
                    bids.append(i)

            best_bid = max(bids, key=lambda x: x.price)
            best_ask = min(asks, key=lambda x: x.price)
            return best_bid.price, best_ask.price
        else:
            None

    def get_mid_price(self, ticker):
        prices = self.get_top_of_book(ticker)
        if prices is not None:
            best_bid, best_ask = prices
            return (best_bid + best_ask)/2
        return None


    def get_active_orders(self):
        orders = mt5.orders_get()

        data = []
        for order_response in orders:
            order_dict = {
                "ticket": order_response.ticket,
                "time_setup": order_response.time_setup,
                "time_setup_msc": order_response.time_setup_msc,
                "time_done": order_response.time_done,
                "time_done_msc": order_response.time_done_msc,
                "time_expiration": order_response.time_expiration,
                "type": order_response.type,
                "type_time": order_response.type_time,
                "type_filling": order_response.type_filling,
                "state": order_response.state,
                "magic": order_response.magic,
                "position_id": order_response.position_id,
                "position_by_id": order_response.position_by_id,
                "reason": order_response.reason,
                "volume_initial": order_response.volume_initial,
                "volume_current": order_response.volume_current,
                "price_open": order_response.price_open,
                "sl": order_response.sl,
                "tp": order_response.tp,
                "price_current": order_response.price_current,
                "price_stoplimit": order_response.price_stoplimit,
                "symbol": order_response.symbol,
                "comment": order_response.comment,
                "external_id": order_response.external_id
            }
            data.append(order_dict)
        df = pd.DataFrame.from_records(data)
        if df.empty == False:
            return df[df["comment"] == "METATRADER5"]
        else:
            return pd.DataFrame(columns=["ticket", "time_setup", "time_setup_msc", "time_done", "time_done_msc", "time_expiration", "type", "type_time", "type_filling", "state", "magic",
                    "position_id", "position_by_id", "reason", "volume_initial", "volume_current", "price_open", "sl", "tp", "price_current", "price_stoplimit", "symbol", "comment", "external_id"])


    def get_active_orders_records(self):
        orders = mt5.orders_get()

        data = []
        for order_response in orders:
            if order_response.comment == "METATRADER":
                order_dict = {
                    "ticket": order_response.ticket,
                    "time_setup": order_response.time_setup,
                    "time_setup_msc": order_response.time_setup_msc,
                    "time_done": order_response.time_done,
                    "time_done_msc": order_response.time_done_msc,
                    "time_expiration": order_response.time_expiration,
                    "type": order_response.type,
                    "type_time": order_response.type_time,
                    "type_filling": order_response.type_filling,
                    "state": order_response.state,
                    "magic": order_response.magic,
                    "position_id": order_response.position_id,
                    "position_by_id": order_response.position_by_id,
                    "reason": order_response.reason,
                    "volume_initial": order_response.volume_initial,
                    "volume_current": order_response.volume_current,
                    "price_open": order_response.price_open,
                    "sl": order_response.sl,
                    "tp": order_response.tp,
                    "price_current": order_response.price_current,
                    "price_stoplimit": order_response.price_stoplimit,
                    "symbol": order_response.symbol,
                    "comment": order_response.comment,
                    "external_id": order_response.external_id
                }
                data.append(order_dict)
        return data


    def get_deals_period(self, from_date_str):
        from_date = datetime.combine(datetime.strptime(from_date_str, '%Y%m%d'), time.min)
        to_date = datetime.now()

        all_orders = mt5.history_deals_get(from_date, to_date)

        data = []
        for order_response in all_orders:
            order_dict = {
                "ticket": order_response.ticket,
                "order": order_response.order,
                "time": order_response.time,
                "time_msc": order_response.time_msc,
                "type": order_response.type,
                "entry": order_response.entry,
                "magic": order_response.magic,
                "position_id": order_response.position_id,
                "reason": order_response.reason,
                "volume": order_response.volume,
                "price": order_response.price,
                "commission": order_response.commission,
                "swap": order_response.swap,
                "profit": order_response.profit,
                "fee": order_response.fee,
                "symbol": order_response.symbol,
                "comment": order_response.comment,
                "external_id": order_response.external_id
            }
            data.append(order_dict)
        df = pd.DataFrame.from_records(data)

        if df.empty == False:
            df["volume_signal"] = [volume if type_value == 0 else -volume for type_value, volume in zip(df["type"], df["volume"])]
            df['time'] = pd.to_datetime(df['time'], unit='s')
            filtered_df = df

            df_ = filtered_df[filtered_df["comment"] == "METATRADER5"]
            df_ = df_.copy()
            df_.drop('time', axis=1, inplace=True)
            return df_
        else:
            return pd.DataFrame(columns=["ticket", "order", "time_msc", "type", "entry", "magic", "position_id", "reason", "volume", "price", "commission", "swap", "profit", "fee", "symbol", "comment", "external_id", "volume_signal"])


    def get_deals(self, days_back=0):
        """
            #TODO: Check if there are optimizations to be made by removing the storage and processing of unused fieds from the return of mt5.history_deals_get to improve speed and memory efficiency
        """
        start_of_day = datetime.now().replace(hour=1, minute=0, second=0, microsecond=0)
        if days_back > 0:
            start_of_day = start_of_day - timedelta(days=days_back)
        orders = mt5.history_deals_get(start_of_day, datetime.now(), group=",".join([f"*{element}*" for element in self.ticker_list]))

        data = []
        for order_response in orders:
            order_dict = {
                "ticket": order_response.ticket,
                "order": order_response.order,
                "time": order_response.time,
                "time_msc": order_response.time_msc,
                "type": order_response.type,
                "entry": order_response.entry,
                "magic": order_response.magic,
                "position_id": order_response.position_id,
                "reason": order_response.reason,
                "volume": order_response.volume,
                "price": order_response.price,
                "commission": order_response.commission,
                "swap": order_response.swap,
                "profit": order_response.profit,
                "fee": order_response.fee,
                "symbol": order_response.symbol,
                "comment": order_response.comment,
                "external_id": order_response.external_id
            }
            data.append(order_dict)
        df = pd.DataFrame.from_records(data)

        if df.empty == False:
            df["volume_signal"] = [volume if type_value == 0 else -volume for type_value, volume in zip(df["type"], df["volume"])]
            df['time'] = pd.to_datetime(df['time'], unit='s')
            today = datetime.now().date()
            if days_back == 0:
                filtered_df = df[df['time'].dt.date == today]
            else:
                filtered_df = df


            df_ = filtered_df[filtered_df["comment"] == "METATRADER5"]
            df_ = df_.copy()
            df_.drop('time', axis=1, inplace=True)
            return df_
        else:
            return pd.DataFrame(columns=["ticket", "order", "time_msc", "type", "entry", "magic", "position_id", "reason", "volume", "price", "commission", "swap", "profit", "fee", "symbol", "comment", "external_id", "volume_signal"])

    def get_deals_tags(self, from_ts:int=None):
        """
            #TODO: Check if there are optimizations to be made by removing the storage and processing of unused fieds from the return of mt5.history_deals_get to improve speed and memory efficiency
        """

        if not from_ts:
            ts = datetime.now().replace(hour=1, minute=0, second=0, microsecond=0)
        else:
            ts = datetime.fromtimestamp((from_ts + 1) / 1000)

        orders = mt5.history_deals_get(ts, datetime.now(), group=",".join([f"*{element}*" for element in self.ticker_list]))

        data = []
        for order_response in orders:
            order_dict = {
                "ticket": order_response.ticket,
                "order": order_response.order,
                "time": order_response.time,
                "time_msc": order_response.time_msc, # WARNING: mt5.history_deals_get time_msc col is in miliseconds at brazil time not UTC, which is non-standard
                "type": order_response.type,
                "entry": order_response.entry,
                "magic": order_response.magic,
                "position_id": order_response.position_id,
                #"reason": order_response.reason,
                "volume": order_response.volume,
                "price": order_response.price,
                #"commission": order_response.commission,
                #"swap": order_response.swap,
                "profit": order_response.profit,
                #"fee": order_response.fee,
                "symbol": order_response.symbol,
                "comment": order_response.comment,
                "external_id": order_response.external_id
            }
            data.append(order_dict)
        df: pd.DataFrame = pd.DataFrame.from_records(data)

        if df.empty == False:
            df["volume_signal"] = [volume if type_value == 0 else -volume for type_value, volume in zip(df["type"], df["volume"])]
            df['time'] = pd.to_datetime(df['time'], unit='s')


            # filter orders whose comment does not contain any of the tags substring
            df_ = df[~df["comment"].str.contains("|".join(self.blacklisted_tags))]
            # filter orders whose comment that contains any of the tags substring
            df_ = df_[df_["comment"].str.contains("|".join(self.whitelisted_tags))]

            df_ = df_.copy()
            df_.drop(['time'], axis=1, inplace=True)
            #df_['time_msc'] = pd.to_datetime(df_['time_msc'], unit='ms')
            return df_
        else:
            return pd.DataFrame(columns=["ticket", "order", "time_msc", "type", "entry", "magic", "position_id", "volume", "price", "profit", "symbol", "comment", "external_id", "volume_signal"])


    def get_wdo_deals(self, days_back=0):
        """
            #TODO: Check if there are optimizations to be made by removing the storage and processing of unused fieds from the return of mt5.history_deals_get to improve speed and memory efficiency
        """
        start_of_day = datetime.now().replace(hour=1, minute=0, second=0, microsecond=0)

        if days_back > 0:
            start_of_day = start_of_day - timedelta(days=days_back)

        orders = mt5.history_deals_get(start_of_day, datetime.now(), group=",".join([f"*{element}*" for element in [WDO_TICKER, WDO_TICKER_NEXT]]))

        data = []
        for order_response in orders:
            order_dict = {
                "ticket": order_response.ticket,
                "order": order_response.order,
                "time": order_response.time,
                "time_msc": order_response.time_msc, # WARNING: mt5.history_deals_get time_msc col is in miliseconds at brazil time not UTC, which is non-standard
                "type": order_response.type,
                "entry": order_response.entry,
                "magic": order_response.magic,
                "position_id": order_response.position_id,
                #"reason": order_response.reason,
                "volume": order_response.volume,
                "price": order_response.price,
                #"commission": order_response.commission,
                #"swap": order_response.swap,
                "profit": order_response.profit,
                #"fee": order_response.fee,
                "symbol": order_response.symbol,
                "comment": order_response.comment,
                "external_id": order_response.external_id
            }
            data.append(order_dict)
        df: pd.DataFrame = pd.DataFrame.from_records(data)

        if df.empty == False:
            df["volume_signal"] = [volume if type_value == 0 else -volume for type_value, volume in zip(df["type"], df["volume"])]
            df['time'] = pd.to_datetime(df['time'], unit='s')
            today = datetime.now().date()

            if days_back == 0:
                filtered_df = df[df['time'].dt.date == today]
            else:
                filtered_df = df

            # filter orders whose comment that contains any of the tags substring
            df_ = filtered_df[filtered_df["comment"].str.contains("|".join(self.whitelisted_tags))]
            df_ = df_.copy()
            df_.drop(['time'], axis=1, inplace=True)
            return df_
        else:
            return pd.DataFrame(columns=["ticket", "order", "time_msc", "type", "entry", "magic", "position_id", "volume", "price", "profit", "symbol", "comment", "external_id", "volume_signal"])


    def get_executed_qty_dict(self):
        df_deals = self.get_deals()
        return df_deals[["symbol", "volume_signal"]].groupby("symbol").sum()["volume_signal"].to_dict()


    def get_nav_prices(self):
        date_str = datetime.now().strftime("%Y-%m-%d")
        df_nav = pd.read_csv(f"data/nav_{date_str}.csv")
        df_nav.sort_values(['ticker', 'date'], inplace=True)
        result = df_nav.drop_duplicates(subset='ticker', keep='last')
        nav_dict = result.set_index('ticker')['nav_price'].to_dict()
        return nav_dict

    def maybe_add_book_subscription(self, ticker) -> bool:
        if ticker not in self.ticker_subscriptions:
            mt5.market_book_release(ticker) # release in case it was already subscribed without our knowledge
            now_subscribed = mt5.market_book_add(ticker)
            if now_subscribed:
                self.ticker_subscriptions.append(ticker)
            return now_subscribed
        return True

    def maybe_remove_book_subscription(self, ticker) -> bool:
        if ticker not in self.ticker_subscriptions:
            return False
        else:
            self.ticker_subscriptions.remove(ticker)
            mt5.market_book_release(ticker)
            print(f"Removed book subscription for {ticker}")
            return True

    def cleanup(self) -> None:
        for ticker in self.ticker_subscriptions:
            self.maybe_remove_book_subscription(ticker)
        print("Book subscriptions cleaned up")

    def format_book_alt(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_pivoted = df.pivot_table(
            index="price", columns="side", values="qty", aggfunc="sum", fill_value=None
        )
        df_bid: pd.DataFrame = df_pivoted[BID_INDEX].dropna().sort_index(ascending=False, kind="stable").reset_index()
        df_ask: pd.DataFrame = df_pivoted[ASK_INDEX].dropna().sort_index(ascending=True, kind="stable").reset_index()
        return df_bid.rename(columns={BID_INDEX: "qty"}), df_ask.rename(columns={ASK_INDEX: "qty"})

    def get_book_alt(self, ticker) -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
        if ticker is None:
            print("Warning: ticker is None")
            return None
        if self.maybe_add_book_subscription(ticker):
            items = mt5.market_book_get(ticker)
            df = pd.DataFrame.from_dict(items)
            if df.empty:
                print(f"Warning: empty book for {ticker}")
                return None
            df.columns = ["side", "price", "qty", "qty_float"]
            try:
                book = self.format_book_alt(df)
                return book
            except Exception as e:
                print(e)
                return None
        else:
            print(f"mt5.market_book_add({ticker}) failed, error code =",mt5.last_error())
            return None


    def format_book(self, df: pd.DataFrame):
        df_groupped = df.groupby(by=["price", "side"]).sum()[["qty"]].reset_index()
        df_groupped["side"] = df_groupped["side"].astype(int)
        df_pivoted = pd.pivot_table(df_groupped, index="price", columns="side", values="qty")

        df_ask = df_pivoted[[1]].dropna().reset_index()
        df_ask.columns = ["price", "qty"]
        df_bid = df_pivoted[[2]].dropna().reset_index()
        df_bid.columns = ["price", "qty"]

        return df_bid, df_ask

    def get_book(self, ticker):
        if mt5.market_book_add(ticker):
            items = mt5.market_book_get(ticker)
            mt5.market_book_release(ticker)
            df = pd.DataFrame.from_dict(items)
            if df.empty:
                return None, None

            try:
                df.columns = ["side", "price", "qty", "qty_float"]
                df["qty"] = df["qty"].apply(int)

                df_bid, df_ask = self.format_book(df)
            except Exception as e:
                print(e)
                return None, None

            return df_bid.sort_values("price", ascending=False), df_ask.sort_values("price", ascending=True)
        else:
            print(f"mt5.market_book_add({ticker}) failed, error code =",mt5.last_error())
            return None, None


    def get_book_batch(self, ticker_list):
        result = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {executor.submit(self.get_book_alt, ticker): ticker for ticker in ticker_list}
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    fair_price = future.result()
                    result[ticker] = fair_price
                except Exception as exc:
                    print(f'{ticker} generated an exception: {exc}')
        return result
    
    def get_info(self, ticker):
        msg = mt5.symbol_info(ticker)
        return msg._asdict()


    def shutdown(self):
        mt5.shutdown()


def get_symbol_info(symbol):
    # get the list of orders on symbols whose names contain "*symbol*"
    order_content=mt5.orders_get(group=f"*{symbol}*")
    if order_content is None:
        # print("No orders with group=\"*{}*\", error code={}".format(symbol, mt5.last_error()))
        return None
    else:
        # print("orders_get(group=\"*{}*\")={}".format(symbol, len(order_content)))

        df=pd.DataFrame(list(order_content),columns=order_content[0]._asdict().keys())
        # df.drop(['time_done', 'time_done_msc', 'position_id', 'position_by_id', 'reason', 'volume_initial', 'price_stoplimit'], axis=1, inplace=True)
        df['time_setup'] = pd.to_datetime(df['time_setup'], unit='s')
        return df
    
    



if __name__ == "__main__":
    broker = Broker()
    info = mt5.symbol_info_tick("LFTS11")
    print(info)
    # # print(broker.get_acc_position())
    # broker._send_order(symbol="LFTS11", side="BUY", volume=1, price=float(info.ask)+0.1, magic=0)
    # print(broker.send_market_order("LFTS11", "BUY", 1))
    # print(broker.get_acc_position())
    # print(broker.get_deals_tags())
    # print(broker.get_info())

    # print(broker.get_active_orders())
    # print(broker.get_deals())

    # account_info=mt5.account_info()
    # print("Account info:")
    # print(account_info)
    # if account_info!=None:
    #     # display trading account data 'as is'
    #     print(account_info)
    #     # display trading account data in the form of a dictionary
    #     print("Show account_info()._asdict():")
    #     account_info_dict = mt5.account_info()._asdict()
    #     for prop in account_info_dict:
    #         print("  {}={}".format(prop, account_info_dict[prop]))
    #     print()

    #     # convert the dictionary into DataFrame and print
    #     df=pd.DataFrame(list(account_info_dict.items()),columns=['property','value'])
    #     print("account_info() as dataframe:")
    #     print(df)

    available_cash = 853.94

    # print(broker.get_acc_position())
    print(broker.get_deals())

    # df_pos = broker.get_acc_position()
    df_deals = broker.get_deals()
