from tvDatafeed import TvDatafeed, Interval
import sys
import logging
import argparse

sys.path.append(".")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TVData:
    def __init__(self, ticker="HASH11", exchange="BMFBOVESPA", interval=60*24) -> None:
        self.tv = TvDatafeed("lucambottino", "NDTTr@d1ngTechnologies")

        # Set interval
        self.inverval = {
            60*24: Interval.in_daily,
            60: Interval.in_1_hour,
            30: Interval.in_30_minute,
            15: Interval.in_15_minute,
            5: Interval.in_5_minute,
            3: Interval.in_3_minute,
            1: Interval.in_1_minute,
        }.get(interval, Interval.in_5_minute)  # Default to 5-minute interval

        self.ticker = ticker
        self.exchange = exchange

    def get_price(self, n_bars=28):
        ticker_price = self.tv.get_hist(
            symbol=self.ticker,
            exchange=self.exchange,
            interval=self.inverval,
            n_bars=n_bars,
        ).reset_index()

        return ticker_price


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the premium calculator server.")

    parser.add_argument(
        "--n_bars", type=int, default=100, help="Number of bars to fetch."
    )
    parser.add_argument(
        "--interval", type=int, default=60*24, help="Interval for fetching data."
    )
    parser.add_argument(
        "--ticker", type=str, default="HASH11", help="Ticker to fetch data for."
    )
    parser.add_argument(
        "--exchange", type=str, default="BMFBOVESPA", help="Exchange to fetch data for."
    )
    args = parser.parse_args()

    tvdata = TVData(ticker=args.ticker, exchange=args.exchange, interval=args.interval)
    price = tvdata.get_price(n_bars=args.n_bars)
    print(price)