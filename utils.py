from typing import Optional
import requests
import picologging as logging
import datetime
import numpy as np
import pandas as pd
import asyncio
import MetaTrader5 as mt5
import ujson as json
from notifypy import Notify
from datetime import timezone, timedelta
import math
import os
from dotenv import load_dotenv

import tkinter as tk

ALL_TICKERS = [
    "LFTS11",
]


def find_index_ascending(sorted_list: np.ndarray, number: float, step: float) -> Optional[float]:
    if len(sorted_list) == 0:
        return None
    if number < sorted_list[0]:
        return sorted_list[0] - step
    elif number == sorted_list[0]:
        return sorted_list[0]

    for index in range(len(sorted_list) - 1):
        if sorted_list[index] == number:
            return sorted_list[index] - step
        if sorted_list[index] < number < sorted_list[index + 1]:
            return sorted_list[index + 1] - step
    if number > sorted_list[-1]:
        return None
    return None

def find_index_descending(sorted_list: np.ndarray, number: float, step: float) -> Optional[float]:
    if len(sorted_list) == 0:
        return None
    if number > sorted_list[0]:
        return sorted_list[0] + step
    elif number == sorted_list[0]:
        return sorted_list[0]

    for index in range(len(sorted_list) - 1):
        if sorted_list[index] == number:
            return sorted_list[index] + step
        if sorted_list[index] > number > sorted_list[index + 1]:
            return sorted_list[index + 1] + step
    if number < sorted_list[-1]:
        return None
    return None


def adjust_price(target_price, side, df_bid, df_ask, step=0.01, mode='MIXED'):
    mode = mode.upper()
    # this is the default adjust, mixed one
    if side.upper() == 'SELL':

        if df_bid.empty:
            return None

        try:
            variable = target_price <= df_bid["price"].iloc[0] and mode in ['MIXED', 'AGRESSIVE']
        except:
            print(target_price, df_bid["price"], mode)

        if target_price <= df_bid["price"].iloc[0] and mode in ['MIXED', 'AGRESSIVE']:
            # if mixed and target_price is below the bid, tp does not change
            # if passive, return None
            return target_price
        elif mode in ['MIXED', 'PASSIVE']:
            # if mixed, adjust the price to fit in optimal place in the book
            # if agression, return None
            return find_index_ascending(df_ask["price"].values, target_price, step)
        else:
            return None

    elif side.upper() == 'BUY':

        try:
            variable = target_price >= df_ask["price"].iloc[0] and mode in ['MIXED', 'AGRESSIVE']
        except:
            print(target_price, df_bid["price"], mode)

        if df_ask.empty:
            return None

        if target_price >= df_ask["price"].iloc[0] and mode in ['MIXED', 'AGRESSIVE']:
            return target_price
        elif mode in ['MIXED', 'PASSIVE']:
            return find_index_descending(df_bid["price"].values, target_price, step)
        else:
            return None


def is_time_between(begin_time, end_time, check_time=None):
    # If check time is not given, default to current UTC time
    check_time = check_time or datetime.datetime.now().time()
    if begin_time < end_time:
        return check_time >= begin_time and check_time <= end_time
    else: # crosses midnight
        return check_time >= begin_time or check_time <= end_time


def check_tradable(ticker):
    # for ticker in ticker_list:
    if ticker in ALL_TICKERS:
        if is_time_between(datetime.time(9, 0), datetime.time(18, 31)):
            return True
        else:
            return False
    else:
        if is_time_between(datetime.time(10, 5), datetime.time(10, 20)):
            msg = mt5.symbol_info(ticker)
            if msg == None:
                return False
            price_theoretical = msg ._asdict()["price_theoretical"]
            # print(price_theoretical)
            if price_theoretical < -100000:
                return True
            else:
                return False
        else:
            if is_time_between(datetime.time(10, 20), datetime.time(16, 54)):
                return True
            else:
                return False


async def check_tradable_async(ticker):
    if ticker in ALL_TICKERS:
        if is_time_between(datetime.time(9, 0), datetime.time(18, 31)):
            return True
        else:
            return False
    else:
        if is_time_between(datetime.time(10, 5), datetime.time(10, 20)):
            msg = await asyncio.to_thread(mt5.symbol_info, ticker)
            if msg is None:
                return False
            price_theoretical = msg._asdict().get("price_theoretical")
            print(price_theoretical)
            if price_theoretical < -100000:
                return True
            else:
                return False
        else:
            if is_time_between(datetime.time(10, 20), datetime.time(16, 54)):
                return True
            else:
                return False

async def check_tradability_for_tickers(ticker_list):
    tasks = [check_tradable_async(ticker) for ticker in ticker_list]
    results = await asyncio.gather(*tasks)
    return dict(zip(ticker_list, results))


def check_frp_valid(frp0, frp0_last_updated):
    if frp0 is None:
        return False
    else:
        today = datetime.datetime.today().strftime('%Y-%m-%d')
        if today != frp0_last_updated:
            return False
        else:
            return True


def adjust_timestamp(timestamp, hours):
    try:
        ts_ms = float(timestamp)
    except (ValueError, TypeError):
        raise ValueError("The provided timestamp must be numeric (milliseconds).")
    try:
        delta_hours = float(hours)
    except (ValueError, TypeError):
        raise ValueError("The provided hours must be numeric.")

    dt = datetime.datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    delta = timedelta(hours=delta_hours)
    dt_new = dt + delta
    new_timestamp = int(dt_new.timestamp() * 1000)
    return new_timestamp



def push_notification(title: str, message: str, critical=False):

    if critical is True:
        # Create a root window
        root = tk.Tk()

        # Set the title of the window
        root.title(title)

        # Set the size of the pop-up window (width x height)
        root.geometry("400x200")

        # Set the window to appear in the center of the screen
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = (screen_width // 2) - (400 // 2)
        center_y = (screen_height // 2) - (200 // 2)
        root.geometry(f"400x200+{center_x}+{center_y}")

        # Set the background color of the pop-up window
        root.configure(bg="red")

        # Create a label with the warning message
        label = tk.Label(root, text=message, font=("Helvetica", 12), bg="red", fg="white", padx=10, pady=10)
        label.pack(expand=True)

        # Add an OK button to allow the user to close the window
        ok_button = tk.Button(root, text="OK", command=root.destroy, font=("Helvetica", 12), bg="white")
        ok_button.pack(pady=10)

        # Start the GUI event loop to display the pop-up window
        root.mainloop()
    else:
        notification = Notify()
        notification.title = title
        notification.message = message
        notification.send()



def setup_logger(name: str, stdout: bool = True, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up and configure a logger with the specified name, output options, and log level.

    Args:
        name (str): The name of the logger.
        stdout (bool): If True, log to standard output. Default is True.
        log_file (Optional[str]): Path to a log file. If provided, logs will be written to this file. Default is None.
        level (int): Logging level. Default is logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    if stdout:
        stdout_handler = logging.StreamHandler()
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger



