from utils import get_lfts11_price
import pandas as pd
import matplotlib.pyplot as plt
import io

df_closing_prices = get_lfts11_price()
df_deals = pd.read_csv('data/deals_2026-01-07.csv')
df_aum = pd.read_csv('data/df_aum.csv')

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
# deals_csv = """ticket,order,time_msc,type,entry,magic,position_id,reason,volume,price,commission,swap,profit,fee,symbol,comment,external_id,volume_signal
# ,1767704400000,0,0,0,,1,1.0,145.80,0.0,0.0,0.0,0.0,LFTS11,MANUAL,,1.0
# 130945096,256705204,1767790559589,0,0,0,256632122,3,1.0,145.84,0.0,0.0,0.0,0.0,LFTS11,METATRADER5,,1.0
# 130950944,256719011,1767794010377,0,0,0,256632122,3,1.0,145.84,0.0,0.0,0.0,0.0,LFTS11,METATRADER5,,1.0
# 130959944,256750112,1767801239776,0,0,0,256632122,3,3.0,145.84,0.0,0.0,0.0,0.0,LFTS11,METATRADER5,,3.0"""

# prices_csv = """datetime,symbol,open,high,low,close,volume
# 0,2026-01-06 10:00:00,BMFBOVESPA:LFTS11,145.66,145.82,145.66,145.74,235605.0
# 1,2026-01-07 10:00:00,BMFBOVESPA:LFTS11,145.75,145.85,145.74,145.84,72912.0
# 2,2026-01-08 10:00:00,BMFBOVESPA:LFTS11,145.83,145.92,145.83,145.91,67133.0
# 3,2026-01-09 10:00:00,BMFBOVESPA:LFTS11,145.94,146.08,145.94,146.01,108109.0
# 4,2026-01-12 10:00:00,BMFBOVESPA:LFTS11,146.07,146.09,146.01,146.09,78995.0"""

# aum_csv = """date,aum
# 2026-01-06,1000"""

# df_deals = pd.read_csv(io.StringIO(deals_csv))
# df_closing_prices = pd.read_csv(io.StringIO(prices_csv))
# df_aum = pd.read_csv(io.StringIO(aum_csv))

# ---------------------------------------------------------
# 2. DATA PREPROCESSING
# ---------------------------------------------------------
# Deals: Convert msc to datetime date
df_deals['date'] = pd.to_datetime(df_deals['time_msc'], unit='ms').dt.normalize()
df_deals['signed_volume'] = df_deals.apply(lambda x: x['volume'] if x['type'] == 0 else -x['volume'], axis=1)
costs = df_deals['commission'] + df_deals['swap'] + df_deals['fee']
df_deals['cash_flow'] = -(df_deals['signed_volume'] * df_deals['price']) - costs

# Prices: Convert string to datetime date, fix symbol
df_closing_prices['date'] = pd.to_datetime(df_closing_prices['datetime']).dt.normalize()
df_closing_prices['symbol'] = df_closing_prices['symbol'].str.replace('BMFBOVESPA:', '')

# AUM: Convert to datetime date
df_aum['date'] = pd.to_datetime(df_aum['date']).dt.normalize()

# ---------------------------------------------------------
# 3. CALCULATE ABSOLUTE P&L
# ---------------------------------------------------------
unique_symbols = df_closing_prices['symbol'].unique()
pl_frames = []

for sym in unique_symbols:
    asset_deals = df_deals[df_deals['symbol'] == sym]
    asset_prices = df_closing_prices[df_closing_prices['symbol'] == sym].copy()
    
    if asset_prices.empty:
        continue

    # Aggregate deals by Date
    daily_deals = asset_deals.groupby('date')[['signed_volume', 'cash_flow']].sum()
    
    # Merge deals into price history
    df_merged = pd.merge(asset_prices, daily_deals, on='date', how='left')
    df_merged['signed_volume'] = df_merged['signed_volume'].fillna(0)
    df_merged['cash_flow'] = df_merged['cash_flow'].fillna(0)
    
    # Calculate holdings and cash
    df_merged['current_holdings'] = df_merged['signed_volume'].cumsum()
    df_merged['cumulative_cash'] = df_merged['cash_flow'].cumsum()
    
    # Calculate Total Absolute PL
    df_merged['market_value'] = df_merged['current_holdings'] * df_merged['close']
    df_merged['total_pl'] = df_merged['cumulative_cash'] + df_merged['market_value']
    
    pl_frames.append(df_merged[['date', 'total_pl']])

# ---------------------------------------------------------
# 4. CALCULATE PERCENTAGE P&L & PLOT
# ---------------------------------------------------------
if pl_frames:
    # Combine all assets
    df_portfolio = pd.concat(pl_frames)
    portfolio_curve = df_portfolio.groupby('date')['total_pl'].sum().reset_index()
    
    # Merge with AUM Data
    # 'how=left' ensures we keep all dates from the curve
    portfolio_curve = pd.merge(portfolio_curve, df_aum, on='date', how='left')
    
    # Forward Fill AUM (Assumes AUM stays constant until a new entry appears in df_aum)
    portfolio_curve['aum'] = portfolio_curve['aum'].ffill()
    
    # Calculate Percentage
    portfolio_curve['pl_pct'] = (portfolio_curve['total_pl'] / portfolio_curve['aum']) * 100
    
    print("Portfolio PL Data:")
    print(portfolio_curve[['date', 'total_pl', 'aum', 'pl_pct']])

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_curve['date'], portfolio_curve['pl_pct'], marker='o', linestyle='-')
    
    plt.title('BRD Portfolio Cumulative P&L (%)')
    plt.xlabel('Date')
    plt.ylabel('Profit/Loss (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("No matching data found.")