import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.ticker as mtick

# -----------------------------
# INPUT DATA
# -----------------------------
df_selic = pd.read_excel("selic.xlsx", sheet_name="Clean")
df_usdbrl = pd.read_excel("selic.xlsx", sheet_name="PTAX")
df_us_treasury = pd.read_excel("selic.xlsx", sheet_name="USTREASURY")

# -----------------------------
# PREPARE SELIC DATA
# -----------------------------
df_selic["period_begin"] = pd.to_datetime(df_selic["period_begin"], dayfirst=True)
df_selic["period_end"] = pd.to_datetime(df_selic["period_end"], dayfirst=True)

df_selic["interest_annualized"] = (
    df_selic["interest_annualized"]
    .astype(str)
    .str.replace(",", ".")
    .astype(float) / 100
)

# -----------------------------
# PREPARE US TREASURY DATA
# -----------------------------
df_us_treasury["date"] = pd.to_datetime(df_us_treasury["date"])

df_us_treasury.rename(columns={"percentage": "us_treasury"}, inplace=True)

df_us_treasury["us_treasury"] = df_us_treasury["us_treasury"] / 100  # convert to decimal

df_us_treasury = df_us_treasury.sort_values("date").set_index("date")

# Convert annualized yield -> daily rate (252 convention)
df_us_treasury["ust_daily_rate"] = (1 + df_us_treasury["us_treasury"])**(1/252) - 1

# Expand monthly -> daily (business days)
df_us_treasury = df_us_treasury.resample("B").ffill()


# -----------------------------
# BUILD DAILY SELIC SERIES
# -----------------------------
daily_rates = []

for _, row in df_selic.iterrows():

    start = row["period_begin"]
    end = row["period_end"]

    if pd.isna(end):
        end = pd.Timestamp.today()

    dates = pd.date_range(start, end, freq="B")

    annual_rate = row["interest_annualized"]

    daily_rate = (1 + annual_rate)**(1/252) - 1

    tmp = pd.DataFrame({
        "date": dates,
        "brl_rate": daily_rate,
        "interest_annualized": row["interest_annualized"]
    })

    daily_rates.append(tmp)

df_daily_selic = pd.concat(daily_rates).drop_duplicates("date")
df_daily_selic = df_daily_selic.sort_values("date").set_index("date")


# df_selic_nominal = df_selic


# -----------------------------
# PREPARE FX DATA
# -----------------------------
df_usdbrl["date"] = pd.to_datetime(df_usdbrl["date"])
df_usdbrl = df_usdbrl.set_index("date").sort_index()

# Merge everything
df = df_daily_selic.join(df_usdbrl["close"], how="inner")
df = df.rename(columns={"close": "usdbrl"})

# Join treasury
df = df.join(df_us_treasury, how="left")

# Forward fill treasury (important for alignment)
df["ust_daily_rate"] = df["ust_daily_rate"].ffill()

df = df[df.index > datetime(2000, 1, 1)]

df.to_csv("df.csv")

# -----------------------------
# SELIC CARRY IN USD
# -----------------------------
df["selic_brl"] = (1 + df["brl_rate"]).cumprod()
df["selic_usd"] = (1 / df["usdbrl"]) * df["selic_brl"]

df["fx_ratio"] = df["usdbrl"] / df["usdbrl"].shift(-1)

df["usd_return"] = (1 + df["brl_rate"]) * df["fx_ratio"] - 1
df["usd_growth"] = (1 + df["usd_return"]).cumprod()

# -----------------------------
# US TREASURY CUMULATIVE RETURN
# -----------------------------
df["ust_growth"] = (1 + df["ust_daily_rate"]).cumprod()

# -----------------------------
# NORMALIZE BOTH SERIES (START = 0%)
# -----------------------------
df["usd_growth_norm"] = df["usd_growth"] / df["usd_growth"].iloc[0] - 1
df["ust_growth_norm"] = df["ust_growth"] / df["ust_growth"].iloc[0] - 1



df_inflation = pd.read_csv("inflation.csv")
df_inflation["date"] = pd.to_datetime(df_inflation["date"])
df_inflation = df_inflation.set_index("date")

# -----------------------------
# PLOTS
# -----------------------------
fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(1,1,1)

plt.plot(df.index, 100 * df["interest_annualized"], label="Brazil Treasury Nominal Rate",
    color="blue",
    linewidth=2)
plt.plot(df.index, 100 * df["us_treasury"], label="US Treasury Nominal Rate", color="orange",
    linewidth=2)

plt.plot(df_inflation.index, df_inflation["br_inflation_yoy_pct"], label="Brazil Inflation YoY (%)",     color="blue",
    linestyle="--",
    alpha=0.7)

plt.plot(df_inflation.index, df_inflation["us_inflation_yoy_pct"], label="US Inflation YoY (%)",     color="orange",
    linestyle="--",
    alpha=0.7)


plt.title("Brazilian Treasury vs US Treasury (Nominal Rate)")
plt.ylabel("Yearlly Interest (% in local currency)")
plt.xlabel("Date")

ax.yaxis.set_major_formatter(mtick.PercentFormatter())

plt.legend()
plt.grid(True)
plt.show()

df = df.join(df_inflation, how="outer").ffill()
df.to_csv("df.csv")




fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(1,1,1)

plt.plot(df.index, 100 * df["usd_growth_norm"], label="Brazil Treasury (USD)")
plt.plot(df.index, 100 * df["ust_growth_norm"], label="US Treasury (USD)")

plt.title("Brazil Treasury in USD vs US Treasury (Cumulative Return)")
plt.ylabel("Return (% in USD)")
plt.xlabel("Date")

ax.yaxis.set_major_formatter(mtick.PercentFormatter())

plt.legend()
plt.grid(True)
plt.show()

