from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Dict, Iterable, Tuple, Literal

import numpy as np
import pandas as pd


# =========================
# Utilities / Validation
# =========================

def _require_cols(df: pd.DataFrame, cols: Iterable[str], df_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing columns: {missing}")

def _as_utc(ts: pd.Series) -> pd.Series:
    # Accept naive timestamps; interpret as UTC. You can adapt to specific TZ policy.
    s = pd.to_datetime(ts, utc=True, errors="raise")
    return s

def _normalize_side(side: pd.Series) -> pd.Series:
    s = side.astype(str).str.lower().str.strip()
    s = s.replace({"buy": "buy", "b": "buy", "long": "buy",
                   "sell": "sell", "s": "sell", "short": "sell"})
    bad = ~s.isin(["buy", "sell"])
    if bad.any():
        raise ValueError(f"Invalid side values (expected buy/sell). Examples: {side[bad].head(5).tolist()}")
    return s

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0:
        return default
    return a / b

def _round_money(x: pd.Series, decimals: int = 2) -> pd.Series:
    return x.round(decimals)

def _round_shares(x: pd.Series, decimals: int = 8) -> pd.Series:
    return x.round(decimals)


# =========================
# Data Schemas (Expected)
# =========================
"""
FUND FLOWS DF (subscriptions/redemptions)
----------------------------------------
Required columns:
- ts: datetime-like
- amount: float (cash amount)
- direction: 'deposit' or 'withdrawal'
Optional:
- currency: str
- user_id: str (if flows are user-attributed; not required for fund-level only)

Convention:
- amount MUST be positive.
- direction determines sign into fund:
    deposit    -> +cash
    withdrawal -> -cash

TRADES DF
---------
Required columns:
- ts: datetime-like
- instrument: str
- qty: float (positive)
- side: buy/sell
- price: float (trade price in base currency)
Optional:
- fee: float (>=0), in base currency (reduces cash)
- multiplier: float (defaults 1.0; for derivatives)
- venue_trade_id, etc.

PRICES DF (marks)
-----------------
Required columns:
- ts: datetime-like (mark timestamp)
- instrument: str
- price: float (mark price)
Optional:
- multiplier: float (if not in trades; default 1.0)

USERS BALANCE DF (assets/liabilities snapshot)
----------------------------------------------
Required columns:
- user_id: str
- total_assets: float
- total_liabilities: float
Optional:
- ts: datetime-like (if multiple snapshots)

We compute:
net_worth = total_assets - total_liabilities
"""


# =========================
# Fund Accounting Engine
# =========================

@dataclass(frozen=True)
class FundAccountingConfig:
    base_currency: str = "USD"
    shares_decimals: int = 8
    money_decimals: int = 2

    mark_method: Literal["last", "ffill"] = "ffill"
    require_marks: bool = True

    # NEW:
    backfill_initial_marks: bool = True   # allow using first available mark for times before first mark
    trade_price_fallback_mark: bool = True  # if still missing, synthesize a mark from the trade price

    initial_share_price: float = 1.0
    allow_negative_nav: bool = False



class FundLedger:
    """
    Computes NAV and quota (share price) over time from:
      - external cash flows (deposits/withdrawals)
      - trades (cash impact + positions)
      - market prices for marking positions

    Output: a time series ledger with:
      - cash
      - position MV
      - nav
      - shares_outstanding
      - share_price (quota)
      - flow_shares_issued/redeemed
    """

    def __init__(
        self,
        config: FundAccountingConfig,
        flows: pd.DataFrame,
        trades: pd.DataFrame,
        prices: pd.DataFrame,
        initial_cash: float = 0.0,
        initial_shares: float = 0.0,
    ):
        self.cfg = config

        self.flows = flows.copy()
        self.trades = trades.copy()
        self.prices = prices.copy()

        self.initial_cash = float(initial_cash)
        self.initial_shares = float(initial_shares)

        self._validate_and_normalize_inputs()

    def _validate_and_normalize_inputs(self) -> None:
        _require_cols(self.flows, ["ts", "amount", "direction"], "flows")
        _require_cols(self.trades, ["ts", "instrument", "qty", "side", "price"], "trades")
        _require_cols(self.prices, ["ts", "instrument", "price"], "prices")

        # Normalize timestamps
        self.flows["ts"] = _as_utc(self.flows["ts"])
        self.trades["ts"] = _as_utc(self.trades["ts"])
        self.prices["ts"] = _as_utc(self.prices["ts"])

        # Validate flows
        self.flows["direction"] = self.flows["direction"].astype(str).str.lower().str.strip()
        bad_dir = ~self.flows["direction"].isin(["deposit", "withdrawal"])
        if bad_dir.any():
            raise ValueError(f"flows.direction must be deposit/withdrawal. Bad: {self.flows.loc[bad_dir,'direction'].unique()}")
        if (self.flows["amount"] < 0).any():
            raise ValueError("flows.amount must be >= 0 with direction controlling sign.")

        # Flow signed amount (cash into fund)
        self.flows["signed_cash"] = np.where(
            self.flows["direction"].eq("deposit"),
            self.flows["amount"].astype(float),
            -self.flows["amount"].astype(float),
        )

        # Normalize trades
        self.trades["side"] = _normalize_side(self.trades["side"])
        if (self.trades["qty"] < 0).any():
            raise ValueError("trades.qty must be >= 0; side determines direction.")
        if (self.trades["price"] < 0).any():
            raise ValueError("trades.price must be >= 0.")

        if "fee" not in self.trades.columns:
            self.trades["fee"] = 0.0
        if "multiplier" not in self.trades.columns:
            self.trades["multiplier"] = 1.0

        if (self.trades["fee"] < 0).any():
            raise ValueError("trades.fee must be >= 0.")

        # Signed quantity for positions
        self.trades["signed_qty"] = np.where(
            self.trades["side"].eq("buy"),
            self.trades["qty"].astype(float),
            -self.trades["qty"].astype(float),
        )

        # Cash impact of trades: buy consumes cash, sell generates cash, fee consumes cash
        # cash_delta = -(signed_qty * price * multiplier) - fee
        self.trades["cash_delta"] = (
            -(self.trades["signed_qty"] * self.trades["price"] * self.trades["multiplier"])
            - self.trades["fee"]
        )

        # Prices
        if "multiplier" not in self.prices.columns:
            self.prices["multiplier"] = 1.0

        # Sort everything
        self.flows.sort_values("ts", inplace=True)
        self.trades.sort_values("ts", inplace=True)
        self.prices.sort_values(["instrument", "ts"], inplace=True)

    def _build_event_timeline(self) -> pd.DatetimeIndex:
        # All timestamps where anything happens + mark points (optional)
        ts = pd.Index(self.flows["ts"]).union(pd.Index(self.trades["ts"])).union(pd.Index(self.prices["ts"]))
        ts = ts.sort_values().unique()
        return pd.DatetimeIndex(ts)
    
    
    def _marks_asof(self, timeline: pd.DatetimeIndex) -> pd.DataFrame:
        p = self.prices[["ts", "instrument", "price"]].copy()
        wide = p.pivot_table(index="ts", columns="instrument", values="price", aggfunc="last").sort_index()

        # Align to timeline
        wide = wide.reindex(timeline)

        if self.cfg.mark_method == "ffill":
            wide = wide.ffill()
        elif self.cfg.mark_method == "last":
            # no fill
            pass
        else:
            raise ValueError(f"Unknown mark_method: {self.cfg.mark_method}")

        # NEW: allow initial backfill so early timestamps don't error before first mark exists
        if self.cfg.backfill_initial_marks:
            wide = wide.bfill(limit=None)  # bfill ONLY helps before first mark (and any remaining gaps)

        return wide


    def compute_ledger(self) -> pd.DataFrame:
        """
        Main output: ledger dataframe indexed by event timeline:
          cash, nav, shares_outstanding, share_price, position_mv, plus flow issuance/redemption.
        """
        timeline = self._build_event_timeline()
        marks = self._marks_asof(timeline)

        # Aggregate cash deltas per timestamp from flows + trades
        flow_cash_by_ts = self.flows.groupby("ts", as_index=True)["signed_cash"].sum()
        trade_cash_by_ts = self.trades.groupby("ts", as_index=True)["cash_delta"].sum()

        # Aggregate position deltas per timestamp per instrument
        pos_delta = (
            self.trades
            .groupby(["ts", "instrument"], as_index=False)["signed_qty"]
            .sum()
        )

        # Prepare running positions table
        instruments = sorted(set(self.trades["instrument"].unique()))
        pos = pd.DataFrame(0.0, index=timeline, columns=instruments)

        # Apply position deltas
        if not pos_delta.empty:
            # map ts to row index quickly
            # We'll add deltas per timestamp/instrument and cumsum over time
            for inst in instruments:
                s = (
                    pos_delta.loc[pos_delta["instrument"] == inst]
                    .set_index("ts")["signed_qty"]
                    .reindex(timeline)
                    .fillna(0.0)
                )
                pos[inst] = s.cumsum()

        # Validate marks availability for traded instruments
        if self.cfg.require_marks and instruments:
            missing_mask = marks[instruments].isna()
            if missing_mask.any().any():
                # Find first problematic timestamp/instrument
                bad = missing_mask.stack()
                bad = bad[bad].index.tolist()
                ts0, inst0 = bad[0]
                raise ValueError(
                    f"Missing mark price for traded instrument '{inst0}' at {ts0}. "
                    f"Provide prices earlier or set mark_method='ffill' and ensure initial marks exist."
                )

        # Valuation: position MV = sum(pos_i * mark_i * multiplier_i)
        # multipliers might differ; we take from prices table latest multiplier per instrument (or 1.0)
        mult = (
            self.prices.sort_values("ts")
            .groupby("instrument")["multiplier"]
            .last()
            .reindex(instruments)
            .fillna(1.0)
        )

        position_mv = (pos * marks[instruments].values) * mult.values  # broadcast
        position_mv_sum = position_mv.sum(axis=1)

        # Cash running
        cash = pd.Series(0.0, index=timeline)
        cash += self.initial_cash
        cash += flow_cash_by_ts.reindex(timeline).fillna(0.0).cumsum()
        cash += trade_cash_by_ts.reindex(timeline).fillna(0.0).cumsum()

        nav = cash + position_mv_sum

        if not self.cfg.allow_negative_nav and (nav < -1e-9).any():
            bad_ts = nav[nav < -1e-9].index[0]
            raise ValueError(f"NAV became negative at {bad_ts}. Check inputs/signs or set allow_negative_nav=True.")

        # Shares & quota:
        # Standard fund convention:
        #  - flow at timestamp t is processed at share_price computed *just before* applying that flow
        #  - trades do not issue shares; they impact NAV
        #
        # Because flows and trades can share same timestamp, we must define ordering.
        # We'll use a deterministic ordering:
        #   1) apply trades at ts (affects NAV)
        #   2) compute share_price_pre_flow at ts
        #   3) apply flows at ts -> issue/redeem shares
        #
        # We implement this by stepping through timeline and tracking:
        shares_out = []
        share_price = []
        flow_shares = []      # +issued, -redeemed at ts
        shares_outstanding = self.initial_shares

        # Precompute per-ts deltas
        flow_cash_ts = flow_cash_by_ts.reindex(timeline).fillna(0.0)
        trade_cash_ts = trade_cash_by_ts.reindex(timeline).fillna(0.0)

        # For position MV we already have pos and marks (after applying pos deltas at ts),
        # but our "trades first then flow" policy means:
        #   - pos at ts includes trades at ts (already true via cumsum)
        #   - cash at ts includes trades at ts and flows up to ts (cumsum includes flows at ts),
        #     which is NOT what we want for pricing flows.
        #
        # So we will compute "cash_after_trades_before_flow" as:
        #   initial_cash + cumsum(trades) + cumsum(flows excluding current ts)
        flow_cash_cum_excl = flow_cash_ts.cumsum() - flow_cash_ts
        cash_after_trades_before_flow = (
            self.initial_cash
            + trade_cash_ts.cumsum()
            + flow_cash_cum_excl
        )

        nav_pre_flow = cash_after_trades_before_flow + position_mv_sum

        for ts in timeline:
            nav0 = float(nav_pre_flow.loc[ts])
            f0 = float(flow_cash_ts.loc[ts])

            if shares_outstanding <= 0 and abs(nav0) < 1e-12:
                # Fund empty. Define initial share price.
                sp = float(self.cfg.initial_share_price)
            else:
                sp = _safe_div(nav0, shares_outstanding, default=float(self.cfg.initial_share_price))

            # Apply flow -> shares delta
            # Deposit (positive f0) issues shares: +f0/sp
            # Withdrawal (negative f0) redeems shares: f0/sp (negative)
            if abs(f0) < 1e-18:
                d_sh = 0.0
            else:
                if sp <= 0:
                    raise ValueError(f"Non-positive share price at {ts}: {sp}. Cannot process flows.")
                d_sh = f0 / sp

            shares_outstanding += d_sh

            # After flow, nav should become nav_pre_flow + f0 (since flow is cash into/out of fund)
            # Share price after flow should remain equal if no other changes at the same instant.
            shares_out.append(shares_outstanding)
            share_price.append(sp)
            flow_shares.append(d_sh)

            if shares_outstanding < -1e-12:
                raise ValueError(
                    f"Shares outstanding became negative at {ts}: {shares_outstanding}. "
                    "This indicates over-redemption or wrong flow sign."
                )

        ledger = pd.DataFrame(index=timeline)
        ledger.index.name = "ts"

        ledger["cash"] = _round_money(cash, self.cfg.money_decimals)
        ledger["position_mv"] = _round_money(position_mv_sum, self.cfg.money_decimals)
        ledger["nav"] = _round_money(nav, self.cfg.money_decimals)
        ledger["flow_cash"] = _round_money(flow_cash_ts, self.cfg.money_decimals)
        ledger["trade_cash"] = _round_money(trade_cash_ts, self.cfg.money_decimals)

        ledger["flow_shares_delta"] = _round_shares(pd.Series(flow_shares, index=timeline), self.cfg.shares_decimals)
        ledger["shares_outstanding"] = _round_shares(pd.Series(shares_out, index=timeline), self.cfg.shares_decimals)

        # "share_price" we recorded as the price used to process the flow at that timestamp
        ledger["share_price"] = pd.Series(share_price, index=timeline).round(10)

        # Optional: include per-instrument positions if useful
        for inst in instruments:
            ledger[f"pos:{inst}"] = pos[inst].round(10)

        return ledger

    def compute_latest_quota(self) -> Dict[str, float]:
        led = self.compute_ledger()
        last = led.iloc[-1]
        return {
            "ts": led.index[-1].to_pydatetime(),
            "share_price": float(last["share_price"]),
            "nav": float(last["nav"]),
            "shares_outstanding": float(last["shares_outstanding"]),
            "cash": float(last["cash"]),
            "position_mv": float(last["position_mv"]),
        }


# =========================
# User Allocation / Capital Calls
# =========================

@dataclass(frozen=True)
class UserAllocationConfig:
    money_decimals: int = 2
    min_transfer_abs: float = 0.0  # ignore tiny transfers below this threshold

    # How to convert user net worth into target fund capital:
    # You can plug your own policy function if needed.
    # Examples below: proportional, fixed target, cap, floor, etc.
    allocation_policy: Literal["proportional_to_net_worth", "fixed_target_total"] = "proportional_to_net_worth"


class UserCapitalAllocator:
    """
    Given user balance snapshots (assets/liabilities) and current user capital in fund,
    compute required transfers per user and total fund flow.

    Typical workflow:
      1) Compute fund quota (share price) from FundLedger.
      2) Convert user current fund shares -> current capital.
      3) Compute target capital per user by policy.
      4) Delta = target - current => deposit (>0) or withdraw (<0).
    """

    def __init__(
        self,
        cfg: UserAllocationConfig,
        users_balance: pd.DataFrame,
        users_current_fund_shares: pd.DataFrame,
        share_price: float,
        user_id_col: str = "user_id",
    ):
        self.cfg = cfg
        self.users_balance = users_balance.copy()
        self.users_shares = users_current_fund_shares.copy()
        self.share_price = float(share_price)
        self.user_id_col = user_id_col

        self._validate()

    def _validate(self) -> None:
        _require_cols(self.users_balance, [self.user_id_col, "total_assets", "total_liabilities"], "users_balance")
        _require_cols(self.users_shares, [self.user_id_col, "fund_shares"], "users_current_fund_shares")

        if self.share_price <= 0:
            raise ValueError(f"share_price must be > 0. Got {self.share_price}")

        self.users_balance[self.user_id_col] = self.users_balance[self.user_id_col].astype(str)
        self.users_shares[self.user_id_col] = self.users_shares[self.user_id_col].astype(str)

        if (self.users_shares["fund_shares"] < 0).any():
            raise ValueError("users_current_fund_shares.fund_shares must be >= 0 (unless you support short fund shares).")

    def compute_transfers(
        self,
        target_total_fund_capital: Optional[float] = None,
        target_per_user_capital: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Returns:
          transfers_df with columns:
            user_id, net_worth, current_fund_shares, current_fund_capital,
            target_fund_capital, delta_capital, direction, amount
          summary dict: totals
        """

        # Merge
        bal = self.users_balance.copy()
        bal["net_worth"] = bal["total_assets"].astype(float) - bal["total_liabilities"].astype(float)

        sh = self.users_shares.copy()
        sh["fund_shares"] = sh["fund_shares"].astype(float)

        df = bal.merge(sh, on=self.user_id_col, how="outer")
        df["total_assets"] = df["total_assets"].fillna(0.0).astype(float)
        df["total_liabilities"] = df["total_liabilities"].fillna(0.0).astype(float)
        df["net_worth"] = df["net_worth"].fillna(df["total_assets"] - df["total_liabilities"]).astype(float)
        df["fund_shares"] = df["fund_shares"].fillna(0.0).astype(float)

        # Current capital in fund
        df["current_fund_capital"] = df["fund_shares"] * self.share_price

        # Determine targets
        if target_per_user_capital is not None:
            _require_cols(target_per_user_capital, [self.user_id_col, "target_fund_capital"], "target_per_user_capital")
            t = target_per_user_capital.copy()
            t[self.user_id_col] = t[self.user_id_col].astype(str)
            df = df.merge(t[[self.user_id_col, "target_fund_capital"]], on=self.user_id_col, how="left")
            df["target_fund_capital"] = df["target_fund_capital"].fillna(0.0).astype(float)

        else:
            if target_total_fund_capital is None:
                # Default: keep total fund capital equal to current total (no net flow),
                # but re-allocate among users if policy says so.
                target_total_fund_capital = float(df["current_fund_capital"].sum())

            target_total_fund_capital = float(target_total_fund_capital)

            if self.cfg.allocation_policy == "proportional_to_net_worth":
                # Only users with positive net worth get allocation weight by default.
                # If you want to include negative net worth users, change this logic.
                weights = df["net_worth"].clip(lower=0.0)
                wsum = float(weights.sum())
                if wsum <= 0:
                    # No positive net worth; allocate zero everywhere
                    df["target_fund_capital"] = 0.0
                else:
                    df["target_fund_capital"] = target_total_fund_capital * (weights / wsum)

            elif self.cfg.allocation_policy == "fixed_target_total":
                # If you pass target_total_fund_capital, spread equally across users with positive net worth.
                eligible = df["net_worth"] > 0
                n = int(eligible.sum())
                df["target_fund_capital"] = 0.0
                if n > 0:
                    df.loc[eligible, "target_fund_capital"] = target_total_fund_capital / n
            else:
                raise ValueError(f"Unknown allocation_policy: {self.cfg.allocation_policy}")

        # Delta and direction
        df["delta_capital"] = df["target_fund_capital"] - df["current_fund_capital"]

        # Filter tiny deltas if requested
        if self.cfg.min_transfer_abs > 0:
            df.loc[df["delta_capital"].abs() < self.cfg.min_transfer_abs, "delta_capital"] = 0.0

        df["direction"] = np.where(df["delta_capital"] >= 0, "deposit", "withdrawal")
        df["amount"] = df["delta_capital"].abs()

        # Round money outputs
        df["current_fund_capital"] = _round_money(df["current_fund_capital"], self.cfg.money_decimals)
        df["target_fund_capital"] = _round_money(df["target_fund_capital"], self.cfg.money_decimals)
        df["delta_capital"] = _round_money(df["delta_capital"], self.cfg.money_decimals)
        df["amount"] = _round_money(df["amount"], self.cfg.money_decimals)

        # Also provide the implied shares delta (helpful for booking)
        df["shares_delta"] = _round_shares(df["delta_capital"] / self.share_price, 8)

        transfers = df[
            [self.user_id_col, "total_assets", "total_liabilities", "net_worth",
             "fund_shares", "current_fund_capital",
             "target_fund_capital", "delta_capital",
             "direction", "amount", "shares_delta"]
        ].sort_values(["direction", "amount"], ascending=[True, False])

        # Summary / reconciliation
        total_deposits = float(transfers.loc[transfers["direction"] == "deposit", "amount"].sum())
        total_withdrawals = float(transfers.loc[transfers["direction"] == "withdrawal", "amount"].sum())
        net_flow = total_deposits - total_withdrawals  # net cash into fund required

        summary = {
            "share_price": self.share_price,
            "total_current_fund_capital": float(transfers["current_fund_capital"].sum()),
            "total_target_fund_capital": float(transfers["target_fund_capital"].sum()),
            "total_deposits": total_deposits,
            "total_withdrawals": total_withdrawals,
            "net_flow_into_fund": net_flow,
        }

        return transfers, summary


# =========================
# Example Usage
# =========================
if __name__ == "__main__":
    # Example placeholders (replace with your real dataframes)
    flows = pd.DataFrame({
        "ts": ["2026-01-02 10:00:00", "2026-01-03 10:00:00"],
        "amount": [1_000_000, 100_000],
        "direction": ["deposit", "withdrawal"],
    })

    trades = pd.DataFrame({
        "ts": ["2026-01-02 12:00:00", "2026-01-03 09:30:00"],
        "instrument": ["AAPL", "AAPL"],
        "qty": [1000, 300],
        "side": ["buy", "sell"],
        "price": [180.0, 185.0],
        "fee": [50.0, 10.0],
        "multiplier": [1.0, 1.0],
    })

    prices = pd.DataFrame({
        "ts": ["2026-01-02 16:00:00", "2026-01-03 16:00:00"],
        "instrument": ["AAPL", "AAPL"],
        "price": [182.0, 184.0],
    })

    fund_cfg = FundAccountingConfig(
        base_currency="USD",
        mark_method="ffill",
        require_marks=True,
        initial_share_price=1.0,
        allow_negative_nav=False,
    )

    fund = FundLedger(
        config=fund_cfg,
        flows=flows,
        trades=trades,
        prices=prices,
        initial_cash=0.0,
        initial_shares=0.0,
    )

    ledger = fund.compute_ledger()
    latest = fund.compute_latest_quota()
    print("Latest quota snapshot:", latest)
    print(ledger.tail(10))

    # User allocator inputs
    users_balance = pd.DataFrame({
        "user_id": ["u1", "u2", "u3"],
        "total_assets": [5_000_000, 1_000_000, 200_000],
        "total_liabilities": [1_000_000, 200_000, 50_000],
    })

    users_current_fund_shares = pd.DataFrame({
        "user_id": ["u1", "u2", "u3"],
        "fund_shares": [200_000, 30_000, 5_000],
    })

    alloc_cfg = UserAllocationConfig(
        allocation_policy="proportional_to_net_worth",
        min_transfer_abs=1.0,
    )

    allocator = UserCapitalAllocator(
        cfg=alloc_cfg,
        users_balance=users_balance,
        users_current_fund_shares=users_current_fund_shares,
        share_price=latest["share_price"],
    )

    transfers, summary = allocator.compute_transfers(
        target_total_fund_capital=2_500_000  # total target AUM for these users
    )

    print("\nTransfers:")
    print(transfers)
    print("\nSummary:", summary)
