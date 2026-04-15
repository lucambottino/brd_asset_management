import pandas as pd
import numpy as np
import re
from decimal import Decimal, ROUND_HALF_UP

# ============================================================
# Helpers
# ============================================================

def brl_to_decimal(x) -> Decimal:
    """
    Convert Brazilian currency/number strings like:
      "R$ 422.051,73" -> Decimal("422051.73")
      "422.051,73"    -> Decimal("422051.73")
      422051.73       -> Decimal("422051.73")
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return Decimal("0")
    if isinstance(x, Decimal):
        return x
    if isinstance(x, (int, float, np.integer, np.floating)):
        # avoid float artifacts by converting via str
        return Decimal(str(x))
    s = str(x).strip()
    s = s.replace("R$", "").replace("\u00a0", " ").strip()
    # keep digits, dot, comma, minus
    s = re.sub(r"[^\d,.\-]", "", s)
    # Brazilian format: thousands '.' and decimal ','
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    return Decimal(s)

def pct_to_decimal(x) -> Decimal:
    """
    Convert percentage strings like "38.24" or "38,24" or "38.24%"
    to Decimal fraction: 0.3824
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return Decimal("0")
    if isinstance(x, Decimal):
        return x
    if isinstance(x, (int, float, np.integer, np.floating)):
        return (Decimal(str(x)) / Decimal("100"))
    s = str(x).strip().replace("%", "").replace("\u00a0", " ")
    s = re.sub(r"[^\d,.\-]", "", s)
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    return Decimal(s) / Decimal("100")

def q2(x: Decimal) -> Decimal:
    """Quantize to cents."""
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def allocate_with_residual(total: Decimal, weights: pd.Series) -> pd.Series:
    """
    Allocate 'total' across rows with given weights (non-negative),
    rounding to cents while preserving exact total by distributing residual.
    """
    if total == 0:
        return pd.Series([Decimal("0.00")] * len(weights), index=weights.index)

    w = weights.astype("float64").values
    if np.allclose(w.sum(), 0.0):
        # fallback: equal split
        w = np.ones_like(w, dtype="float64")

    # raw allocations as Decimal
    raw = [Decimal(str(total)) * Decimal(str(wi)) / Decimal(str(w.sum())) for wi in w]
    rounded = [q2(a) for a in raw]
    allocated = pd.Series(rounded, index=weights.index, dtype=object)

    residual = total - sum(allocated.tolist(), start=Decimal("0.00"))
    # distribute residual in cents to the largest fractional remainders (or any stable order)
    cent = Decimal("0.01")
    steps = int((residual / cent).to_integral_value(rounding=ROUND_HALF_UP))
    if steps != 0:
        # compute fractional remainders to decide who gets residual cents
        # remainder = raw - rounded
        remainders = pd.Series([r - q2(r) for r in raw], index=weights.index, dtype=object)

        # If residual positive, give +0.01 to those with largest remainders
        # If residual negative, subtract 0.01 from those with smallest remainders
        if steps > 0:
            order = remainders.sort_values(ascending=False).index.tolist()
            for i in range(steps):
                allocated.loc[order[i % len(order)]] = allocated.loc[order[i % len(order)]] + cent
        else:
            order = remainders.sort_values(ascending=True).index.tolist()
            for i in range(abs(steps)):
                allocated.loc[order[i % len(order)]] = allocated.loc[order[i % len(order)]] - cent

    # final safety: exact match
    final_sum = sum(allocated.tolist(), start=Decimal("0.00"))
    if final_sum != total:
        # last-resort fix by adjusting first row
        allocated.iloc[0] = allocated.iloc[0] + (total - final_sum)

    return allocated

# ============================================================
# Inputs (fill from your tables; these are exactly your values)
# ============================================================

# LFT "rendimentos" table (one row per purchase lot)
lft_lots = pd.DataFrame(
    [
        {
            "Data de compra": "23/01/2026",
            "Valor de compra": "R$ 419.656,79",
            "Valor líquido": "R$ 423712,36",
            "Valor bruto": "R$ 424.889,78",
            "IOF": "R$ 0",
            "IR": "R$ 1177,42",
        },
        {
            "Data de compra": "30/01/2026",
            "Valor de compra": "R$ 256.148,18",
            "Valor líquido": "R$ 257955,14",
            "Valor bruto": "R$ 258.628,56",
            "IOF": "R$ 148,82",
            "IR": "R$ 524,60",
        },
    ]
)

# Accounts purchases table
accounts = pd.DataFrame(
    [
        {"Account": "AiYxij88KRXxWiAhNMqzxBd33wu5CSUd6KPAdoZETgfL", "Token Account": "5MZZn4D96MkndVoBjSpuqCdnMDuv87VFV1A6kZrHLuA4", "Quantity": 260870, "Percentage": "38.24", "Data de Compra": "30/01/2026"},
        {"Account": "6pgtAbfCwukveUfKYm22FEHP8w7NKD9Vd73NaCExLAKi", "Token Account": "58kQw1fVjtSdKHYHBsvRenDPfDFSV8MvwesVWsE8MsZM", "Quantity": 60000, "Percentage": "8.80", "Data de Compra": "23/01/2026"},
        {"Account": "CbVUNS2kgnJszuWwP7ioxGLkUHGWppbcAitC5AynnDSx", "Token Account": "4f2BJC3B3LRiapnz8xiT4YkXLvQzHbnCFxbMaSito4fh", "Quantity": 60000, "Percentage": "8.80", "Data de Compra": "23/01/2026"},
        {"Account": "czTNJnh5ihaYfaWzSjjratZrqMcw318KAPBQHdsvVBf", "Token Account": "HnRpBvc3gMfSrJzCer36jnobnHWTph56hFKyvKG91iYP", "Quantity": 30000, "Percentage": "4.40", "Data de Compra": "23/01/2026"},
        {"Account": "Ai6gzjwzCL9E1rVMR9gP5gD9Y9Eds1uKj1NoC3ggJw4T", "Token Account": "7Sxy6aovd4VEThL7P8X6ZcS9DxbH9Sy6HkMMSeisVf8b", "Quantity": 30000, "Percentage": "4.40", "Data de Compra": "23/01/2026"},
        {"Account": "9bxQUSioYmCfqeraQ4x1UND5k2PLMmMxLDMCGxVW9Mzz", "Token Account": "4bNa2pBXGVra7kqixKsGYkwx3MdSCuPiZpGrC7Tsc3Lf", "Quantity": 30000, "Percentage": "4.40", "Data de Compra": "23/01/2026"},
        {"Account": "8wju6VrZpjoZC5TZTGWVJSjNMnz8qeKweZpArcu8BoX9", "Token Account": "2CvmUYrv4CcZR2zvJhttNMLy4zNKKicwWoBRzBEveXx6", "Quantity": 30000, "Percentage": "4.40", "Data de Compra": "23/01/2026"},
        {"Account": "3qsNsvWUJsZTk9FS3mmFv2YiUD6gPUJwpeqLQv4ahof8", "Token Account": "2WvvG3sHfHyRBajuzCxbREedRHNGX1YxiEVohGK5GEEf", "Quantity": 30000, "Percentage": "4.40", "Data de Compra": "23/01/2026"},
        {"Account": "3hJgnmP5gZfkV876PnvJpzCUrq9vGm9fckQNhUB6ftWV", "Token Account": "FPzm4TruXnfGkf2t3YgC5vCPjkLxwYGvPCJXTc4UP1WC", "Quantity": 30000, "Percentage": "4.40", "Data de Compra": "23/01/2026"},
        {"Account": "5h1sZWC93q9mWihsa2tvrsuJoqGY58655CSHkug2opRV", "Token Account": "FMb9i4T61D6YNh8cBWvMYKVuPMV6F1ETq2ZV9GLJU51e", "Quantity": 30000, "Percentage": "4.40", "Data de Compra": "23/01/2026"},
        {"Account": "GMuR2i1TfXKvdXJn3iBZCdXomnqVQN3NpEApnmixEndb", "Token Account": "2Sm5kBgNey4zMnxaGBxNN8HhsoikbmgwEJu4cy9jfrS1", "Quantity": 30000, "Percentage": "4.40", "Data de Compra": "23/01/2026"},
        {"Account": "3DHW2ZCYcKdzqBxyeVcG6TxG6foHdokvJS6Xr55VKPZq", "Token Account": "CSXsNbG9jGk4hYwVz1W2DTXX4TX7dRZ9bfypHgxtuDfV", "Quantity": 30000, "Percentage": "4.40", "Data de Compra": "23/01/2026"},
        {"Account": "5C7R3fhGJ37wBWEbSrw8KU8sfj8zeoctDAFDkLxcJmgv", "Token Account": "5cKSj1EmK8tViMjKGnaKWSD2jBFK5FN6aQ8hEPf3vTor", "Quantity": 30000, "Percentage": "4.40", "Data de Compra": "23/01/2026"},
        {"Account": "GuU4YH1v6DdkbZwh5Qi7prDxEupGFTtUaTU7EpzRHbQU", "Token Account": "HV2LwjQJj7PCFJGYRWVPZN14YPfEEaTKy4YanFaDJrMx", "Quantity": 1250, "Percentage": "0.18", "Data de Compra": "23/01/2026"},
    ]
)

# ============================================================
# Core computation
# ============================================================

def compute_yield_by_account(
    lft_lots_df: pd.DataFrame,
    accounts_df: pd.DataFrame,
    *,
    weight_mode: str = "quantity",  # "quantity" or "percentage"
) -> pd.DataFrame:
    """
    For each purchase date:
      - compute net yield (after IOF/IR) from the LFT lot:
            rendimento_liquido = Valor líquido - Valor de compra
      - allocate this yield across accounts that have the same Data de Compra,
        proportionally to Quantity or Percentage.

    Returns a DataFrame with (Account, Data de Compra, Rendimento (BRL)).
    """
    lft = lft_lots_df.copy()
    acc = accounts_df.copy()

    # Parse dates
    lft["Data de compra"] = pd.to_datetime(lft["Data de compra"], dayfirst=True)
    acc["Data de Compra"] = pd.to_datetime(acc["Data de Compra"], dayfirst=True)

    # Parse money fields (Decimal)
    for col in ["Valor de compra", "Valor líquido", "Valor bruto", "IOF", "IR"]:
        lft[col] = lft[col].apply(brl_to_decimal)

    # Net yield after taxes (liquid)
    lft["Rendimento líquido"] = lft["Valor líquido"] - lft["Valor de compra"]

    # Validate and aggregate lots by purchase date (in case there are multiple lots same day)
    lots_by_date = (
        lft.groupby("Data de compra", as_index=False)
           .agg(
               Valor_de_compra=("Valor de compra", lambda s: sum(s.tolist(), start=Decimal("0"))),
               Valor_liquido=("Valor líquido", lambda s: sum(s.tolist(), start=Decimal("0"))),
               Rendimento_liquido=("Rendimento líquido", lambda s: sum(s.tolist(), start=Decimal("0"))),
               IOF=("IOF", lambda s: sum(s.tolist(), start=Decimal("0"))),
               IR=("IR", lambda s: sum(s.tolist(), start=Decimal("0"))),
           )
    )

    # Prepare weights
    if weight_mode not in ("quantity", "percentage"):
        raise ValueError("weight_mode must be 'quantity' or 'percentage'")

    if weight_mode == "quantity":
        acc["_w"] = acc["Quantity"].astype("int64")
    else:
        acc["_w"] = acc["Percentage"].apply(pct_to_decimal).apply(lambda d: float(d))  # numeric weights

    # Allocate per date
    out_rows = []
    for _, lot in lots_by_date.iterrows():
        dt = lot["Data de compra"]
        total_yield = lot["Rendimento_liquido"]  # Decimal
        subset = acc.loc[acc["Data de Compra"] == dt].copy()

        if subset.empty:
            # No accounts for that date; skip but keep traceability
            continue

        alloc = allocate_with_residual(total_yield, subset["_w"])
        subset["Rendimento (BRL)"] = alloc.values  # Decimals

        # Output
        for _, r in subset.iterrows():
            out_rows.append(
                {
                    "Account": r["Account"],
                    "Token Account": r.get("Token Account", None),
                    "Data de Compra": r["Data de Compra"].date().isoformat(),
                    "Weight Mode": weight_mode,
                    "Weight": r["_w"],
                    "Rendimento (BRL)": str(r["Rendimento (BRL)"]),  # keep exact decimals
                }
            )

    out = pd.DataFrame(out_rows)

    # Add totals per account (summing across purchase dates if needed)
    if not out.empty:
        # sum Decimals safely
        def sum_decimals(series):
            return str(sum((Decimal(x) for x in series), start=Decimal("0.00")))

        totals = (
            out.groupby("Account", as_index=False)["Rendimento (BRL)"]
               .apply(sum_decimals)
               .rename(columns={"Rendimento (BRL)": "Rendimento Total (BRL)"})
        )
        out = out.merge(totals, on="Account", how="left")

    return out


# ============================================================
# Run
# ============================================================

result = compute_yield_by_account(lft_lots, accounts, weight_mode="quantity")

# Pretty print (as BRL string)
def fmt_brl(dec_str: str) -> str:
    d = Decimal(dec_str)
    d = q2(d)
    # format with thousands '.' and decimals ','
    s = f"{d:,.2f}"  # US style "1,234.56"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")  # -> "1.234,56"
    return f"R$ {s}"

if not result.empty:
    display_cols = ["Account", "Data de Compra", "Rendimento (BRL)", "Rendimento Total (BRL)"]
    view = result[display_cols].copy()
    view["Rendimento (BRL)"] = view["Rendimento (BRL)"].apply(fmt_brl)
    view["Rendimento Total (BRL)"] = view["Rendimento Total (BRL)"].apply(fmt_brl)
    print(view.to_string(index=False))

    # Sanity check totals by date
    # (Compare allocated vs computed lot yields)
    r = result.copy()
    r["_dec"] = r["Rendimento (BRL)"].apply(Decimal)
    r["Data de Compra"] = pd.to_datetime(r["Data de Compra"])
    allocated_by_date = r.groupby("Data de Compra")["_dec"].apply(lambda s: sum(s.tolist(), start=Decimal("0.00")))

    l = lft_lots.copy()
    l["Data de compra"] = pd.to_datetime(l["Data de compra"], dayfirst=True)
    l["Valor de compra"] = l["Valor de compra"].apply(brl_to_decimal)
    l["Valor líquido"] = l["Valor líquido"].apply(brl_to_decimal)
    l["Rendimento líquido"] = l["Valor líquido"] - l["Valor de compra"]
    lot_by_date = l.groupby("Data de compra")["Rendimento líquido"].apply(lambda s: sum(s.tolist(), start=Decimal("0.00")))

    print("\nSanity check (per date):")
    for dt in sorted(set(lot_by_date.index) | set(allocated_by_date.index)):
        lot_y = lot_by_date.get(dt, Decimal("0.00"))
        al_y = allocated_by_date.get(dt, Decimal("0.00"))
        print(f"  {dt.date().isoformat()}  lot={fmt_brl(str(lot_y))}  allocated={fmt_brl(str(al_y))}  diff={fmt_brl(str(lot_y - al_y))}")
else:
    print("No allocations produced. Check date formats and that accounts dates match LFT purchase dates.")
