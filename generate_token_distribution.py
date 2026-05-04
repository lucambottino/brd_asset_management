import argparse
import csv
import json
import re
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from pathlib import Path


DEFAULT_CSV_PATH = Path("data/token_holders.csv")
DEFAULT_WALLETS_JSON_PATH = Path("data/carteiras-whitelisted.json")
DEFAULT_OUTPUT_PATH = Path("data/token_distribution.json")


def parse_decimal(value: str) -> Decimal:
    text = str(value).strip().replace("_", "")
    text = re.sub(r"[^\d,.\-]", "", text)
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        text = text.replace(",", ".")
    return Decimal(text)


def read_token_holders(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"Account", "Token Account", "Quantity"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required CSV column(s): {missing}")

        holders = []
        for row in reader:
            quantity = parse_decimal(row["Quantity"])
            if quantity <= 0:
                continue
            holders.append(
                {
                    "account": row["Account"].strip(),
                    "token_account": row["Token Account"].strip(),
                    "quantity": quantity,
                }
            )

    if not holders:
        raise ValueError(f"No positive token quantities found in {csv_path}")

    return holders


def load_wallet_names(wallets_json_path: Path) -> dict[str, dict]:
    if not wallets_json_path.exists():
        return {}

    with wallets_json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    names_by_account = {}
    for wallet in data.get("wallets", []):
        account = str(wallet.get("chaveCarteira", "")).strip()
        if not account:
            continue
        names_by_account[account] = {
            "wallet_name": wallet.get("nomeCarteira") or account,
            "client_name": wallet.get("nomeCliente"),
        }

    return names_by_account


def allocate_proportionally(
    total_to_allocate: Decimal,
    holders: list[dict],
    decimals: int,
) -> list[Decimal]:
    unit = Decimal("1").scaleb(-decimals)
    rounded_total = total_to_allocate.quantize(unit, rounding=ROUND_HALF_UP)
    total_quantity = sum((holder["quantity"] for holder in holders), Decimal("0"))

    raw_allocations = [
        rounded_total * holder["quantity"] / total_quantity for holder in holders
    ]
    allocations = [
        raw.quantize(unit, rounding=ROUND_DOWN) for raw in raw_allocations
    ]

    allocated = sum(allocations, Decimal("0"))
    remainder_units = int(((rounded_total - allocated) / unit).to_integral_value())
    remainders = [
        (raw - allocation, index)
        for index, (raw, allocation) in enumerate(zip(raw_allocations, allocations))
    ]
    remainders.sort(reverse=True)

    for _, index in remainders[:remainder_units]:
        allocations[index] += unit

    return allocations


def decimal_to_string(value: Decimal, decimals: int | None = None) -> str:
    if decimals is not None:
        value = value.quantize(Decimal("1").scaleb(-decimals))
    return format(value, "f")


def build_distribution(
    holders: list[dict],
    wallet_names: dict[str, dict],
    tokens_minted_so_far: Decimal,
    total_collateral: Decimal,
    decimals: int,
) -> dict:
    tokens_to_distribute = total_collateral - tokens_minted_so_far
    if tokens_to_distribute < 0:
        raise ValueError(
            "total_collateral is lower than tokens_minted_so_far; "
            "there are no new tokens to distribute."
        )

    allocations = allocate_proportionally(tokens_to_distribute, holders, decimals)
    total_quantity = sum((holder["quantity"] for holder in holders), Decimal("0"))
    allocated_total = sum(allocations, Decimal("0"))

    wallets = []
    for holder, allocation in zip(holders, allocations):
        wallet_data = wallet_names.get(holder["account"], {})
        wallet_name = wallet_data.get("wallet_name") or holder["account"]
        percentage = holder["quantity"] / total_quantity * Decimal("100")

        wallets.append(
            {
                "walletName": wallet_name,
                "clientName": wallet_data.get("client_name"),
                "account": holder["account"],
                "tokenAccount": holder["token_account"],
                "tokensHeld": decimal_to_string(holder["quantity"]),
                "holderPercentage": decimal_to_string(percentage, 8),
                "tokensToReceive": decimal_to_string(allocation, decimals),
            }
        )

    return {
        "totals": {
            "tokensMintedSoFar": decimal_to_string(tokens_minted_so_far),
            "totalCollateral": decimal_to_string(total_collateral),
            "tokensToDistribute": decimal_to_string(tokens_to_distribute, decimals),
            "totalTokensHeld": decimal_to_string(total_quantity),
            "allocatedTokens": decimal_to_string(allocated_total, decimals),
            "decimals": decimals,
        },
        "wallets": wallets,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a JSON token distribution from data/token_holders.csv. "
            "Each wallet receives (wallet_quantity / total_quantity) times "
            "(total_collateral - tokens_minted_so_far)."
        )
    )
    parser.add_argument(
        "--tokens-minted-so-far",
        required=True,
        type=parse_decimal,
        help="Amount of tokens already minted.",
    )
    parser.add_argument(
        "--total-collateral",
        required=True,
        type=parse_decimal,
        help="Total collateral amount backing the token supply.",
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV_PATH,
        type=Path,
        help=f"Token holders CSV path. Default: {DEFAULT_CSV_PATH}",
    )
    parser.add_argument(
        "--wallets-json",
        default=DEFAULT_WALLETS_JSON_PATH,
        type=Path,
        help=(
            "Optional wallet metadata JSON used to map Account to wallet names. "
            f"Default: {DEFAULT_WALLETS_JSON_PATH}"
        ),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        type=Path,
        help=f"Output JSON path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--decimals",
        default=6,
        type=int,
        help="Decimal places to use for token allocations. Default: 6",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.decimals < 0:
        raise ValueError("--decimals must be greater than or equal to zero")

    holders = read_token_holders(args.csv)
    wallet_names = load_wallet_names(args.wallets_json)
    distribution = build_distribution(
        holders=holders,
        wallet_names=wallet_names,
        tokens_minted_so_far=args.tokens_minted_so_far,
        total_collateral=args.total_collateral,
        decimals=args.decimals,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(distribution, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
