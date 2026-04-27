import json
import re
from decimal import Decimal
from pathlib import Path

from PyPDF2 import PdfReader


PDF_PATH = Path("data/PosicaoConsolidada.pdf")
JSON_PATH = Path("data/carteiras-whitelisted.json")


def brl_to_decimal(value: str) -> Decimal:
    value = value.strip()
    value = value.replace("R$", "").replace(".", "").replace(",", ".").strip()
    return Decimal(value)


def extract_patrimonio_total(pdf_path: Path) -> Decimal:
    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    match = re.search(
        r"PATRIM[ÔO]NIO TOTAL\s*R\$\s*([\d\.\,]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError("Could not find 'PATRIMÔNIO TOTAL' in the PDF.")

    return brl_to_decimal(match.group(1))


def sum_saldo_carteira(json_path: Path) -> Decimal:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    wallets = data.get("wallets", [])
    total = Decimal("0")

    for wallet in wallets:
        saldo = wallet.get("saldoCarteira", 0)
        total += Decimal(str(saldo))

    return total


def format_brl(value: Decimal) -> str:
    sign = "-" if value < 0 else ""
    value = abs(value)
    s = f"{value:.2f}"
    integer, decimal = s.split(".")
    parts = []

    while integer:
        parts.append(integer[-3:])
        integer = integer[:-3]

    integer_formatted = ".".join(reversed(parts))
    return f"{sign}R$ {integer_formatted},{decimal}"


def main() -> None:
    patrimonio_total = extract_patrimonio_total(PDF_PATH)
    saldo_total = sum_saldo_carteira(JSON_PATH)

    # As requested: subtract the PDF value from the sum of saldoCarteira
    result = saldo_total - patrimonio_total

    print("PATRIMÔNIO TOTAL:", format_brl(patrimonio_total))
    print("Soma saldoCarteira:", format_brl(saldo_total))
    print("Resultado:", format_brl(result))


if __name__ == "__main__":
    main()