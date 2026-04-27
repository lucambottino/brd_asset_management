import os
import time
import hmac
import json
import base64
import hashlib
from decimal import Decimal
from typing import Any, Dict, Optional

import requests
import qrcode
from dotenv import load_dotenv


load_dotenv()

# ============================================================
# Configuration
# ============================================================
API_KEY = os.environ["BINANCE_API_KEY"]
API_SECRET = os.environ["BINANCE_API_SECRET"]

BASE_URL = "https://api.binance.com"
DEPOSIT_AMOUNT_BRL = Decimal("1000.00")

# Output files
QR_IMAGE_PATH = "binance_pix_qr.png"
QR_TEXT_PATH = "binance_pix_payload.txt"

# ============================================================
# Helpers
# ============================================================
class BinanceAPIError(Exception):
    pass


def sign_query_string(query_string: str, secret: str) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def signed_request(
    method: str,
    path: str,
    api_key: str,
    api_secret: str,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    if params is None:
        params = {}

    params = dict(params)
    params["timestamp"] = int(time.time() * 1000)

    # Binance documented a signing-order notice in its fiat changelog;
    # requests should be signed from the encoded payload. :contentReference[oaicite:2]{index=2}
    from urllib.parse import urlencode

    query_string = urlencode(params, doseq=True)
    signature = sign_query_string(query_string, api_secret)
    final_query = f"{query_string}&signature={signature}"

    url = f"{BASE_URL}{path}?{final_query}"

    headers = {
        "X-MBX-APIKEY": api_key,
        "Content-Type": "application/json",
    }

    response = requests.request(
        method=method.upper(),
        url=url,
        headers=headers,
        json=json_body,
        timeout=timeout,
    )

    try:
        data = response.json()
    except Exception as exc:
        raise BinanceAPIError(
            f"Non-JSON response from Binance: HTTP {response.status_code} - {response.text}"
        ) from exc

    if response.status_code >= 400:
        raise BinanceAPIError(
            f"HTTP {response.status_code} error from Binance: {json.dumps(data, ensure_ascii=False)}"
        )

    # Fiat endpoints commonly return code/message/success envelopes.
    if isinstance(data, dict):
        code = data.get("code")
        success = data.get("success")

        if success is False:
            raise BinanceAPIError(f"Binance request failed: {json.dumps(data, ensure_ascii=False)}")

        # Official examples show success code "000000". :contentReference[oaicite:3]{index=3}
        if code not in (None, "000000", 0, "0"):
            raise BinanceAPIError(f"Binance returned error code: {json.dumps(data, ensure_ascii=False)}")

    return data


def create_brl_pix_deposit_order(amount_brl: Decimal) -> str:
    body = {
        "currency": "BRL",
        "apiPaymentMethod": "Pix",
        "amount": float(amount_brl),
    }

    result = signed_request(
        method="POST",
        path="/sapi/v1/fiat/deposit",
        api_key=API_KEY,
        api_secret=API_SECRET,
        json_body=body,
    )

    data = result.get("data", {})
    order_id = data.get("orderId")
    if not order_id:
        raise BinanceAPIError(f"Deposit order created but no orderId returned: {result}")

    return str(order_id)


def get_order_detail(order_id: str) -> Dict[str, Any]:
    result = signed_request(
        method="GET",
        path="/sapi/v1/fiat/get-order-detail",
        api_key=API_KEY,
        api_secret=API_SECRET,
        params={"orderNo": order_id},
    )
    return result.get("data", {})


def find_pix_payload(detail: Dict[str, Any]) -> Optional[str]:
    """
    Binance's public docs do not specify the exact field name for the PIX
    copy-paste payload in order details. This function checks common possibilities.
    """
    candidate_keys = [
        "pixCode",
        "pix_code",
        "pixPayload",
        "pix_payload",
        "qrCode",
        "qr_code",
        "qrCodeText",
        "qr_code_text",
        "copyPaste",
        "copy_paste",
        "paymentCode",
        "payment_code",
        "code",
    ]

    # top-level search
    for key in candidate_keys:
        value = detail.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # ext search
    ext = detail.get("ext")
    if isinstance(ext, dict):
        for key in candidate_keys:
            value = ext.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


def save_qr_from_payload(payload: str, image_path: str, text_path: str) -> None:
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(payload)

    img = qrcode.make(payload)
    img.save(image_path)


def main() -> None:
    print("Creating Binance BRL PIX deposit order for 1000.00 BRL...")
    order_id = create_brl_pix_deposit_order(DEPOSIT_AMOUNT_BRL)
    print(f"Order created successfully. orderId={order_id}")

    # Give Binance a brief moment in case order details are not immediately enriched
    time.sleep(2)

    print("Fetching order details...")
    detail = get_order_detail(order_id)
    print("Order detail response:")
    print(json.dumps(detail, indent=2, ensure_ascii=False))

    pix_payload = find_pix_payload(detail)

    if pix_payload:
        print("PIX payload found. Generating QR code image...")
        save_qr_from_payload(
            payload=pix_payload,
            image_path=QR_IMAGE_PATH,
            text_path=QR_TEXT_PATH,
        )
        print(f"Saved QR image to: {QR_IMAGE_PATH}")
        print(f"Saved PIX copy-paste payload to: {QR_TEXT_PATH}")
    else:
        print(
            "\nNo PIX payload field was found in the documented order-detail response.\n"
            "This likely means Binance did not expose the QR/copy-paste code in this API response,\n"
            "or it uses an undocumented field name for your account/region.\n"
            "In that case, open the deposit order in the Binance app/web UI and use the QR shown there."
        )


if __name__ == "__main__":
    main()