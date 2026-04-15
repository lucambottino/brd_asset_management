import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import List, Dict
import pandas as pd

# =========================
# CONFIGURATION
# =========================

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "luca@brd.capital"
SENDER_EMAIL = "lucambottino@gmail.com"

# SENDER_EMAIL = os.getenv("SMTP_EMAIL")
SENDER_PASSWORD = os.getenv("SMTP_PASSWORD")
print(f"Using sender email: {SENDER_EMAIL}")
# print(f"Using sender password: {'****' if SENDER_PASSWORD else 'Not Set'}")
print(SENDER_PASSWORD)
SENDER_PASSWORD = "J0@quim523"

if not SENDER_EMAIL or not SENDER_PASSWORD:
    raise ValueError("SMTP_EMAIL and SMTP_PASSWORD must be set as environment variables.")


# =========================
# DATASET
# =========================

investors: List[Dict[str, str]] = [
    {
        "Account": "6pgtAbfCwukveUfKYm22FEHP8w7NKD9Vd73NaCExLAKi",
        "Data de Compra": "23/01/2026",
        "Investido (BRL)": "R$ 59.773,07",
        "Rendimento Bruto (BRL)": "R$ 341,12",
        "Rendimento Líquido (BRL)": "R$ 150,69",
        "Lucro Bruto %": "0,57%",
        "Lucro Líquido %": "0,25%",
        "e-mail": "stephane@prosperojus.com.br",
        "name": "Stephane Alberto Lopes",
    },
    {
        "Account": "CbVUNS2kgnJszuWwP7ioxGLkUHGWppbcAitC5AynnDSx",
        "Data de Compra": "23/01/2026",
        "Investido (BRL)": "R$ 59.773,07",
        "Rendimento Bruto (BRL)": "R$ 341,12",
        "Rendimento Líquido (BRL)": "R$ 150,69",
        "Lucro Bruto %": "0,57%",
        "Lucro Líquido %": "0,25%",
        "e-mail": "ivan.yuri31@gmail.com",
        "name": "Ivan Augusto Yuri Rodarte",
    },
    {
        "Account": "czTNJnh5ihaYfaWzSjjratZrqMcw318KAPBQHdsvVBf",
        "Data de Compra": "23/01/2026",
        "Investido (BRL)": "R$ 29.886,53",
        "Rendimento Bruto (BRL)": "R$ 170,56",
        "Rendimento Líquido (BRL)": "R$ 75,34",
        "Lucro Bruto %": "0,57%",
        "Lucro Líquido %": "0,25%",
        "e-mail": "marcelo@mbsadv.com",
        "name": "Marcelo Godoy da Cunha Magalhaes",
    },
    {
        "Account": "AiYxij88KRXxWiAhNMqzxBd33wu5CSUd6KPAdoZETgfL",
        "Data de Compra": "30/01/2026",
        "Investido (BRL)": "R$ 256.148,18",
        "Rendimento Bruto (BRL)": "R$ 752,87",
        "Rendimento Líquido (BRL)": "R$ 198,38",
        "Lucro Bruto %": "0,29%",
        "Lucro Líquido %": "0,08%",
        "e-mail": "alexandreacp@gmail.com",
        "name": "Delfos",
    },
    # Add remaining valid entries here if needed
]


investors: List[Dict[str, str]] = [
    {
        "Account": "6pgtAbfCwukveUfKYm22FEHP8w7NKD9Vd73NaCExLAKi",
        "Data de Compra": "23/01/2026",
        "Investido (BRL)": "R$ 59.773,07",
        "Rendimento Bruto (BRL)": "R$ 341,12",
        "Rendimento Líquido (BRL)": "R$ 150,69",
        "Lucro Bruto %": "0,57%",
        "Lucro Líquido %": "0,25%",
        "e-mail": "lucambottino@gmail.com",
        "name": "Stephane Alberto Lopes",
    }
    # Add remaining valid entries here if needed
]

investors = pd.read_excel("lft_table.xlsx", sheet_name="Rateio LFT").to_dict(orient="records")

print(investors)

# =========================
# EMAIL BUILDER
# =========================

def build_email_body(investor: Dict[str, str]) -> str:
    df = pd.read_csv("export_transfer.csv")
    df = df[["To", "Signature", "Human Time"]].sort_values(by="Human Time", ascending=True)
    df = df.groupby("To").last().reset_index()
    d = df[["To", "Signature"]].set_index("To").to_dict()["Signature"]
    tx_id = d.get(investor["Account"], "N/A")
    return f"""Hello {investor['name']},

Thank you for being an early adopter of the Brazilian Real Digital.

As part of our transparency and client commitment here follows the details of the proceeds of your staking on BRD:

Total invested: {investor['Investido (BRL)']}
At: {investor['Data de Compra'].strftime('%d/%m/%Y')}
Gross profit: {investor['Rendimento Bruto (BRL)']} BRL
Net Profit: {investor['Rendimento Líquido (BRL)']} BRL
Gross Profit Percentage: {investor['Lucro Bruto %']:.2%}
Net Profit Percentage: {investor['Lucro Líquido %']:.2%}

The proceeds of your staking has been transferred to your account. You can check the transaction details on: https://solscan.io/tx/{tx_id}

If you have any questions, feel free to reach out.

Best regards,
Brazilian Real Digital Team
"""


def send_email(recipient_email: str, subject: str, body: str):
    msg = EmailMessage()
    msg["From"] = SENDER_EMAIL
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.set_content(body)

    context = ssl.create_default_context()

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls(context=context)
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)


# =========================
# MAIN EXECUTION
# =========================

def main():
    subject = "BRD Staking Proceeds Statement"

    print("Starting email dispatch...\n")

    for investor in investors:

        email = investor.get("e-mail")
        name = investor.get("name")

        if not email or email == "#N/A":
            print(f"Skipping invalid email for account {investor['Account']}")
            continue

        if not name or name == "#N/A":
            print(f"Skipping invalid name for account {investor['Account']}")
            continue

        try:
            print(f"Preparing email for {name} ({email})")

            body = build_email_body(investor)
            # send_email(email, subject, body)
            print(body)
            print("---- Email content above ----")
            print(f"Email successfully sent to {name} ({email})")

        except Exception as e:
            print(f"Failed to send email to {email}: {str(e)}")

    print("\nEmail dispatch completed.")


if __name__ == "__main__":
    main()
