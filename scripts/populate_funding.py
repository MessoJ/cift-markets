import os
import sys
from datetime import datetime, timedelta

import psycopg2

# Add parent directory to path to import backend modules if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_db_connection():
    """Connect to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname="cift_markets",
            user="cift_user",
            password="cift_password",
            host="localhost",
            port="5432"
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

def populate_payment_methods(conn, user_id):
    """Add mock payment methods if none exist."""
    cur = conn.cursor()

    # Check existing methods
    cur.execute("SELECT count(*) FROM payment_methods WHERE user_id = %s", (user_id,))
    count = cur.fetchone()[0]

    if count > 0:
        print(f"User {user_id} already has {count} payment methods. Skipping.")
        return

    print(f"Adding payment methods for user {user_id}...")

    methods = [
        {
            "type": "bank_account",
            "name": "Chase Checking",
            "bank_name": "Chase Bank",
            "account_number": "********1234",
            "routing_number": "********5678",
            "last_four": "1234",
            "is_default": True,
            "is_verified": True
        },
        {
            "type": "debit_card",
            "name": "Visa Debit",
            "card_brand": "Visa",
            "last_four": "4242",
            "expiry_month": 12,
            "expiry_year": 2028,
            "is_default": False,
            "is_verified": True
        },
        {
            "type": "crypto_wallet",
            "name": "Metamask",
            "crypto_network": "ethereum",
            "wallet_address": "0x71C...9A23",
            "is_default": False,
            "is_verified": True
        }
    ]

    for m in methods:
        cur.execute("""
            INSERT INTO payment_methods (
                user_id, type, name, bank_name, account_number, routing_number,
                last_four, card_brand, expiry_month, expiry_year,
                crypto_network, wallet_address, is_default, is_verified, created_at, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, NOW(), NOW()
            )
        """, (
            user_id, m["type"], m["name"], m.get("bank_name"), m.get("account_number"), m.get("routing_number"),
            m.get("last_four"), m.get("card_brand"), m.get("expiry_month"), m.get("expiry_year"),
            m.get("crypto_network"), m.get("wallet_address"), m["is_default"], m["is_verified"]
        ))

    conn.commit()
    print("Payment methods added.")

def populate_funding_history(conn, user_id):
    """Add mock funding history if none exists."""
    cur = conn.cursor()

    # Check existing transactions
    cur.execute("SELECT count(*) FROM funding_transactions WHERE user_id = %s", (user_id,))
    count = cur.fetchone()[0]

    if count > 0:
        print(f"User {user_id} already has {count} funding transactions. Skipping.")
        return

    print(f"Adding funding history for user {user_id}...")

    # Get payment method IDs
    cur.execute("SELECT id, type FROM payment_methods WHERE user_id = %s", (user_id,))
    methods = cur.fetchall()
    if not methods:
        print("No payment methods found. Cannot add history.")
        return

    method_map = {m[1]: m[0] for m in methods} # type -> id

    transactions = [
        {
            "type": "deposit",
            "amount": 5000.00,
            "currency": "USD",
            "status": "completed",
            "payment_method_id": method_map.get("bank_account"),
            "created_at": datetime.now() - timedelta(days=30),
            "completed_at": datetime.now() - timedelta(days=28),
            "reference": "DEP-8839201"
        },
        {
            "type": "deposit",
            "amount": 1500.00,
            "currency": "USD",
            "status": "completed",
            "payment_method_id": method_map.get("debit_card"),
            "created_at": datetime.now() - timedelta(days=15),
            "completed_at": datetime.now() - timedelta(days=15),
            "reference": "DEP-9928374"
        },
        {
            "type": "withdrawal",
            "amount": 200.00,
            "currency": "USD",
            "status": "completed",
            "payment_method_id": method_map.get("bank_account"),
            "created_at": datetime.now() - timedelta(days=5),
            "completed_at": datetime.now() - timedelta(days=3),
            "reference": "WTH-1122334"
        },
        {
            "type": "deposit",
            "amount": 10000.00,
            "currency": "USD",
            "status": "processing",
            "payment_method_id": method_map.get("bank_account"),
            "created_at": datetime.now() - timedelta(hours=2),
            "completed_at": None,
            "reference": "DEP-7744112"
        }
    ]

    for t in transactions:
        if not t["payment_method_id"]:
            continue

        cur.execute("""
            INSERT INTO funding_transactions (
                user_id, type, amount, currency, status, payment_method_id,
                created_at, completed_at, reference_number
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s
            )
        """, (
            user_id, t["type"], t["amount"], t["currency"], t["status"], t["payment_method_id"],
            t["created_at"], t["completed_at"], t["reference"]
        ))

    conn.commit()
    print("Funding history added.")

def main():
    conn = get_db_connection()

    # Get the first user (usually the demo user)
    cur = conn.cursor()
    cur.execute("SELECT id FROM users ORDER BY id LIMIT 1")
    user = cur.fetchone()

    if not user:
        print("No users found in database.")
        return

    user_id = user[0]
    print(f"Populating data for user ID: {user_id}")

    populate_payment_methods(conn, user_id)
    populate_funding_history(conn, user_id)

    conn.close()
    print("Done.")

if __name__ == "__main__":
    main()
