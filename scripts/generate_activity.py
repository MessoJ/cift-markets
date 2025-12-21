import asyncio
import os
import random
from datetime import datetime, timedelta
from decimal import Decimal

import asyncpg

# DB Config
DB_USER = os.getenv("POSTGRES_USER", "cift_user")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "changeme123")
DB_NAME = os.getenv("POSTGRES_DB", "cift_markets")
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

async def generate_activity():
    print(f"Connecting to {DB_HOST}:{DB_PORT}...")
    try:
        conn = await asyncpg.connect(
            user=DB_USER, password=DB_PASS, database=DB_NAME, host=DB_HOST, port=DB_PORT
        )
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    print("Connected. Generating supplementary activity...")

    # Get user and account
    row = await conn.fetchrow("SELECT user_id, account_id FROM transactions LIMIT 1")
    if not row:
        print("No existing transactions found. Cannot link activity.")
        return

    user_id = row['user_id']
    account_id = row['account_id']

    # 1. Monthly Platform Fees ($15/month for last 6 months)
    print("Generating Fees...")
    base_date = datetime.now()
    for i in range(6):
        date = base_date - timedelta(days=30 * (i + 1))
        # Check if fee exists for this month
        exists = await conn.fetchval("""
            SELECT 1 FROM transactions
            WHERE transaction_type = 'fee'
            AND date_trunc('month', transaction_date) = date_trunc('month', $1::timestamp)
        """, date)

        if not exists:
            await conn.execute("""
                INSERT INTO transactions (
                    user_id, account_id, transaction_type, amount, balance_after,
                    description, transaction_date, created_at
                ) VALUES ($1, $2, 'fee', -15.00, 0, 'Monthly Platform Data Fee', $3, NOW())
            """, user_id, account_id, date)
            print(f"  - Added Fee for {date.strftime('%B %Y')}")

    # 2. Dividends (Quarterly for AAPL/MSFT)
    print("Generating Dividends...")
    stocks = ['AAPL', 'MSFT']
    for stock in stocks:
        for i in range(2): # Last 2 quarters
            date = base_date - timedelta(days=90 * (i + 1))
            amount = Decimal(random.uniform(15.0, 45.0)).quantize(Decimal("0.01"))

            await conn.execute("""
                INSERT INTO transactions (
                    user_id, account_id, transaction_type, amount, balance_after,
                    symbol, description, transaction_date, created_at
                ) VALUES ($1, $2, 'dividend', $3, 0, $4, $5, $6, NOW())
            """, user_id, account_id, amount, stock, f"{stock} Quarterly Dividend", date)
            print(f"  - Added ${amount} Dividend for {stock}")

    # 3. One Withdrawal
    print("Generating Withdrawal...")
    w_date = base_date - timedelta(days=45)
    await conn.execute("""
        INSERT INTO transactions (
            user_id, account_id, transaction_type, amount, balance_after,
            description, transaction_date, created_at
        ) VALUES ($1, $2, 'withdrawal', -2500.00, 0, 'Wire Transfer to Bank ****8821', $3, NOW())
    """, user_id, account_id, w_date)
    print("  - Added $2,500 Withdrawal")

    # 4. Recalculate ALL Balances (Crucial)
    print("Recalculating all balances...")
    txns = await conn.fetch("SELECT id, amount FROM transactions ORDER BY transaction_date ASC")

    balance = Decimal(0)
    for txn in txns:
        amount = txn['amount']
        balance += amount
        await conn.execute("UPDATE transactions SET balance_after = $1 WHERE id = $2", balance, txn['id'])

    print(f"Balances updated for {len(txns)} transactions.")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(generate_activity())
