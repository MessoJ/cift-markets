import asyncio
import asyncpg
import os
from decimal import Decimal
from datetime import timedelta

# DB Config - Use environment variables or defaults matching docker-compose
DB_USER = os.getenv("POSTGRES_USER", "cift_user")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "changeme123")
DB_NAME = os.getenv("POSTGRES_DB", "cift_markets")
DB_HOST = os.getenv("POSTGRES_HOST", "postgres") # Default to container name
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

async def reconcile():
    print(f"Connecting to {DB_HOST}:{DB_PORT}...")
    try:
        conn = await asyncpg.connect(
            user=DB_USER, 
            password=DB_PASS, 
            database=DB_NAME, 
            host=DB_HOST, 
            port=DB_PORT
        )
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    print("Connected to DB.")
    
    # 1. Get Filled Orders that don't have a transaction
    query = """
        SELECT o.* 
        FROM orders o
        LEFT JOIN transactions t ON o.id = t.order_id
        WHERE o.status = 'filled' AND t.id IS NULL
    """
    
    orders = await conn.fetch(query)
    print(f"Found {len(orders)} filled orders without transactions.")
    
    count = 0
    for order in orders:
        # Calculate amount
        qty = Decimal(str(order['filled_quantity']))
        price = Decimal(str(order['avg_fill_price'])) if order['avg_fill_price'] else Decimal(0)
        
        if price == 0 and order['limit_price']:
             price = Decimal(str(order['limit_price']))
            
        amount = qty * price
        
        if order['side'] == 'buy':
            amount = -amount
            desc = f"Bought {qty} {order['symbol']} @ ${price:.2f}"
        else:
            # Sell is positive inflow
            desc = f"Sold {qty} {order['symbol']} @ ${price:.2f}"
            
        # Insert Transaction
        try:
            await conn.execute("""
                INSERT INTO transactions (
                    user_id, account_id, transaction_type, amount, balance_after, 
                    order_id, symbol, description, transaction_date, created_at
                ) VALUES ($1, $2, 'trade', $3, 0, $4, $5, $6, $7, NOW())
            """, order['user_id'], order['account_id'], amount, order['id'], 
            order['symbol'], desc, order['created_at'])
            count += 1
        except Exception as e:
            print(f"Failed to insert order {order['id']}: {e}")
           
    print(f"Inserted {count} transactions.")
    
    # 2. Recalculate Balances
    print("Recalculating balances...")
    txns = await conn.fetch("SELECT id, amount FROM transactions ORDER BY transaction_date ASC")
    
    balance = Decimal(0)
    for txn in txns:
        amount = txn['amount']
        balance += amount
        await conn.execute("UPDATE transactions SET balance_after = $1 WHERE id = $2", balance, txn['id'])
        
    print(f"Recalculated balances for {len(txns)} transactions.")
    
    # 3. Ensure positive balance (Seed Deposit)
    # Find minimum balance
    min_balance = await conn.fetchval("SELECT MIN(balance_after) FROM transactions")
    if min_balance is not None and min_balance < 0:
        print(f"Minimum balance is {min_balance}. Injecting seed deposit...")
        needed = abs(min_balance) + 100000 # Add $100k buffer
        
        # Get date of first transaction
        first_txn = await conn.fetchrow("SELECT transaction_date, user_id, account_id FROM transactions ORDER BY transaction_date ASC LIMIT 1")
        
        if first_txn:
            # Calculate date in python
            deposit_date = first_txn['transaction_date'] - timedelta(minutes=1)
            
            # Insert deposit 1 minute before first transaction
            await conn.execute("""
                INSERT INTO transactions (
                    user_id, account_id, transaction_type, amount, balance_after, 
                    description, transaction_date, created_at
                ) VALUES ($1, $2, 'deposit', $3, 0, 'Initial Funding', $4, NOW())
            """, first_txn['user_id'], first_txn['account_id'], needed, deposit_date)
            
            print(f"Inserted seed deposit of ${needed:,.2f}")
            
            # Recalculate again
            print("Recalculating balances again...")
            txns = await conn.fetch("SELECT id, amount FROM transactions ORDER BY transaction_date ASC")
            balance = Decimal(0)
            for txn in txns:
                amount = txn['amount']
                balance += amount
                await conn.execute("UPDATE transactions SET balance_after = $1 WHERE id = $2", balance, txn['id'])

    await conn.close()

if __name__ == "__main__":
    asyncio.run(reconcile())
