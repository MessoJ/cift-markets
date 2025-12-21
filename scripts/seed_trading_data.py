"""
CIFT Markets - Seed Trading Data Script

Creates realistic trading data for admin user to make the dashboard, portfolio,
and transaction history look full and operational.

This script creates:
- Multiple positions (current holdings)
- Order history (filled, partial, cancelled)
- Transaction history (trades, dividends, fees)
- Portfolio snapshots for equity curve
- Market data entries

Usage:
    python -m scripts.seed_trading_data
"""

import asyncio
import os
import random
from datetime import datetime, timedelta
from uuid import UUID, uuid4

import asyncpg
from loguru import logger

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============================================================================
# CONFIGURATION
# ============================================================================

# Database connection from environment
DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
DB_PORT = int(os.getenv('POSTGRES_PORT', '5432'))
DB_NAME = os.getenv('POSTGRES_DB', 'cift_markets')
DB_USER = os.getenv('POSTGRES_USER', 'cift_user')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'changeme123')

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Symbols to trade
SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "META", "JPM", "V", "JNJ",
    "WMT", "PG", "UNH", "MA", "HD"
]

# Portfolio configuration
INITIAL_CASH = 100000.00  # $100k starting capital
NUM_POSITIONS = 8  # Number of current holdings
NUM_PAST_TRADES = 50  # Number of historical closed positions
NUM_ORDERS = 75  # Total orders (mix of filled, partial, cancelled)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def random_date(start_days_ago: int, end_days_ago: int = 0) -> datetime:
    """Generate random datetime between start_days_ago and end_days_ago."""
    start = datetime.utcnow() - timedelta(days=start_days_ago)
    end = datetime.utcnow() - timedelta(days=end_days_ago)

    delta = end - start
    random_seconds = random.random() * delta.total_seconds()
    return start + timedelta(seconds=random_seconds)

def random_price(symbol: str) -> float:
    """Generate realistic price for symbol."""
    prices = {
        "AAPL": (170, 195),
        "MSFT": (350, 420),
        "GOOGL": (135, 155),
        "AMZN": (145, 180),
        "NVDA": (450, 550),
        "TSLA": (230, 290),
        "META": (350, 490),
        "JPM": (140, 170),
        "V": (240, 280),
        "JNJ": (150, 170),
        "WMT": (150, 170),
        "PG": (140, 160),
        "UNH": (480, 550),
        "MA": (400, 480),
        "HD": (320, 380),
    }

    min_price, max_price = prices.get(symbol, (50, 200))
    return round(random.uniform(min_price, max_price), 2)

def random_quantity() -> int:
    """Generate realistic share quantity."""
    # More common to buy smaller amounts
    if random.random() < 0.7:
        return random.choice([1, 2, 5, 10, 15, 20, 25, 50])
    else:
        return random.choice([75, 100, 150, 200, 500])

# ============================================================================
# MAIN SEEDING LOGIC
# ============================================================================

async def get_admin_user_id(conn: asyncpg.Connection) -> UUID:
    """Get or create admin user."""
    # Try to get existing admin by email or username
    row = await conn.fetchrow(
        "SELECT id FROM users WHERE email = $1 OR username = $2 LIMIT 1",
        "admin@cift.markets", "admin"
    )

    if row:
        logger.info(f"Found existing admin user: {row['id']}")
        return row['id']

    # Create admin if doesn't exist
    from cift.core.auth import hash_password

    user_id = uuid4()
    try:
        await conn.execute("""
            INSERT INTO users (id, email, username, hashed_password, full_name, is_active, is_superuser)
            VALUES ($1, $2, $3, $4, $5, true, true)
        """, user_id, "admin@cift.markets", "admin", hash_password("admin123"), "CIFT Admin")

        logger.info(f"Created admin user: {user_id}")
        return user_id
    except Exception as e:
        # If creation failed, try to fetch again (race condition)
        logger.warning(f"Failed to create admin user (may already exist): {e}")
        row = await conn.fetchrow(
            "SELECT id FROM users WHERE email = $1 OR username = $2 LIMIT 1",
            "admin@cift.markets", "admin"
        )
        if row:
            logger.info(f"Retrieved existing admin user after creation failure: {row['id']}")
            return row['id']
        raise


async def get_account_id(conn: asyncpg.Connection, user_id: UUID) -> UUID:
    """Get or create trading account for user."""
    # Try to get existing account
    row = await conn.fetchrow(
        "SELECT id FROM accounts WHERE user_id = $1 AND is_active = true LIMIT 1",
        user_id
    )

    if row:
        logger.info(f"Found existing account: {row['id']}")
        return row['id']

    # Create account
    account_id = uuid4()
    account_number = f"CIFT{random.randint(10000000, 99999999)}"

    await conn.execute("""
        INSERT INTO accounts (
            id, user_id, account_number, account_type, account_name,
            cash_balance, buying_power, equity, margin_used,
            is_active, is_pattern_day_trader
        ) VALUES (
            $1, $2, $3, 'margin', 'Primary Trading Account',
            $4, $5, $4, 0, true, false
        )
    """, account_id, user_id, account_number, INITIAL_CASH, INITIAL_CASH * 2)

    logger.info(f"Created trading account: {account_id}")
    return account_id


async def seed_positions(conn: asyncpg.Connection, user_id: UUID, account_id: UUID):
    """Create current positions (holdings)."""
    logger.info(f"Creating {NUM_POSITIONS} positions...")

    # Select random symbols for current holdings
    holdings = random.sample(SYMBOLS, NUM_POSITIONS)

    positions_data = []
    for symbol in holdings:
        # Generate position details
        quantity = random_quantity()
        avg_cost = random_price(symbol)
        current_price = avg_cost * random.uniform(0.85, 1.25)  # -15% to +25% P&L

        # Calculate P&L
        total_cost = avg_cost * quantity
        market_value = current_price * quantity
        unrealized_pnl = market_value - total_cost
        realized_pnl = random.uniform(-500, 1500)  # From previous trades
        day_pnl = random.uniform(-200, 400)

        position = {
            'id': uuid4(),
            'user_id': user_id,
            'account_id': account_id,
            'symbol': symbol,
            'quantity': quantity,
            'side': 'long',
            'avg_cost': round(avg_cost, 2),
            'total_cost': round(total_cost, 2),
            'current_price': round(current_price, 2),
            'market_value': round(market_value, 2),
            'unrealized_pnl': round(unrealized_pnl, 2),
            'unrealized_pnl_pct': round((unrealized_pnl / total_cost * 100) if total_cost > 0 else 0, 2),
            'realized_pnl': round(realized_pnl, 2),
            'day_pnl': round(day_pnl, 2),
            'day_pnl_pct': round((day_pnl / market_value * 100) if market_value > 0 else 0, 2),
            'opened_at': random_date(180, 7),
            'updated_at': datetime.utcnow()
        }

        positions_data.append(position)

    # Insert positions
    await conn.executemany("""
        INSERT INTO positions (
            id, user_id, account_id, symbol, quantity, side, avg_cost, total_cost,
            current_price, market_value, unrealized_pnl, unrealized_pnl_pct,
            realized_pnl, day_pnl, day_pnl_pct, opened_at, updated_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17
        )
        ON CONFLICT (account_id, symbol) DO UPDATE SET
            quantity = EXCLUDED.quantity,
            avg_cost = EXCLUDED.avg_cost,
            total_cost = EXCLUDED.total_cost,
            current_price = EXCLUDED.current_price,
            market_value = EXCLUDED.market_value,
            unrealized_pnl = EXCLUDED.unrealized_pnl,
            unrealized_pnl_pct = EXCLUDED.unrealized_pnl_pct,
            realized_pnl = EXCLUDED.realized_pnl,
            day_pnl = EXCLUDED.day_pnl,
            day_pnl_pct = EXCLUDED.day_pnl_pct,
            updated_at = EXCLUDED.updated_at
    """, [(
        p['id'], p['user_id'], p['account_id'], p['symbol'], p['quantity'], p['side'],
        p['avg_cost'], p['total_cost'], p['current_price'], p['market_value'],
        p['unrealized_pnl'], p['unrealized_pnl_pct'], p['realized_pnl'],
        p['day_pnl'], p['day_pnl_pct'], p['opened_at'], p['updated_at']
    ) for p in positions_data])

    logger.success(f"✅ Created {len(positions_data)} positions")
    return positions_data


async def seed_orders(conn: asyncpg.Connection, user_id: UUID, account_id: UUID, positions_data: list):
    """Create order history."""
    logger.info(f"Creating {NUM_ORDERS} orders...")

    orders_data = []

    # Create orders for current positions (buy orders)
    for position in positions_data:
        num_buys = random.randint(1, 4)  # 1-4 buy orders per position

        for _ in range(num_buys):
            order = {
                'id': uuid4(),
                'user_id': user_id,
                'account_id': account_id,
                'symbol': position['symbol'],
                'side': 'buy',
                'order_type': random.choice(['market', 'market', 'limit']),
                'time_in_force': 'day',
                'quantity': random.randint(1, position['quantity']),
                'filled_quantity': 0,  # Will set below
                'remaining_quantity': 0,
                'limit_price': None,
                'stop_price': None,
                'avg_fill_price': position['avg_cost'] * random.uniform(0.95, 1.05),
                'status': 'filled',
                'total_value': 0,  # Will calculate
                'commission': round(random.uniform(0, 2.5), 2),
                'created_at': random_date(180, 30),
                'filled_at': None,
                'cancelled_at': None
            }

            order['filled_quantity'] = order['quantity']
            order['total_value'] = round(order['quantity'] * order['avg_fill_price'], 2)
            order['filled_at'] = order['created_at'] + timedelta(seconds=random.randint(1, 300))

            if order['order_type'] == 'limit':
                order['limit_price'] = round(order['avg_fill_price'] * random.uniform(0.98, 1.02), 2)

            orders_data.append(order)

    # Create some sell orders (realized P&L)
    for _ in range(NUM_PAST_TRADES):
        symbol = random.choice(SYMBOLS)
        quantity = random_quantity()
        price = random_price(symbol)

        order = {
            'id': uuid4(),
            'user_id': user_id,
            'account_id': account_id,
            'symbol': symbol,
            'side': random.choice(['buy', 'sell']),
            'order_type': random.choice(['market', 'market', 'market', 'limit']),
            'time_in_force': random.choice(['day', 'gtc']),
            'quantity': quantity,
            'filled_quantity': quantity,
            'remaining_quantity': 0,
            'limit_price': None,
            'stop_price': None,
            'avg_fill_price': price,
            'status': 'filled',
            'total_value': round(quantity * price, 2),
            'commission': round(random.uniform(0, 2.5), 2),
            'created_at': random_date(180, 1),
            'filled_at': None,
            'cancelled_at': None
        }

        order['filled_at'] = order['created_at'] + timedelta(seconds=random.randint(1, 300))

        if order['order_type'] == 'limit':
            order['limit_price'] = round(price * random.uniform(0.98, 1.02), 2)

        orders_data.append(order)

    # Create some cancelled/pending/partial orders
    for _ in range(random.randint(5, 15)):
        symbol = random.choice(SYMBOLS)
        quantity = random_quantity()
        price = random_price(symbol)

        status = random.choice(['pending', 'accepted', 'cancelled', 'partial'])

        order = {
            'id': uuid4(),
            'user_id': user_id,
            'account_id': account_id,
            'symbol': symbol,
            'side': random.choice(['buy', 'sell']),
            'order_type': 'limit',  # Open/cancelled usually limit orders
            'time_in_force': random.choice(['day', 'gtc']),
            'quantity': quantity,
            'filled_quantity': quantity // 2 if status == 'partial' else 0,
            'remaining_quantity': quantity // 2 if status == 'partial' else quantity,
            'limit_price': round(price * random.uniform(0.95, 1.05), 2),
            'stop_price': None,
            'avg_fill_price': price if status == 'partial' else None,
            'status': status,
            'total_value': round(quantity * price / 2, 2) if status == 'partial' else 0,
            'commission': round(random.uniform(0, 1.5), 2) if status == 'partial' else 0,
            'created_at': random_date(30, 0),
            'filled_at': random_date(30, 0) if status == 'partial' else None,
            'cancelled_at': random_date(15, 0) if status == 'cancelled' else None
        }

        orders_data.append(order)

    # Insert orders
    await conn.executemany("""
        INSERT INTO orders (
            id, user_id, account_id, symbol, side, order_type, time_in_force,
            quantity, filled_quantity, remaining_quantity, limit_price, stop_price,
            avg_fill_price, status, total_value, commission,
            created_at, filled_at, cancelled_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19
        )
        ON CONFLICT (id) DO NOTHING
    """, [(
        o['id'], o['user_id'], o['account_id'], o['symbol'], o['side'],
        o['order_type'], o['time_in_force'], o['quantity'], o['filled_quantity'],
        o['remaining_quantity'], o['limit_price'], o['stop_price'],
        o['avg_fill_price'], o['status'], o['total_value'], o['commission'],
        o['created_at'], o['filled_at'], o['cancelled_at']
    ) for o in orders_data])

    logger.success(f"✅ Created {len(orders_data)} orders")
    return orders_data


async def seed_transactions(conn: asyncpg.Connection, user_id: UUID, account_id: UUID, orders_data: list):
    """Create transaction history."""
    logger.info("Creating transactions...")

    transactions_data = []

    # Create transactions for filled orders
    for order in orders_data:
        if order['status'] in ['filled', 'partial'] and order['filled_quantity'] > 0:
            # Trade transaction
            amount = -order['total_value'] if order['side'] == 'buy' else order['total_value']

            transaction = {
                'id': uuid4(),
                'user_id': user_id,
                'account_id': account_id,
                'transaction_type': 'trade',
                'amount': round(amount, 2),
                'balance_after': 0,  # Will calculate later
                'symbol': order['symbol'],
                'description': f"{order['side'].upper()} {order['filled_quantity']} {order['symbol']} @ ${order['avg_fill_price']:.2f}",
                'order_id': order['id'],
                'external_ref': f"TRD-{order['id']}"[:16],
                'transaction_date': order['filled_at'] or order['created_at'],
                'created_at': order['filled_at'] or order['created_at']
            }

            transactions_data.append(transaction)

            # Commission transaction
            if order['commission'] > 0:
                commission_txn = {
                    'id': uuid4(),
                    'user_id': user_id,
                    'account_id': account_id,
                    'transaction_type': 'commission',
                    'amount': -order['commission'],
                    'balance_after': 0,
                    'symbol': order['symbol'],
                    'description': f"Commission for {order['symbol']} trade",
                    'order_id': order['id'],
                    'external_ref': f"COM-{order['id']}"[:16],
                    'transaction_date': order['filled_at'] or order['created_at'],
                    'created_at': order['filled_at'] or order['created_at']
                }

                transactions_data.append(commission_txn)

    # Add some dividends
    for _ in range(random.randint(5, 15)):
        symbol = random.choice(["AAPL", "MSFT", "JPM", "JNJ", "PG", "WMT"])
        amount = round(random.uniform(10, 200), 2)

        dividend = {
            'id': uuid4(),
            'user_id': user_id,
            'account_id': account_id,
            'transaction_type': 'dividend',
            'amount': amount,
            'balance_after': 0,
            'symbol': symbol,
            'description': f"Dividend from {symbol}",
            'order_id': None,
            'external_ref': f"DIV-{uuid4()}"[:16],
            'transaction_date': random_date(90, 5),
            'created_at': random_date(90, 5)
        }

        transactions_data.append(dividend)

    # Add initial deposit
    initial_deposit = {
        'id': uuid4(),
        'user_id': user_id,
        'account_id': account_id,
        'transaction_type': 'deposit',
        'amount': INITIAL_CASH,
        'balance_after': INITIAL_CASH,
        'symbol': None,
        'description': "Initial deposit",
        'order_id': None,
        'external_ref': "DEP-INIT",
        'transaction_date': random_date(180, 179),
        'created_at': random_date(180, 179)
    }

    transactions_data.append(initial_deposit)

    # Sort by date
    transactions_data.sort(key=lambda x: x['transaction_date'])

    # Calculate running balance
    balance = 0
    for txn in transactions_data:
        balance += txn['amount']
        txn['balance_after'] = round(balance, 2)

    # Insert transactions
    await conn.executemany("""
        INSERT INTO transactions (
            id, user_id, account_id, transaction_type, amount, balance_after,
            symbol, description, order_id, external_ref,
            transaction_date, created_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
        )
        ON CONFLICT (id) DO NOTHING
    """, [(
        t['id'], t['user_id'], t['account_id'], t['transaction_type'],
        t['amount'], t['balance_after'], t['symbol'],
        t['description'], t['order_id'], t['external_ref'],
        t['transaction_date'], t['created_at']
    ) for t in transactions_data])

    logger.success(f"✅ Created {len(transactions_data)} transactions")
    return transactions_data


async def seed_portfolio_snapshots(conn: asyncpg.Connection, user_id: UUID, account_id: UUID, positions_data: list):
    """Create historical portfolio snapshots for equity curve."""
    logger.info("Creating portfolio snapshots...")

    snapshots = []

    # Create daily snapshots for last 90 days
    for days_ago in range(90, -1, -1):
        snapshot_date = datetime.utcnow() - timedelta(days=days_ago)

        # Calculate portfolio value at that time
        # Add some randomness but trending up
        base_value = INITIAL_CASH
        trend = (90 - days_ago) / 90 * 15000  # Trending up $15k over 90 days
        volatility = random.uniform(-2000, 2000)

        total_value = base_value + trend + volatility

        # Calculate positions value
        positions_value = sum(p['market_value'] for p in positions_data)
        cash = total_value - positions_value

        unrealized_pnl = sum(p['unrealized_pnl'] for p in positions_data)
        realized_pnl = random.uniform(-500, trend)  # Realized gains accumulate

        snapshot = {
            'id': uuid4(),
            'user_id': user_id,
            'account_id': account_id,
            'timestamp': snapshot_date,
            'snapshot_type': 'eod',
            'total_value': round(total_value, 2),
            'cash': max(0, round(cash, 2)),
            'positions_value': round(positions_value, 2),
            'equity': round(total_value, 2),
            'unrealized_pnl': round(unrealized_pnl, 2),
            'realized_pnl': round(realized_pnl, 2),
            'day_pnl': round(random.uniform(-1000, 1500), 2),
            'day_pnl_pct': round(random.uniform(-2, 3), 2),
            'created_at': snapshot_date
        }

        snapshots.append(snapshot)

    # Insert snapshots
    await conn.executemany("""
        INSERT INTO portfolio_snapshots (
            id, user_id, account_id, timestamp, snapshot_type, total_value, cash,
            positions_value, equity, unrealized_pnl, realized_pnl, day_pnl, day_pnl_pct, created_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
        )
        ON CONFLICT (id) DO UPDATE SET
            total_value = EXCLUDED.total_value,
            cash = EXCLUDED.cash,
            positions_value = EXCLUDED.positions_value,
            equity = EXCLUDED.equity,
            unrealized_pnl = EXCLUDED.unrealized_pnl,
            realized_pnl = EXCLUDED.realized_pnl,
            day_pnl = EXCLUDED.day_pnl,
            day_pnl_pct = EXCLUDED.day_pnl_pct,
            created_at = EXCLUDED.created_at
    """, [(
        s['id'], s['user_id'], s['account_id'], s['timestamp'], s['snapshot_type'],
        s['total_value'], s['cash'], s['positions_value'], s['equity'],
        s['unrealized_pnl'], s['realized_pnl'], s['day_pnl'], s['day_pnl_pct'],
        s['created_at']
    ) for s in snapshots])

    logger.success(f"✅ Created {len(snapshots)} portfolio snapshots")


async def seed_data():
    """Main seeding function."""
    logger.info("🌱 Starting seed data generation...")
    logger.info(f"📡 Connecting to database: {DB_HOST}:{DB_PORT}/{DB_NAME} as {DB_USER}")

    # Connect to database
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        logger.success("✅ Connected to database successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to connect to database: {e}")
        logger.info(f"💡 Connection string: postgresql://{DB_USER}:***@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        logger.info("💡 Make sure PostgreSQL is running and accessible")
        raise

    try:
        # Get admin user and account
        user_id = await get_admin_user_id(conn)
        account_id = await get_account_id(conn, user_id)

        # Seed in order
        positions_data = await seed_positions(conn, user_id, account_id)
        orders_data = await seed_orders(conn, user_id, account_id, positions_data)
        transactions_data = await seed_transactions(conn, user_id, account_id, orders_data)
        await seed_portfolio_snapshots(conn, user_id, account_id, positions_data)

        logger.success("🎉 Seed data generation complete!")
        logger.info("📊 Summary:")
        logger.info(f"   - User ID: {user_id}")
        logger.info(f"   - Positions: {len(positions_data)}")
        logger.info(f"   - Orders: {len(orders_data)}")
        logger.info(f"   - Transactions: {len(transactions_data)}")
        logger.info("   - Portfolio snapshots: 91 days")
        logger.info(f"   - Initial cash: ${INITIAL_CASH:,.2f}")

        # Calculate summary
        total_pnl = sum(p['unrealized_pnl'] + p['realized_pnl'] for p in positions_data)
        total_value = sum(p['market_value'] for p in positions_data)
        logger.info(f"   - Current value: ${total_value:,.2f}")
        logger.info(f"   - Total P&L: ${total_pnl:,.2f} ({total_pnl/INITIAL_CASH*100:.1f}%)")

    except Exception as e:
        logger.error(f"❌ Error seeding data: {e}")
        raise
    finally:
        await conn.close()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    asyncio.run(seed_data())
