# Seed Trading Data Script

## Overview

This script populates the CIFT Markets database with realistic trading data for the admin user, making the dashboard, portfolio, and transaction history look full and operational.

## What It Creates

### 📊 Data Generated

1. **Admin User**
   - Email: `admin@cift.markets`
   - Password: `admin123`
   - Superuser privileges

2. **Trading Account**
   - Account type: Margin
   - Initial cash: $100,000
   - Buying power: $200,000 (2x margin)

3. **8 Current Positions**
   - Random selection from 15 major symbols (AAPL, MSFT, GOOGL, etc.)
   - Realistic quantities and prices
   - Mix of winning and losing positions
   - P&L calculations (unrealized + realized)

4. **75+ Orders**
   - Filled orders (creating current positions)
   - Historical closed positions
   - Open limit orders
   - Cancelled orders
   - Partial fills

5. **Transaction History**
   - All trades (buys and sells)
   - Commissions
   - Dividends (5-15 dividend payments)
   - Initial deposit

6. **91 Days of Portfolio Snapshots**
   - Daily equity curve data
   - Shows portfolio growth over time
   - Used for analytics charts

## Prerequisites

- PostgreSQL database running
- Database initialized with schema
- Python dependencies installed

## Usage

### 1. Run the Seed Script

```bash
# From project root
python -m scripts.seed_trading_data
```

### 2. Expected Output

```
🌱 Starting seed data generation...
Found existing admin user: <uuid>
Created trading account: <uuid>
Creating 8 positions...
✅ Created 8 positions
Creating 75 orders...
✅ Created 75 orders
Creating transactions...
✅ Created 200+ transactions
Creating portfolio snapshots...
✅ Created 91 portfolio snapshots
🎉 Seed data generation complete!

📊 Summary:
   - User ID: <uuid>
   - Positions: 8
   - Orders: 75
   - Transactions: 200+
   - Portfolio snapshots: 91 days
   - Initial cash: $100,000.00
   - Current value: $115,000.00
   - Total P&L: $15,000.00 (15.0%)
```

## Login Credentials

After seeding, you can login with:

- **Email**: `admin@cift.markets`
- **Password**: `admin123`

## What You'll See

### Dashboard
- Portfolio summary with realistic values
- 8 open positions with P&L
- Recent orders (mix of filled, partial, cancelled)
- Equity curve chart showing growth

### Portfolio Page
- All current holdings
- Sector allocation
- P&L breakdown
- Real-time calculations

### Orders Page
- Historical filled orders
- Some open limit orders
- Cancelled orders
- Order details with fills

### Transactions Page
- Complete transaction history
- Trades, commissions, dividends
- Cash flow analysis
- Filterable by type and date

### Charts
- 90-day equity curve
- Portfolio growth visualization
- P&L trends

## Customization

You can modify the script configuration at the top:

```python
# Portfolio configuration
INITIAL_CASH = 100000.00  # Starting capital
NUM_POSITIONS = 8          # Current holdings
NUM_PAST_TRADES = 50       # Historical trades
NUM_ORDERS = 75            # Total order history

# Symbols to use
SYMBOLS = ["AAPL", "MSFT", "GOOGL", ...]
```

## Re-running

The script uses `ON CONFLICT` clauses, so you can run it multiple times safely. It will:
- Update existing data if found
- Create new data if missing
- Not duplicate records

## Troubleshooting

### Database Connection Error

```bash
# Check PostgreSQL is running
pg_isready

# Verify connection string
psql "postgresql://cift_admin:cift_dev_2024@localhost:5432/cift_markets"
```

### Import Error

```bash
# Make sure you're in the project root
cd /path/to/cift-markets

# Install dependencies
pip install asyncpg loguru
```

### Schema Not Found

Run database migrations first:

```bash
psql -U cift_admin -d cift_markets -f database/init.sql
psql -U cift_admin -d cift_markets -f database/migrations/001_add_drilldown_tables.sql
```

## Data Characteristics

### Realistic Features

- **Price Ranges**: Based on actual stock prices
- **P&L Distribution**: Mix of winners (70%) and losers (30%)
- **Order Types**: Mostly market orders (70%), some limit (30%)
- **Timeframes**: Orders spread over 180 days, positions opened 7-180 days ago
- **Commissions**: $0-2.50 per trade
- **Dividends**: From dividend-paying stocks only
- **Portfolio Growth**: Trending upward with volatility

### Trade Patterns

- 1-4 buy orders per current position (showing accumulation)
- Historical sells (showing realized P&L)
- Some day trades (same symbol buy/sell same day)
- Open limit orders waiting to fill

## Next Steps

After seeding:

1. **Start the backend**: `python -m cift.api.main`
2. **Start the frontend**: `cd frontend && npm run dev`
3. **Login** with admin credentials
4. **Explore** the full trading dashboard

The application will now show realistic data across all pages!
