#!/usr/bin/env python3
"""
Seed portfolio_snapshots table with historical data for analytics.

This script generates realistic portfolio value history over time,
simulating growth/decline based on market-like patterns.
"""

import asyncio
import asyncpg
import random
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://cift_user:changeme123@localhost:5432/cift_markets"
)

async def get_user_accounts(pool: asyncpg.Pool) -> List[Dict[str, Any]]:
    """Get all user accounts for seeding."""
    query = """
        SELECT 
            a.id as account_id,
            a.user_id,
            a.cash,
            a.portfolio_value,
            a.equity,
            u.email
        FROM accounts a
        JOIN users u ON u.id = a.user_id
        WHERE a.is_active = true
    """
    rows = await pool.fetch(query)
    return [dict(row) for row in rows]


async def get_positions_value(pool: asyncpg.Pool, account_id: str) -> float:
    """Get total positions value for an account."""
    query = """
        SELECT COALESCE(SUM(market_value), 0) as positions_value
        FROM positions
        WHERE account_id = $1
    """
    result = await pool.fetchval(query, account_id)
    return float(result) if result else 0.0


def generate_portfolio_series(
    initial_value: float,
    initial_cash: float,
    initial_positions: float,
    days: int = 30,
    volatility: float = 0.02
) -> List[Dict[str, Any]]:
    """
    Generate a realistic portfolio value time series.
    
    Uses geometric Brownian motion with mean reversion for
    realistic equity curve simulation.
    """
    series = []
    
    # Current values (will evolve over time)
    total_value = initial_value
    cash = initial_cash
    positions_value = initial_positions
    
    # Starting point - go back 'days' days
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)
    
    # Daily drift (slight upward bias for bull market)
    daily_drift = 0.0003  # ~7.5% annual
    
    # Track P&L
    cumulative_realized_pnl = 0.0
    
    for day_offset in range(days):
        current_date = start_time + timedelta(days=day_offset)
        
        # Skip weekends for EOD snapshots
        if current_date.weekday() >= 5:
            continue
        
        # Generate daily return using GBM
        daily_return = random.gauss(daily_drift, volatility)
        
        # Add some realistic patterns
        # - Monday effect (slight negative)
        if current_date.weekday() == 0:
            daily_return -= 0.002
        # - Friday momentum
        if current_date.weekday() == 4:
            daily_return += 0.001
            
        # Apply return to positions (cash doesn't change from market moves)
        old_positions_value = positions_value
        positions_value = positions_value * (1 + daily_return)
        
        # Simulate occasional trades (10% chance per day)
        if random.random() < 0.1:
            # Random P&L from trades
            trade_pnl = random.gauss(50, 200)  # Mean $50 profit, $200 std
            cumulative_realized_pnl += trade_pnl
            cash += trade_pnl
            
        # Calculate unrealized P&L (change in positions value)
        unrealized_pnl = positions_value - old_positions_value
        
        # Total value = cash + positions
        total_value = cash + positions_value
        equity = total_value  # For non-margin account
        
        # Day P&L
        day_pnl = (positions_value - old_positions_value)
        day_pnl_pct = (day_pnl / old_positions_value * 100) if old_positions_value > 0 else 0
        
        # EOD timestamp (4 PM ET typically)
        eod_time = current_date.replace(hour=20, minute=0, second=0, microsecond=0)  # 4 PM ET = 20:00 UTC
        
        series.append({
            'timestamp': eod_time,
            'total_value': round(total_value, 2),
            'cash': round(cash, 2),
            'positions_value': round(positions_value, 2),
            'equity': round(equity, 2),
            'unrealized_pnl': round(positions_value - initial_positions, 2),
            'realized_pnl': round(cumulative_realized_pnl, 2),
            'day_pnl': round(day_pnl, 2),
            'day_pnl_pct': round(day_pnl_pct, 4),
            'snapshot_type': 'eod'
        })
    
    return series


async def seed_portfolio_snapshots(
    pool: asyncpg.Pool,
    user_id: str,
    account_id: str,
    snapshots: List[Dict[str, Any]]
) -> int:
    """Insert portfolio snapshots into the database."""
    
    insert_query = """
        INSERT INTO portfolio_snapshots (
            user_id, account_id, total_value, cash, positions_value,
            equity, unrealized_pnl, realized_pnl, day_pnl, day_pnl_pct,
            timestamp, snapshot_type
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        ON CONFLICT DO NOTHING
    """
    
    count = 0
    async with pool.acquire() as conn:
        for snapshot in snapshots:
            try:
                await conn.execute(
                    insert_query,
                    user_id,
                    account_id,
                    snapshot['total_value'],
                    snapshot['cash'],
                    snapshot['positions_value'],
                    snapshot['equity'],
                    snapshot['unrealized_pnl'],
                    snapshot['realized_pnl'],
                    snapshot['day_pnl'],
                    snapshot['day_pnl_pct'],
                    snapshot['timestamp'],
                    snapshot['snapshot_type']
                )
                count += 1
            except Exception as e:
                logger.error(f"Error inserting snapshot: {e}")
                
    return count


async def main():
    """Main entry point."""
    logger.info("Connecting to PostgreSQL...")
    
    pool = await asyncpg.create_pool(DATABASE_URL)
    
    try:
        # Get all user accounts
        accounts = await get_user_accounts(pool)
        logger.info(f"Found {len(accounts)} accounts to seed")
        
        for account in accounts:
            user_id = str(account['user_id'])
            account_id = str(account['account_id'])
            email = account['email']
            
            # Get current values
            cash = float(account['cash']) if account['cash'] else 10000.0
            portfolio_value = float(account['portfolio_value']) if account['portfolio_value'] else 100000.0
            positions_value = await get_positions_value(pool, account_id)
            
            # If no positions, use portfolio_value as estimate
            if positions_value == 0:
                positions_value = portfolio_value - cash if portfolio_value > cash else portfolio_value * 0.8
            
            total_value = cash + positions_value
            
            logger.info(f"Seeding snapshots for {email}...")
            logger.info(f"  Current values: cash=${cash:.2f}, positions=${positions_value:.2f}, total=${total_value:.2f}")
            
            # Generate 60 days of history (will be ~42 trading days)
            snapshots = generate_portfolio_series(
                initial_value=total_value * 0.85,  # Start at 85% of current value (showing growth)
                initial_cash=cash * 1.2,  # Had more cash before
                initial_positions=positions_value * 0.7,  # Fewer positions before
                days=60,
                volatility=0.015  # 1.5% daily volatility
            )
            
            # Insert snapshots
            count = await seed_portfolio_snapshots(pool, user_id, account_id, snapshots)
            logger.info(f"  Inserted {count} portfolio snapshots")
            
        logger.info("Portfolio snapshot seeding complete!")
        
        # Verify data
        verification = await pool.fetchval(
            "SELECT COUNT(*) FROM portfolio_snapshots"
        )
        logger.info(f"Total portfolio snapshots in database: {verification}")
        
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
