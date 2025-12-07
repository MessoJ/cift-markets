"""
Alpaca Sync Service
Synchronizes local database with Alpaca Brokerage Account
"""
import asyncio
from typing import Dict, Any, List
from uuid import UUID
from decimal import Decimal

from loguru import logger

from cift.core.database import get_postgres_pool
from cift.integrations.alpaca import alpaca_client

class AlpacaSyncService:
    """
    Service to sync local database with Alpaca
    """
    
    async def sync_all_accounts(self):
        """Sync all users with Alpaca accounts"""
        if not alpaca_client.is_configured:
            logger.warning("Alpaca not configured, skipping sync")
            return

        pool = await get_postgres_pool()
        
        async with pool.acquire() as conn:
            # Get all users with Alpaca IDs
            users = await conn.fetch(
                "SELECT id, alpaca_account_id FROM users WHERE alpaca_account_id IS NOT NULL"
            )
            
        logger.info(f"Syncing {len(users)} Alpaca accounts")
        
        for user in users:
            try:
                await self.sync_user_account(user['id'], user['alpaca_account_id'])
            except Exception as e:
                logger.error(f"Failed to sync user {user['id']}: {str(e)}")

    async def sync_user_account(self, user_id: UUID, alpaca_id: str):
        """Sync a single user's account and positions"""
        # 1. Get Account Data from Alpaca (using Master Account for now, or Sub-Account endpoint)
        # Note: In a real Broker API integration, we would use the Broker Client to fetch 
        # the specific sub-account's details. 
        # For this implementation, assuming we are using the Trading API keys which map to ONE account 
        # (the master or the single user), we will just fetch `get_account`.
        # If we were using Broker API, we'd need `get_account(account_id=alpaca_id)`.
        
        # Assuming Trading API for now as per `cift/integrations/alpaca.py` implementation
        account = await alpaca_client.get_account()
        
        # 2. Update Local Account
        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            # Get local account ID
            local_account_id = await conn.fetchval(
                "SELECT id FROM accounts WHERE user_id = $1", user_id
            )
            
            if not local_account_id:
                logger.warning(f"No local account found for user {user_id}")
                return

            # Update Balances
            await conn.execute(
                """
                UPDATE accounts 
                SET cash_balance = $1,
                    buying_power = $2,
                    equity = $3,
                    updated_at = NOW()
                WHERE id = $4
                """,
                Decimal(account.get('cash', 0)),
                Decimal(account.get('buying_power', 0)),
                Decimal(account.get('equity', 0)),
                local_account_id
            )
            
            # 3. Sync Positions
            positions = await alpaca_client.get_positions()
            
            # Get current local positions to detect closed ones
            local_positions = await conn.fetch(
                "SELECT symbol FROM positions WHERE account_id = $1 AND quantity != 0",
                local_account_id
            )
            local_symbols = {row['symbol'] for row in local_positions}
            remote_symbols = set()

            for pos in positions:
                symbol = pos['symbol']
                remote_symbols.add(symbol)
                qty = Decimal(pos['qty'])
                market_value = Decimal(pos['market_value'])
                avg_entry_price = Decimal(pos['avg_entry_price'])
                current_price = Decimal(pos['current_price'])
                unrealized_pl = Decimal(pos['unrealized_pl'])
                
                # Upsert Position
                await conn.execute(
                    """
                    INSERT INTO positions (
                        account_id, symbol, quantity, avg_cost, 
                        current_price, market_value, unrealized_pnl,
                        side, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                    ON CONFLICT (account_id, symbol) DO UPDATE SET
                        quantity = EXCLUDED.quantity,
                        avg_cost = EXCLUDED.avg_cost,
                        current_price = EXCLUDED.current_price,
                        market_value = EXCLUDED.market_value,
                        unrealized_pnl = EXCLUDED.unrealized_pnl,
                        updated_at = NOW()
                    """,
                    local_account_id,
                    symbol,
                    qty,
                    avg_entry_price,
                    current_price,
                    market_value,
                    unrealized_pl,
                    'long' if qty > 0 else 'short'
                )
            
            # 4. Handle Closed Positions
            closed_symbols = local_symbols - remote_symbols
            for symbol in closed_symbols:
                await conn.execute(
                    """
                    UPDATE positions 
                    SET quantity = 0, market_value = 0, updated_at = NOW()
                    WHERE account_id = $1 AND symbol = $2
                    """,
                    local_account_id, symbol
                )

alpaca_sync_service = AlpacaSyncService()
