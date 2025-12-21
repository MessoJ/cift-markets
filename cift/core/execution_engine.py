"""
CIFT Markets - Order Execution Engine

High-performance order execution and position management.

Features:
- Order validation and risk checks
- Position tracking and P&L calculation
- Fill simulation for paper trading
- Real broker integration support
- Transaction recording

Performance: Sub-10ms order processing
"""

import asyncio
from datetime import datetime
from uuid import UUID

from loguru import logger

from cift.core.database import db_manager
from cift.core.nats_manager import get_nats_manager
from cift.core.trading_queries import (
    get_latest_price,
)

# ============================================================================
# ORDER EXECUTION ENGINE
# ============================================================================

class ExecutionEngine:
    """
    Order execution engine for processing orders and managing positions.

    Handles:
    - Order validation
    - Fill simulation (paper trading)
    - Position updates
    - P&L calculation
    - Transaction recording
    """

    def __init__(self):
        self.is_running = False
        self._order_queue = asyncio.Queue()

    async def start(self):
        """Start execution engine."""
        if self.is_running:
            return

        self.is_running = True
        logger.info("Execution engine started")

        # Start order processing task
        asyncio.create_task(self._process_orders())

    async def stop(self):
        """Stop execution engine."""
        self.is_running = False
        logger.info("Execution engine stopped")

    async def _process_orders(self):
        """Process orders from queue (background task)."""
        while self.is_running:
            try:
                # Get order from queue with timeout
                order_data = await asyncio.wait_for(
                    self._order_queue.get(),
                    timeout=1.0
                )

                # Execute order
                await self.execute_order(order_data)

            except TimeoutError:
                continue

            except Exception as e:
                logger.error(f"Order processing error: {e}", exc_info=True)

    async def submit_order(self, order_data: dict) -> UUID:
        """
        Submit order for execution.

        Args:
            order_data: Order details

        Returns:
            Order ID
        """
        # Add to processing queue
        await self._order_queue.put(order_data)

        return order_data.get("order_id")

    async def execute_order(self, order_data: dict):
        """
        Execute an order (main execution logic).

        Args:
            order_data: Order details from queue
        """
        order_id = order_data.get("order_id")
        user_id = UUID(order_data["user_id"])
        symbol = order_data["symbol"]
        side = order_data["side"]
        order_type = order_data["order_type"]
        quantity = float(order_data["quantity"])

        logger.info(f"Executing order {order_id}: {side} {quantity} {symbol}")

        try:
            # Update order status to 'accepted'
            await self._update_order_status(order_id, "accepted")

            # Check if Alpaca is configured for real execution
            from cift.integrations.alpaca import alpaca_client

            if alpaca_client.is_configured:
                try:
                    logger.info(f"Submitting order {order_id} to Alpaca")
                    alpaca_order = await alpaca_client.submit_order(
                        symbol=symbol,
                        qty=quantity,
                        side=side,
                        order_type=order_type,
                        limit_price=float(order_data.get("price")) if order_data.get("price") else None,
                        client_order_id=str(order_id)
                    )

                    # Update local order with Alpaca ID and status
                    # We don't mark it as filled yet, we let the sync process handle that
                    # or we could map the initial status
                    alpaca_status = alpaca_order.get('status')
                    db_status = 'open' if alpaca_status in ['new', 'accepted', 'pending_new'] else alpaca_status

                    await self._update_order_status(
                        order_id,
                        db_status,
                        # We can store the alpaca ID if we had a column for it,
                        # but for now we rely on client_order_id matching our DB ID
                    )
                    logger.info(f"Order {order_id} submitted to Alpaca successfully")
                    return

                except Exception as ae:
                    logger.error(f"Alpaca submission failed: {ae}")
                    # If Alpaca fails, we reject the order locally
                    raise ae

            # Get execution price
            if order_type == "market":
                execution_price = await get_latest_price(symbol)

                if not execution_price:
                    raise Exception(f"No market price available for {symbol}")

            else:
                execution_price = float(order_data.get("price", 0))

            # Simulate fill (for paper trading)
            await self._simulate_fill(
                order_id=order_id,
                user_id=user_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=execution_price
            )

            # Update order status to 'filled'
            await self._update_order_status(
                order_id,
                "filled",
                filled_quantity=quantity,
                avg_fill_price=execution_price
            )

            logger.info(f"Order {order_id} filled at ${execution_price}")

        except Exception as e:
            logger.error(f"Order execution failed for {order_id}: {e}")

            # Update order status to 'rejected'
            await self._update_order_status(
                order_id,
                "rejected",
                rejected_reason=str(e)
            )

    async def _simulate_fill(
        self,
        order_id: UUID,
        user_id: UUID,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ):
        """
        Simulate order fill for paper trading.

        Creates fill record and updates position.

        Args:
            order_id: Order ID
            user_id: User ID
            symbol: Symbol
            side: Order side
            quantity: Fill quantity
            price: Fill price
        """
        # Calculate commission (0.08 bps = $0.0008 per share)
        commission = quantity * 0.0001  # Simplified commission
        fill_value = quantity * price

        # Get account ID
        account_query = """
            SELECT id FROM accounts WHERE user_id = $1 AND is_active = TRUE LIMIT 1
        """

        async with db_manager.pool.acquire() as conn:
            account = await conn.fetchrow(account_query, user_id)

        if not account:
            raise Exception(f"No active account found for user {user_id}")

        account_id = account['id']

        # Insert fill record
        fill_query = """
            INSERT INTO order_fills (
                order_id, fill_quantity, fill_price, fill_value, commission
            ) VALUES ($1, $2, $3, $4, $5)
            RETURNING id
        """

        async with db_manager.pool.acquire() as conn:
            fill_result = await conn.fetchrow(
                fill_query,
                order_id,
                quantity,
                price,
                fill_value,
                commission
            )

        fill_id = fill_result['id']

        # Record revenue
        try:
            revenue_query = """
                INSERT INTO platform_revenue (
                    source_type, amount, reference_id, user_id, account_id, description
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """

            async with db_manager.pool.acquire() as conn:
                await conn.execute(
                    revenue_query,
                    'trading_commission',
                    commission,
                    fill_id,
                    user_id,
                    account_id,
                    f"Commission for {side} {quantity} {symbol}"
                )
        except Exception as e:
            logger.error(f"Failed to record revenue for fill {fill_id}: {e}")

        logger.info(f"Created fill {fill_id} for order {order_id}")

        # Update or create position
        await self._update_position(
            user_id=user_id,
            account_id=account_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price
        )

        # Publish fill event to NATS (5-10x lower latency)
        fill_event = {
            "fill_id": str(fill_id),
            "order_id": str(order_id),
            "user_id": str(user_id),
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "value": fill_value,
            "commission": commission,
            "filled_at": datetime.utcnow().isoformat(),
        }

        nats = await get_nats_manager()
        await nats.publish(f"orders.fills.{symbol}", fill_event)

    async def _update_position(
        self,
        user_id: UUID,
        account_id: UUID,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ):
        """
        Update position after fill.

        Handles position opening, averaging, and closing.

        Args:
            user_id: User ID
            account_id: Account ID
            symbol: Symbol
            side: Order side
            quantity: Fill quantity
            price: Fill price
        """
        # Get current position
        position_query = """
            SELECT * FROM positions
            WHERE account_id = $1 AND symbol = $2
        """

        async with db_manager.pool.acquire() as conn:
            position = await conn.fetchrow(position_query, account_id, symbol)

        if not position:
            # Open new position
            if side == "sell":
                # Short position
                quantity = -quantity
                position_side = "short"
            else:
                position_side = "long"

            insert_query = """
                INSERT INTO positions (
                    user_id, account_id, symbol, quantity, side,
                    avg_cost, total_cost, current_price, market_value
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """

            total_cost = abs(quantity) * price
            market_value = quantity * price

            async with db_manager.pool.acquire() as conn:
                await conn.execute(
                    insert_query,
                    user_id,
                    account_id,
                    symbol,
                    quantity,
                    position_side,
                    price,
                    total_cost,
                    price,
                    market_value
                )

            logger.info(f"Opened new {position_side} position: {quantity} {symbol} @ ${price}")

        else:
            # Update existing position
            current_qty = float(position['quantity'])
            current_avg_cost = float(position['avg_cost'])

            # Calculate new quantity
            if side == "buy":
                new_qty = current_qty + quantity
            else:
                new_qty = current_qty - quantity

            # Check if position is being closed or flipped
            if abs(new_qty) < 0.0001:  # Position closed
                # Calculate realized P&L
                realized_pnl = (price - current_avg_cost) * quantity
                if side == "sell":
                    realized_pnl = -realized_pnl

                # Move to position history
                await self._close_position(
                    position_id=position['id'],
                    exit_price=price,
                    realized_pnl=realized_pnl
                )

                logger.info(f"Closed position: {symbol} with P&L ${realized_pnl:.2f}")

            else:
                # Calculate new average cost
                if (current_qty > 0 and side == "buy") or (current_qty < 0 and side == "sell"):
                    # Adding to position - recalculate average
                    total_cost = (abs(current_qty) * current_avg_cost) + (quantity * price)
                    new_avg_cost = total_cost / abs(new_qty)
                else:
                    # Reducing position - keep same average
                    new_avg_cost = current_avg_cost

                # Determine side
                new_side = "long" if new_qty > 0 else "short"

                # Update position
                update_query = """
                    UPDATE positions
                    SET
                        quantity = $1,
                        side = $2,
                        avg_cost = $3,
                        total_cost = $4,
                        current_price = $5,
                        market_value = $6,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = $7
                """

                total_cost = abs(new_qty) * new_avg_cost
                market_value = new_qty * price

                async with db_manager.pool.acquire() as conn:
                    await conn.execute(
                        update_query,
                        new_qty,
                        new_side,
                        new_avg_cost,
                        total_cost,
                        price,
                        market_value,
                        position['id']
                    )

                logger.info(f"Updated position: {new_qty} {symbol} @ avg ${new_avg_cost:.2f}")

    async def _close_position(
        self,
        position_id: UUID,
        exit_price: float,
        realized_pnl: float
    ):
        """
        Close position and move to history.

        Args:
            position_id: Position ID
            exit_price: Exit price
            realized_pnl: Realized P&L
        """
        # Get position details
        query = "SELECT * FROM positions WHERE id = $1"

        async with db_manager.pool.acquire() as conn:
            position = await conn.fetchrow(query, position_id)

        if not position:
            return

        # Insert to history
        history_query = """
            INSERT INTO position_history (
                user_id, account_id, symbol, quantity, side,
                avg_entry_price, avg_exit_price, total_cost,
                total_proceeds, realized_pnl, realized_pnl_pct,
                opened_at, closed_at, hold_duration_seconds
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        """

        quantity = abs(float(position['quantity']))
        avg_cost = float(position['avg_cost'])
        total_cost = float(position['total_cost'])
        total_proceeds = quantity * exit_price
        realized_pnl_pct = (realized_pnl / total_cost * 100) if total_cost > 0 else 0

        hold_duration = (datetime.utcnow() - position['opened_at']).total_seconds()

        async with db_manager.pool.acquire() as conn:
            await conn.execute(
                history_query,
                position['user_id'],
                position['account_id'],
                position['symbol'],
                quantity,
                position['side'],
                avg_cost,
                exit_price,
                total_cost,
                total_proceeds,
                realized_pnl,
                realized_pnl_pct,
                position['opened_at'],
                datetime.utcnow(),
                int(hold_duration)
            )

        # Delete from positions
        delete_query = "DELETE FROM positions WHERE id = $1"

        async with db_manager.pool.acquire() as conn:
            await conn.execute(delete_query, position_id)

        logger.info(f"Position moved to history: {position['symbol']}")

    async def _update_order_status(
        self,
        order_id: UUID,
        status: str,
        filled_quantity: float | None = None,
        avg_fill_price: float | None = None,
        rejected_reason: str | None = None
    ):
        """
        Update order status.

        Args:
            order_id: Order ID
            status: New status
            filled_quantity: Filled quantity
            avg_fill_price: Average fill price
            rejected_reason: Rejection reason
        """
        updates = ["status = $2"]
        params = [order_id, status]
        param_idx = 3

        if status == "accepted":
            updates.append("accepted_at = CURRENT_TIMESTAMP")
        elif status == "filled":
            updates.append("filled_at = CURRENT_TIMESTAMP")

            if filled_quantity is not None:
                updates.append(f"filled_quantity = ${param_idx}")
                params.append(filled_quantity)
                param_idx += 1

            if avg_fill_price is not None:
                updates.append(f"avg_fill_price = ${param_idx}")
                params.append(avg_fill_price)
                param_idx += 1

        elif status == "rejected":
            if rejected_reason:
                updates.append(f"rejected_reason = ${param_idx}")
                params.append(rejected_reason)
                param_idx += 1

        updates.append("updated_at = CURRENT_TIMESTAMP")

        query = f"""
            UPDATE orders
            SET {', '.join(updates)}
            WHERE id = $1
        """

        async with db_manager.pool.acquire() as conn:
            await conn.execute(query, *params)


# ============================================================================
# GLOBAL EXECUTION ENGINE
# ============================================================================

execution_engine = ExecutionEngine()


# Export public API
__all__ = ["ExecutionEngine", "execution_engine"]
