"""
Transaction Settlement Service - RULES COMPLIANT
Background service to clear and settle pending funding transactions
"""

import asyncio
from datetime import datetime, timedelta

from cift.core.database import get_postgres_pool
from cift.core.logging import logger


class TransactionSettlement:
    """Handle transaction clearing and settlement - RULES COMPLIANT"""

    @staticmethod
    async def process_pending_deposits():
        """
        Process pending deposit transactions
        Clear ACH transfers that have reached their expected arrival time
        """
        pool = await get_postgres_pool()

        async with pool.acquire() as conn:
            # Find pending deposits that should be completed
            pending_deposits = await conn.fetch(
                """
                SELECT
                    id::text,
                    user_id,
                    amount,
                    payment_method_id::text,
                    expected_arrival,
                    created_at
                FROM funding_transactions
                WHERE type = 'deposit'
                AND status = 'processing'
                AND expected_arrival <= NOW()
                ORDER BY created_at ASC
                LIMIT 100
                """
            )

            completed_count = 0
            for txn in pending_deposits:
                try:
                    # Credit user account
                    await conn.execute(
                        "UPDATE accounts SET cash = cash + $1 WHERE user_id = $2 AND is_active = true",
                        txn["amount"],
                        txn["user_id"],
                    )

                    # Mark transaction as completed
                    await conn.execute(
                        """
                        UPDATE funding_transactions
                        SET status = 'completed', completed_at = NOW()
                        WHERE id = $1::uuid
                        """,
                        txn["id"],
                    )

                    completed_count += 1
                    logger.info(
                        f"Cleared deposit transaction {txn['id']} for user {txn['user_id']}"
                    )

                except Exception as e:
                    logger.error(f"Error clearing deposit {txn['id']}: {str(e)}")
                    # Mark as failed
                    await conn.execute(
                        "UPDATE funding_transactions SET status = 'failed' WHERE id = $1::uuid",
                        txn["id"],
                    )

            if completed_count > 0:
                logger.info(f"Settlement: Cleared {completed_count} deposit transactions")

            return completed_count

    @staticmethod
    async def process_pending_withdrawals():
        """
        Process pending withdrawal transactions
        Complete withdrawals that have been successfully sent
        """
        pool = await get_postgres_pool()

        async with pool.acquire() as conn:
            # Find pending withdrawals that should be completed
            pending_withdrawals = await conn.fetch(
                """
                SELECT
                    id::text,
                    user_id,
                    amount,
                    fee,
                    payment_method_id::text,
                    expected_arrival,
                    created_at
                FROM funding_transactions
                WHERE type = 'withdrawal'
                AND status = 'processing'
                AND expected_arrival <= NOW()
                ORDER BY created_at ASC
                LIMIT 100
                """
            )

            completed_count = 0
            for txn in pending_withdrawals:
                try:
                    # Mark transaction as completed
                    await conn.execute(
                        """
                        UPDATE funding_transactions
                        SET status = 'completed', completed_at = NOW()
                        WHERE id = $1::uuid
                        """,
                        txn["id"],
                    )

                    completed_count += 1
                    logger.info(
                        f"Cleared withdrawal transaction {txn['id']} for user {txn['user_id']}"
                    )

                except Exception as e:
                    logger.error(f"Error clearing withdrawal {txn['id']}: {str(e)}")
                    # Mark as failed and refund
                    await conn.execute(
                        "UPDATE funding_transactions SET status = 'failed' WHERE id = $1::uuid",
                        txn["id"],
                    )
                    # Refund to user account
                    await conn.execute(
                        "UPDATE accounts SET cash = cash + $1 WHERE user_id = $2 AND is_active = true",
                        txn["amount"] + txn["fee"],
                        txn["user_id"],
                    )
                    logger.info(f"Refunded failed withdrawal {txn['id']}")

            if completed_count > 0:
                logger.info(f"Settlement: Cleared {completed_count} withdrawal transactions")

            return completed_count

    @staticmethod
    async def check_stuck_transactions():
        """
        Check for transactions stuck in processing for too long
        Auto-fail transactions that are stuck for > 7 days
        """
        pool = await get_postgres_pool()

        async with pool.acquire() as conn:
            stuck_cutoff = datetime.utcnow() - timedelta(days=7)

            stuck_txns = await conn.fetch(
                """
                SELECT
                    id::text,
                    user_id,
                    type,
                    amount,
                    fee,
                    created_at
                FROM funding_transactions
                WHERE status = 'processing'
                AND created_at < $1
                ORDER BY created_at ASC
                LIMIT 50
                """,
                stuck_cutoff,
            )

            failed_count = 0
            for txn in stuck_txns:
                try:
                    # Mark as failed
                    await conn.execute(
                        "UPDATE funding_transactions SET status = 'failed' WHERE id = $1::uuid",
                        txn["id"],
                    )

                    # Refund if withdrawal
                    if txn["type"] == "withdrawal":
                        await conn.execute(
                            "UPDATE accounts SET cash = cash + $1 WHERE user_id = $2 AND is_active = true",
                            txn["amount"] + txn["fee"],
                            txn["user_id"],
                        )

                    failed_count += 1
                    logger.warning(
                        f"Auto-failed stuck transaction {txn['id']} (created: {txn['created_at']})"
                    )

                except Exception as e:
                    logger.error(f"Error failing stuck transaction {txn['id']}: {str(e)}")

            if failed_count > 0:
                logger.warning(f"Settlement: Failed {failed_count} stuck transactions")

            return failed_count

    @staticmethod
    async def run_settlement_cycle():
        """
        Run a complete settlement cycle
        Process deposits, withdrawals, and check for stuck transactions
        """
        logger.info("Starting settlement cycle...")

        try:
            deposits = await TransactionSettlement.process_pending_deposits()
            withdrawals = await TransactionSettlement.process_pending_withdrawals()
            stuck = await TransactionSettlement.check_stuck_transactions()

            logger.info(
                f"Settlement cycle complete: {deposits} deposits, {withdrawals} withdrawals, {stuck} stuck"
            )

            return {
                "deposits_cleared": deposits,
                "withdrawals_cleared": withdrawals,
                "stuck_failed": stuck,
            }
        except Exception as e:
            logger.error(f"Settlement cycle error: {str(e)}")
            return {"error": str(e)}

    @staticmethod
    async def start_background_settlement(interval_seconds: int = 60):
        """
        Start background settlement task

        Args:
            interval_seconds: How often to run settlement (default: 60 seconds)
        """
        logger.info(f"Starting background settlement task (interval: {interval_seconds}s)")

        while True:
            try:
                await TransactionSettlement.run_settlement_cycle()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Background settlement error: {str(e)}")
                await asyncio.sleep(interval_seconds)


# Convenience instance
transaction_settlement = TransactionSettlement()
