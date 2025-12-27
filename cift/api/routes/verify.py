"""
Public Transaction Verification API

Allows anyone to verify a transaction receipt by transaction ID.
No authentication required - this is a public endpoint for receipt verification.
"""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException

from cift.core.database import get_postgres_pool
from cift.core.logging import logger

router = APIRouter(tags=["verification"])


@router.get("/verify/{transaction_id}")
async def verify_transaction(transaction_id: str) -> dict[str, Any]:
    """
    Public endpoint to verify a transaction receipt.

    Returns limited transaction information for verification purposes.
    Does not require authentication - anyone with a receipt can verify it.

    Args:
        transaction_id: UUID of the transaction to verify

    Returns:
        Verification result with transaction details
    """
    try:
        # Validate UUID format
        try:
            txn_uuid = UUID(transaction_id)
        except ValueError:
            return {
                "valid": False,
                "message": "Invalid transaction ID format"
            }

        pool = await get_postgres_pool()

        async with pool.acquire() as conn:
            # Fetch transaction with limited fields (no sensitive user data)
            transaction = await conn.fetchrow(
                """
                SELECT
                    id,
                    type,
                    status,
                    amount,
                    fee,
                    created_at,
                    payment_method_id
                FROM funding_transactions
                WHERE id = $1
                """,
                txn_uuid
            )

            if not transaction:
                logger.info(f"Verification attempted for non-existent transaction: {transaction_id}")
                return {
                    "valid": False,
                    "message": "Transaction not found in our records"
                }

            # Fetch payment method type (no sensitive details like full account number)
            payment_method = None
            if transaction['payment_method_id']:
                payment_method = await conn.fetchrow(
                    """
                    SELECT type, last_four
                    FROM payment_methods
                    WHERE id = $1
                    """,
                    transaction['payment_method_id']
                )

            # Build verification response with limited information
            response = {
                "valid": True,
                "transaction": {
                    "id": str(transaction['id']),
                    "type": transaction['type'],
                    "status": transaction['status'],
                    "amount": float(transaction['amount']),
                    "fee": float(transaction['fee']),
                    "created_at": transaction['created_at'].isoformat(),
                }
            }

            # Add payment method info if available (no sensitive data)
            if payment_method:
                response["transaction"]["payment_method_type"] = payment_method['type']
                response["transaction"]["payment_method_last4"] = payment_method['last_four']

            logger.info(f"Transaction verified successfully: {transaction_id}")
            return response

    except Exception as e:
        logger.error(f"Error verifying transaction {transaction_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while verifying the transaction"
        ) from e
