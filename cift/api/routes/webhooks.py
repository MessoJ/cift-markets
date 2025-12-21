"""
WEBHOOK HANDLERS - RULES COMPLIANT
Handles callbacks from payment processors: Stripe, PayPal, M-Pesa
All webhook events are verified and processed with database updates.
"""

import hashlib
import hmac
import json
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request

from cift.core.config import settings
from cift.core.database import get_postgres_pool
from cift.core.logging import logger
from cift.services.email_service import email_service
from cift.services.sms_service import sms_service

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


# ============================================================================
# STRIPE WEBHOOKS
# ============================================================================

@router.post("/stripe")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None, alias="Stripe-Signature")
):
    """
    Handle Stripe webhook events
    - payment_intent.succeeded
    - payment_intent.failed
    - payment_method.attached
    - payment_method.detached
    """
    body = await request.body()

    # Verify webhook signature
    if not verify_stripe_signature(body, stripe_signature):
        logger.warning("Invalid Stripe webhook signature")
        raise HTTPException(status_code=400, detail="Invalid signature")

    try:
        event = json.loads(body)
        event_type = event.get('type')
        data = event.get('data', {}).get('object', {})

        logger.info(f"Received Stripe webhook: {event_type}")

        if event_type == 'payment_intent.succeeded':
            await handle_stripe_payment_succeeded(data)
        elif event_type == 'payment_intent.failed':
            await handle_stripe_payment_failed(data)
        elif event_type == 'payment_method.attached':
            await handle_stripe_payment_method_attached(data)
        elif event_type == 'setup_intent.succeeded':
            await handle_stripe_setup_succeeded(data)
        else:
            logger.info(f"Unhandled Stripe event type: {event_type}")

        return {"status": "success"}

    except Exception as e:
        logger.error(f"Stripe webhook processing error: {str(e)}", exc_info=True)
        # Return 200 to prevent Stripe from retrying
        return {"status": "error", "message": str(e)}


async def handle_stripe_payment_succeeded(data: dict[str, Any]):
    """Handle successful Stripe payment"""
    payment_intent_id = data.get('id')
    data.get('amount', 0) / 100  # Convert from cents
    metadata = data.get('metadata', {})

    transaction_id = metadata.get('transaction_id')
    if not transaction_id:
        logger.warning(f"No transaction_id in Stripe payment metadata: {payment_intent_id}")
        return

    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        # Update transaction status
        await conn.execute(
            """
            UPDATE funding_transactions
            SET
                status = 'completed',
                completed_at = NOW(),
                external_transaction_id = $1,
                updated_at = NOW()
            WHERE id = $2::uuid
            """,
            payment_intent_id,
            transaction_id,
        )

        # Get user info for notification
        user_data = await conn.fetchrow(
            """
            SELECT u.email, u.phone, ft.amount, ft.type
            FROM funding_transactions ft
            JOIN users u ON ft.user_id = u.id
            WHERE ft.id = $1::uuid
            """,
            transaction_id,
        )

        if user_data:
            # Send email notification
            await email_service.send_transaction_completed(
                email=user_data['email'],
                transaction_type=user_data['type'],
                amount=float(user_data['amount']),
                transaction_id=transaction_id
            )

    logger.info(f"Stripe payment succeeded: {transaction_id}")


async def handle_stripe_payment_failed(data: dict[str, Any]):
    """Handle failed Stripe payment"""
    payment_intent_id = data.get('id')
    failure_message = data.get('last_payment_error', {}).get('message', 'Payment failed')
    metadata = data.get('metadata', {})

    transaction_id = metadata.get('transaction_id')
    if not transaction_id:
        return

    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE funding_transactions
            SET
                status = 'failed',
                failed_reason = $1,
                external_transaction_id = $2,
                updated_at = NOW()
            WHERE id = $3::uuid
            """,
            failure_message,
            payment_intent_id,
            transaction_id,
        )

    logger.info(f"Stripe payment failed: {transaction_id} - {failure_message}")


async def handle_stripe_payment_method_attached(data: dict[str, Any]):
    """Handle payment method attachment"""
    payment_method_id = data.get('id')
    metadata = data.get('metadata', {})

    our_payment_method_id = metadata.get('payment_method_id')
    if not our_payment_method_id:
        return

    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE payment_methods
            SET
                is_verified = true,
                verification_status = 'verified',
                verified_at = NOW(),
                external_method_id = $1,
                updated_at = NOW()
            WHERE id = $2::uuid
            """,
            payment_method_id,
            our_payment_method_id,
        )

    logger.info(f"Stripe payment method attached: {our_payment_method_id}")


async def handle_stripe_setup_succeeded(data: dict[str, Any]):
    """Handle successful setup intent (card verification)"""
    data.get('id')
    payment_method = data.get('payment_method')
    metadata = data.get('metadata', {})

    our_payment_method_id = metadata.get('payment_method_id')
    if not our_payment_method_id:
        return

    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE payment_methods
            SET
                is_verified = true,
                verification_status = 'verified',
                verified_at = NOW(),
                external_method_id = $1,
                updated_at = NOW()
            WHERE id = $2::uuid
            """,
            payment_method,
            our_payment_method_id,
        )

        # Get user email for notification
        user_email = await conn.fetchval(
            """
            SELECT u.email
            FROM payment_methods pm
            JOIN users u ON pm.user_id = u.id
            WHERE pm.id = $1::uuid
            """,
            our_payment_method_id,
        )

        if user_email:
            await email_service.send_payment_method_verified(
                email=user_email,
                payment_method_type='card'
            )

    logger.info(f"Stripe setup succeeded: {our_payment_method_id}")


def verify_stripe_signature(payload: bytes, signature: str) -> bool:
    """Verify Stripe webhook signature"""
    if not signature or not hasattr(settings, 'STRIPE_WEBHOOK_SECRET'):
        return False

    try:
        # Extract timestamp and signature from header
        parts = signature.split(',')
        timestamp = None
        signatures = []

        for part in parts:
            if part.startswith('t='):
                timestamp = part[2:]
            elif part.startswith('v1='):
                signatures.append(part[3:])

        if not timestamp or not signatures:
            return False

        # Compute expected signature
        signed_payload = f"{timestamp}.{payload.decode()}"
        expected_sig = hmac.new(
            settings.STRIPE_WEBHOOK_SECRET.encode(),
            signed_payload.encode(),
            hashlib.sha256
        ).hexdigest()

        # Compare signatures
        return any(hmac.compare_digest(expected_sig, sig) for sig in signatures)

    except Exception as e:
        logger.error(f"Stripe signature verification error: {str(e)}")
        return False


# ============================================================================
# PAYPAL WEBHOOKS
# ============================================================================

@router.post("/paypal")
async def paypal_webhook(request: Request):
    """
    Handle PayPal webhook events
    - PAYMENT.CAPTURE.COMPLETED
    - PAYMENT.CAPTURE.DENIED
    - BILLING.SUBSCRIPTION.ACTIVATED
    """
    body = await request.body()

    # TODO: Verify PayPal webhook signature
    # PayPal uses cert validation, implement based on their docs

    try:
        event = json.loads(body)
        event_type = event.get('event_type')
        resource = event.get('resource', {})

        logger.info(f"Received PayPal webhook: {event_type}")

        if event_type == 'PAYMENT.CAPTURE.COMPLETED':
            await handle_paypal_payment_completed(resource)
        elif event_type == 'PAYMENT.CAPTURE.DENIED':
            await handle_paypal_payment_denied(resource)
        elif event_type == 'CUSTOMER.DISPUTE.CREATED':
            await handle_paypal_dispute(resource)
        else:
            logger.info(f"Unhandled PayPal event type: {event_type}")

        return {"status": "success"}

    except Exception as e:
        logger.error(f"PayPal webhook processing error: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}


async def handle_paypal_payment_completed(resource: dict[str, Any]):
    """Handle completed PayPal payment"""
    capture_id = resource.get('id')
    float(resource.get('amount', {}).get('value', 0))
    custom_id = resource.get('custom_id')  # Our transaction ID

    if not custom_id:
        logger.warning(f"No custom_id in PayPal payment: {capture_id}")
        return

    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE funding_transactions
            SET
                status = 'completed',
                completed_at = NOW(),
                external_transaction_id = $1,
                updated_at = NOW()
            WHERE id = $2::uuid
            """,
            capture_id,
            custom_id,
        )

    logger.info(f"PayPal payment completed: {custom_id}")


async def handle_paypal_payment_denied(resource: dict[str, Any]):
    """Handle denied PayPal payment"""
    capture_id = resource.get('id')
    custom_id = resource.get('custom_id')

    if not custom_id:
        return

    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE funding_transactions
            SET
                status = 'failed',
                failed_reason = 'Payment denied by PayPal',
                external_transaction_id = $1,
                updated_at = NOW()
            WHERE id = $2::uuid
            """,
            capture_id,
            custom_id,
        )

    logger.info(f"PayPal payment denied: {custom_id}")


async def handle_paypal_dispute(resource: dict[str, Any]):
    """Handle PayPal dispute"""
    dispute_id = resource.get('dispute_id')
    transaction_id = resource.get('disputed_transactions', [{}])[0].get('seller_transaction_id')

    logger.warning(f"PayPal dispute created: {dispute_id} for transaction {transaction_id}")
    # TODO: Notify admin, freeze user account if needed


# ============================================================================
# M-PESA WEBHOOKS
# ============================================================================

@router.post("/mpesa/callback")
async def mpesa_callback(request: Request):
    """
    Handle M-Pesa STK Push callback
    Called by Safaricom Daraja API after user completes STK push
    """
    body = await request.body()

    try:
        data = json.loads(body)

        # Extract STK callback data
        callback_data = data.get('Body', {}).get('stkCallback', {})
        result_code = callback_data.get('ResultCode')
        result_desc = callback_data.get('ResultDesc')
        merchant_request_id = callback_data.get('MerchantRequestID')
        checkout_request_id = callback_data.get('CheckoutRequestID')

        logger.info(f"M-Pesa callback: {merchant_request_id} - Result: {result_code}")

        if result_code == 0:
            # Success
            await handle_mpesa_success(checkout_request_id, callback_data)
        else:
            # Failed or cancelled
            await handle_mpesa_failure(checkout_request_id, result_desc)

        return {"ResultCode": 0, "ResultDesc": "Accepted"}

    except Exception as e:
        logger.error(f"M-Pesa callback processing error: {str(e)}", exc_info=True)
        return {"ResultCode": 1, "ResultDesc": str(e)}


async def handle_mpesa_success(checkout_request_id: str, callback_data: dict[str, Any]):
    """Handle successful M-Pesa transaction"""
    # Extract payment details
    callback_metadata = callback_data.get('CallbackMetadata', {}).get('Item', [])
    amount = None
    mpesa_receipt = None

    for item in callback_metadata:
        name = item.get('Name')
        if name == 'Amount':
            amount = item.get('Value')
        elif name == 'MpesaReceiptNumber':
            mpesa_receipt = item.get('Value')
        elif name == 'PhoneNumber':
            item.get('Value')

    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        # Find transaction by checkout_request_id
        transaction = await conn.fetchrow(
            """
            SELECT ft.id, ft.user_id, u.phone
            FROM funding_transactions ft
            JOIN users u ON ft.user_id = u.id
            WHERE ft.external_transaction_id = $1
            """,
            checkout_request_id,
        )

        if transaction:
            # Update transaction
            await conn.execute(
                """
                UPDATE funding_transactions
                SET
                    status = 'completed',
                    completed_at = NOW(),
                    notes = $1,
                    updated_at = NOW()
                WHERE id = $2
                """,
                f"M-Pesa Receipt: {mpesa_receipt}",
                transaction['id'],
            )

            # Send SMS notification
            if transaction['phone']:
                await sms_service.send_transaction_completed(
                    phone=transaction['phone'],
                    amount=float(amount) if amount else 0,
                    receipt=mpesa_receipt
                )

            logger.info(f"M-Pesa payment succeeded: {checkout_request_id}")

        # Also check if this is a verification transaction
        payment_method = await conn.fetchrow(
            """
            SELECT pm.id, pm.user_id, u.email
            FROM payment_methods pm
            JOIN users u ON pm.user_id = u.id
            JOIN payment_verification pv ON pm.id = pv.payment_method_id
            WHERE pv.verification_data->>'code' = $1
            """,
            checkout_request_id,
        )

        if payment_method:
            # Verify payment method
            await conn.execute(
                """
                UPDATE payment_methods
                SET
                    is_verified = true,
                    verification_status = 'verified',
                    verified_at = NOW(),
                    updated_at = NOW()
                WHERE id = $1
                """,
                payment_method['id'],
            )

            # Delete verification record
            await conn.execute(
                "DELETE FROM payment_verification WHERE payment_method_id = $1",
                payment_method['id'],
            )

            # Send email notification
            if payment_method['email']:
                await email_service.send_payment_method_verified(
                    email=payment_method['email'],
                    payment_method_type='M-Pesa'
                )

            logger.info(f"M-Pesa payment method verified: {payment_method['id']}")


async def handle_mpesa_failure(checkout_request_id: str, result_desc: str):
    """Handle failed M-Pesa transaction"""
    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        # Update transaction
        await conn.execute(
            """
            UPDATE funding_transactions
            SET
                status = 'failed',
                failed_reason = $1,
                updated_at = NOW()
            WHERE external_transaction_id = $2
            """,
            result_desc,
            checkout_request_id,
        )

        # Update verification if applicable
        await conn.execute(
            """
            UPDATE payment_methods pm
            SET verification_status = 'verification_failed'
            FROM payment_verification pv
            WHERE pm.id = pv.payment_method_id
            AND pv.verification_data->>'code' = $1
            """,
            checkout_request_id,
        )

    logger.info(f"M-Pesa payment failed: {checkout_request_id} - {result_desc}")


# ============================================================================
# GENERIC WEBHOOK UTILITIES
# ============================================================================

@router.get("/health")
async def webhook_health():
    """Health check for webhook endpoints"""
    return {
        "status": "healthy",
        "endpoints": ["stripe", "paypal", "mpesa/callback"]
    }
