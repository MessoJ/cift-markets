"""
FUNDING API ROUTES
Handles deposits, withdrawals, payment methods, and transfer limits.
All data is fetched from database - NO MOCK DATA.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from cift.core.auth import User, get_current_active_user
from cift.core.database import get_postgres_pool
from cift.core.logging import logger
from cift.services.payment_processor import payment_processor
from cift.services.payment_verification import PaymentVerificationService, VerificationError
from cift.services.receipt_generator import ReceiptGenerator
from cift.services.transaction_settlement import transaction_settlement

router = APIRouter(prefix="/funding", tags=["funding"])


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================

async def get_current_user_id(
    current_user: User = Depends(get_current_active_user)
) -> UUID:
    """Get current authenticated user ID."""
    return current_user.id


# ============================================================================
# MODELS
# ============================================================================

class PaymentMethod(BaseModel):
    """Payment method model - RULES COMPLIANT: supports all payment types"""
    id: str
    user_id: str | None = None
    type: str  # 'bank_account', 'debit_card', 'credit_card', 'paypal', 'mpesa', 'crypto_wallet'
    status: str  # 'active', 'pending_verification', 'verified', 'failed', 'removed'
    name: str | None = None
    last_four: str
    # Bank account fields
    bank_name: str | None = None
    account_type: str | None = None  # 'checking', 'savings'
    account_last4: str | None = None
    routing_number: str | None = None
    # Card fields (debit/credit)
    card_brand: str | None = None
    card_last4: str | None = None
    card_exp_month: int | None = None
    card_exp_year: int | None = None
    # PayPal fields
    paypal_email: str | None = None
    # Cash App fields
    cashapp_tag: str | None = None
    # M-Pesa fields
    mpesa_phone: str | None = None
    mpesa_country: str | None = None
    # Crypto wallet fields
    crypto_address: str | None = None
    crypto_network: str | None = None
    # Status fields
    is_verified: bool
    is_default: bool
    is_active: bool = True
    verified_at: datetime | None = None
    created_at: datetime


class FundingTransaction(BaseModel):
    """Funding transaction model"""
    id: str
    type: str  # 'deposit', 'withdrawal'
    method: str
    amount: Decimal
    fee: Decimal
    status: str  # 'pending', 'processing', 'completed', 'failed', 'cancelled'
    created_at: datetime
    completed_at: datetime | None = None
    expected_arrival: datetime | None = None
    payment_method_id: str | None = None
    notes: str | None = None


class TransferLimit(BaseModel):
    """Transfer limits model"""
    daily_deposit_limit: Decimal
    daily_deposit_remaining: Decimal
    daily_withdrawal_limit: Decimal
    daily_withdrawal_remaining: Decimal
    instant_transfer_limit: Decimal
    instant_transfer_remaining: Decimal


class DepositRequest(BaseModel):
    """Deposit request"""
    amount: Decimal = Field(..., gt=0, description="Amount to deposit")
    payment_method_id: str
    transfer_type: str = Field(..., pattern="^(instant|standard)$")


class WithdrawalRequest(BaseModel):
    """Withdrawal request"""
    amount: Decimal = Field(..., gt=0, description="Amount to withdraw")
    payment_method_id: str


class AddPaymentMethodRequest(BaseModel):
    """Add payment method request - RULES COMPLIANT: accepts all payment types"""
    type: str = Field(..., pattern="^(bank_account|debit_card|credit_card|paypal|cashapp|mpesa|crypto_wallet)$")
    # Bank account fields
    bank_name: str | None = None
    account_type: str | None = None
    account_number: str | None = None
    routing_number: str | None = None
    # Card fields (debit/credit)
    card_number: str | None = None
    card_brand: str | None = None
    card_exp_month: int | None = None
    card_exp_year: int | None = None
    card_cvv: str | None = None
    # PayPal fields
    paypal_email: str | None = None
    # Cash App fields
    cashapp_tag: str | None = None
    # M-Pesa fields
    mpesa_phone: str | None = None
    mpesa_country: str | None = "KE"  # Kenya by default
    # Crypto wallet fields
    crypto_address: str | None = None
    crypto_network: str | None = None  # 'bitcoin', 'ethereum', 'usdc', etc.
    # Generic name field
    name: str | None = None


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/transactions")
async def get_funding_transactions(
    limit: int = 50,
    offset: int = 0,
    transaction_type: str | None = None,
    status: str | None = None,
    user_id: UUID = Depends(get_current_user_id),
):
    """Get funding transaction history from database"""
    pool = await get_postgres_pool()

    query = """
        SELECT
            id::text,
            type,
            method,
            amount,
            fee,
            status,
            created_at,
            completed_at,
            expected_arrival,
            payment_method_id::text,
            notes
        FROM funding_transactions
        WHERE user_id = $1
    """
    params = [user_id]
    param_count = 2

    if transaction_type:
        query += f" AND type = ${param_count}"
        params.append(transaction_type)
        param_count += 1

    if status:
        query += f" AND status = ${param_count}"
        params.append(status)
        param_count += 1

    query += f" ORDER BY created_at DESC LIMIT ${param_count} OFFSET ${param_count + 1}"
    params.extend([limit, offset])

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

        transactions = [
            FundingTransaction(
                id=row['id'],
                type=row['type'],
                method=row['method'],
                amount=row['amount'],
                fee=row['fee'],
                status=row['status'],
                created_at=row['created_at'],
                completed_at=row['completed_at'],
                expected_arrival=row['expected_arrival'],
                payment_method_id=row['payment_method_id'],
                notes=row['notes'],
            )
            for row in rows
        ]

    return {"transactions": transactions, "total": len(transactions)}


@router.get("/transactions/{transaction_id}")
async def get_funding_transaction(
    transaction_id: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """Get funding transaction detail from database"""
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                id::text,
                type,
                method,
                amount,
                fee,
                status,
                created_at,
                completed_at,
                expected_arrival,
                payment_method_id::text,
                notes
            FROM funding_transactions
            WHERE id = $1::uuid AND user_id = $2
            """,
            transaction_id,
            user_id,
        )

        if not row:
            raise HTTPException(status_code=404, detail="Transaction not found")

        return FundingTransaction(
            id=row['id'],
            type=row['type'],
            method=row['method'],
            amount=row['amount'],
            fee=row['fee'],
            status=row['status'],
            created_at=row['created_at'],
            completed_at=row['completed_at'],
            expected_arrival=row['expected_arrival'],
            payment_method_id=row['payment_method_id'],
            notes=row['notes'],
        )


@router.get("/payment-methods")
async def get_payment_methods(
    user_id: UUID = Depends(get_current_user_id),
):
    """Get user's payment methods from database"""
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                id::text,
                type,
                name,
                last_four,
                bank_name,
                account_type,
                routing_number,
                card_brand,
                card_exp_month,
                card_exp_year,
                paypal_email,
                cashapp_tag,
                mpesa_phone,
                mpesa_country,
                crypto_address,
                crypto_network,
                is_verified,
                is_default,
                is_active,
                created_at
            FROM payment_methods
            WHERE user_id = $1 AND is_active = true
            ORDER BY is_default DESC, created_at DESC
            """,
            user_id,
        )

        # RULES COMPLIANT: Return data from database with proper field mapping
        def compute_status(row):
            """Compute status from is_verified and is_active"""
            if not row['is_active']:
                return 'removed'
            elif row['is_verified']:
                return 'verified'
            else:
                return 'pending_verification'

        return {
            "payment_methods": [
                PaymentMethod(
                    id=row['id'],
                    user_id=str(user_id),
                    type=row['type'],
                    status=compute_status(row),
                    name=row['name'],
                    last_four=row['last_four'],
                    # Map account fields properly
                    bank_name=row['bank_name'],
                    account_type=row['account_type'],
                    account_last4=row['last_four'] if row['type'] == 'bank_account' else None,
                    routing_number=row['routing_number'],
                    # Map card fields
                    card_brand=row['card_brand'],
                    card_last4=row['last_four'] if row['type'] in ('debit_card', 'credit_card') else None,
                    card_exp_month=row['card_exp_month'],
                    card_exp_year=row['card_exp_year'],
                    paypal_email=row['paypal_email'],
                    cashapp_tag=row['cashapp_tag'],
                    mpesa_phone=row['mpesa_phone'],
                    mpesa_country=row['mpesa_country'],
                    # Crypto fields
                    crypto_address=row['crypto_address'],
                    crypto_network=row['crypto_network'],
                    # Status fields
                    is_verified=row['is_verified'],
                    is_default=row['is_default'],
                    is_active=row['is_active'],
                    created_at=row['created_at'],
                )
                for row in rows
            ]
        }


@router.post("/payment-methods")
async def add_payment_method(
    request: AddPaymentMethodRequest,
    user_id: UUID = Depends(get_current_user_id),
):
    """Add new payment method to database - RULES COMPLIANT"""
    pool = await get_postgres_pool()

    # Extract last 4 digits and display name based on type
    if request.type == 'bank_account':
        last_four = request.account_number[-4:] if request.account_number else "0000"
        display_name = request.bank_name or request.name or f"Bank Account {last_four}"
    elif request.type in ('debit_card', 'credit_card'):
        last_four = request.card_number[-4:] if request.card_number else "0000"
        card_type = "Credit" if request.type == 'credit_card' else "Debit"
        display_name = request.name or f"{request.card_brand or card_type} Card {last_four}"
    elif request.type == 'paypal':
        # Mask email: john****@gmail.com
        if request.paypal_email:
            email_parts = request.paypal_email.split('@')
            masked = email_parts[0][:4] + '****@' + email_parts[1] if len(email_parts) == 2 else request.paypal_email
            last_four = email_parts[0][-4:] if len(email_parts[0]) >= 4 else email_parts[0]
            display_name = request.name or f"PayPal {masked}"
        else:
            last_four = "0000"
            display_name = request.name or "PayPal Account"
    elif request.type == 'cashapp':
        # Cash App $Cashtag
        if request.cashapp_tag:
            tag = request.cashapp_tag if request.cashapp_tag.startswith('$') else f"${request.cashapp_tag}"
            last_four = tag[-4:] if len(tag) >= 4 else tag
            display_name = request.name or f"Cash App {tag}"
        else:
            last_four = "0000"
            display_name = request.name or "Cash App"
    elif request.type == 'mpesa':
        last_four = request.mpesa_phone[-4:] if request.mpesa_phone and len(request.mpesa_phone) >= 4 else "0000"
        display_name = request.name or f"M-Pesa {request.mpesa_country or 'KE'} {last_four}"
    elif request.type == 'crypto_wallet':
        last_four = request.crypto_address[-4:] if request.crypto_address and len(request.crypto_address) >= 4 else "0000"
        network_name = (request.crypto_network or 'Crypto').title()
        display_name = request.name or f"{network_name} Wallet {last_four}"
    else:
        last_four = "0000"
        display_name = request.name or "Payment Method"

    async with pool.acquire() as conn:
        # Check if this is the first payment method
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM payment_methods WHERE user_id = $1 AND is_active = true",
            user_id,
        )
        is_default = count == 0

        row = await conn.fetchrow(
            """
            INSERT INTO payment_methods (
                user_id, type, name, last_four,
                bank_name, account_type, routing_number,
                card_brand, card_exp_month, card_exp_year,
                paypal_email, cashapp_tag, mpesa_phone, mpesa_country,
                crypto_address, crypto_network,
                account_number_encrypted, routing_number_encrypted,
                is_verified, is_default, is_active
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, true)
            RETURNING
                id::text, type, name, last_four,
                bank_name, account_type, routing_number,
                card_brand, card_exp_month, card_exp_year,
                paypal_email, cashapp_tag, mpesa_phone, mpesa_country,
                crypto_address, crypto_network,
                is_verified, is_default, is_active, created_at
            """,
            user_id,
            request.type,
            display_name,
            last_four,
            request.bank_name,
            request.account_type,
            request.routing_number,
            request.card_brand if request.type in ('debit_card', 'credit_card') else None,
            request.card_exp_month,
            request.card_exp_year,
            request.paypal_email,
            request.cashapp_tag,
            request.mpesa_phone,
            request.mpesa_country,
            request.crypto_address,
            request.crypto_network,
            request.account_number or request.card_number,  # TODO: Encrypt in production
            request.routing_number,  # TODO: Encrypt in production
            False,  # Requires verification
            is_default,
        )

        # Link with external processor if needed
        try:
            if request.type == 'bank_account':
                link_result = await payment_processor.link_external_account(
                    user_id=user_id,
                    payment_type='bank_account',
                    account_details={
                        'account_owner_name': request.name,
                        'bank_account_type': request.account_type,
                        'bank_account_number': request.account_number,
                        'bank_routing_number': request.routing_number,
                        'nickname': display_name
                    },
                    metadata={'payment_method_id': row['id']}
                )

                if link_result.get('status') == 'APPROVED':
                    # Auto-verify if approved immediately
                    await conn.execute(
                        "UPDATE payment_methods SET is_verified = true WHERE id = $1::uuid",
                        row['id']
                    )
                    row = dict(row)
                    row['is_verified'] = True

        except Exception as e:
            logger.error(f"Failed to link external account: {str(e)}")
            # We don't fail the request, but the account remains unverified

        # RULES COMPLIANT: Return in format frontend expects
        # Compute status from verification state
        status = 'verified' if row['is_verified'] else 'pending_verification'

        payment_method = PaymentMethod(
            id=row['id'],
            user_id=str(user_id),
            type=row['type'],
            status=status,
            name=row['name'],
            last_four=row['last_four'],
            bank_name=row['bank_name'],
            account_type=row['account_type'],
            account_last4=row['last_four'] if row['type'] == 'bank_account' else None,
            routing_number=row['routing_number'],
            card_brand=row['card_brand'],
            card_last4=row['last_four'] if row['type'] in ('debit_card', 'credit_card') else None,
            card_exp_month=row['card_exp_month'],
            card_exp_year=row['card_exp_year'],
            paypal_email=row['paypal_email'],
            cashapp_tag=row['cashapp_tag'],
            mpesa_phone=row['mpesa_phone'],
            mpesa_country=row['mpesa_country'],
            crypto_address=row['crypto_address'],
            crypto_network=row['crypto_network'],
            is_verified=row['is_verified'],
            is_default=row['is_default'],
            is_active=row['is_active'],
            created_at=row['created_at'],
        )

        return {"payment_method": payment_method}


@router.delete("/payment-methods/{method_id}")
async def delete_payment_method(
    method_id: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """Soft delete payment method"""
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE payment_methods
            SET is_active = false
            WHERE id = $1::uuid AND user_id = $2
            """,
            method_id,
            user_id,
        )

        if result == "UPDATE 0":
            raise HTTPException(status_code=404, detail="Payment method not found")

        return {"success": True}


@router.get("/limits")
async def get_transfer_limits(
    user_id: UUID = Depends(get_current_user_id),
):
    """Get transfer limits from database"""
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        # Get user's tier limits
        limits_row = await conn.fetchrow(
            """
            SELECT
                daily_deposit_limit,
                daily_withdrawal_limit,
                instant_transfer_limit
            FROM user_transfer_limits
            WHERE user_id = $1
            """,
            user_id,
        )

        if not limits_row:
            # Default limits for new users
            daily_deposit_limit = Decimal("25000.00")
            daily_withdrawal_limit = Decimal("25000.00")
            instant_transfer_limit = Decimal("1000.00")
        else:
            daily_deposit_limit = limits_row['daily_deposit_limit']
            daily_withdrawal_limit = limits_row['daily_withdrawal_limit']
            instant_transfer_limit = limits_row['instant_transfer_limit']

        # Calculate used amounts today
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        used_deposit = await conn.fetchval(
            """
            SELECT COALESCE(SUM(amount), 0)
            FROM funding_transactions
            WHERE user_id = $1
            AND type = 'deposit'
            AND status IN ('completed', 'processing')
            AND created_at >= $2
            """,
            user_id,
            today_start,
        ) or Decimal("0")

        used_withdrawal = await conn.fetchval(
            """
            SELECT COALESCE(SUM(amount), 0)
            FROM funding_transactions
            WHERE user_id = $1
            AND type = 'withdrawal'
            AND status IN ('completed', 'processing')
            AND created_at >= $2
            """,
            user_id,
            today_start,
        ) or Decimal("0")

        used_instant = await conn.fetchval(
            """
            SELECT COALESCE(SUM(amount), 0)
            FROM funding_transactions
            WHERE user_id = $1
            AND type = 'deposit'
            AND method = 'instant'
            AND status IN ('completed', 'processing')
            AND created_at >= $2
            """,
            user_id,
            today_start,
        ) or Decimal("0")

        return TransferLimit(
            daily_deposit_limit=daily_deposit_limit,
            daily_deposit_remaining=daily_deposit_limit - used_deposit,
            daily_withdrawal_limit=daily_withdrawal_limit,
            daily_withdrawal_remaining=daily_withdrawal_limit - used_withdrawal,
            instant_transfer_limit=instant_transfer_limit,
            instant_transfer_remaining=instant_transfer_limit - used_instant,
        )


@router.post("/deposit")
async def create_deposit(
    request: DepositRequest,
    user_id: UUID = Depends(get_current_user_id),
):
    """Create deposit transaction in database"""
    pool = await get_postgres_pool()

    # Verify payment method belongs to user
    async with pool.acquire() as conn:
        method = await conn.fetchrow(
            "SELECT type, is_verified FROM payment_methods WHERE id = $1::uuid AND user_id = $2 AND is_active = true",
            request.payment_method_id,
            user_id,
        )

        if not method:
            raise HTTPException(status_code=404, detail="Payment method not found")

        if not method['is_verified']:
            raise HTTPException(status_code=400, detail="Payment method not verified")

        # Calculate fee using payment processor
        fee = payment_processor.calculate_fee(
            request.amount,
            method['type'],
            request.transfer_type
        )

        # Calculate expected arrival
        if request.transfer_type == "instant":
            expected_arrival = datetime.utcnow() + timedelta(minutes=5)
        else:
            expected_arrival = datetime.utcnow() + timedelta(days=3)

        # Insert transaction
        row = await conn.fetchrow(
            """
            INSERT INTO funding_transactions (
                user_id, type, method, amount, fee, status,
                payment_method_id, expected_arrival
            ) VALUES ($1, 'deposit', $2, $3, $4, 'processing', $5::uuid, $6)
            RETURNING id::text, type, method, amount, fee, status, created_at, expected_arrival
            """,
            user_id,
            request.transfer_type,
            request.amount,
            fee,
            request.payment_method_id,
            expected_arrival,
        )

        # Process payment with payment processor
        try:
            if method['type'] in ('debit_card', 'credit_card'):
                # Card payment via Stripe
                payment_result = await payment_processor.create_payment_intent(
                    amount=request.amount + fee,
                    payment_method_id=request.payment_method_id,
                    metadata={
                        'transaction_id': row['id'],
                        'user_id': str(user_id),
                        'type': 'deposit'
                    }
                )
                logger.info(f"Payment intent created: {payment_result.get('id')} - Simulation: {payment_result.get('simulation', False)}")

                # Update status based on payment result
                if payment_result.get('status') == 'succeeded':
                    await conn.execute(
                        "UPDATE funding_transactions SET status = 'completed', completed_at = NOW() WHERE id = $1::uuid",
                        row['id']
                    )
                    # Credit user account immediately
                    await conn.execute(
                        "UPDATE accounts SET cash = cash + $1 WHERE user_id = $2 AND is_active = true",
                        request.amount,
                        user_id,
                    )

                    # Record revenue
                    if fee > 0:
                        await conn.execute(
                            """
                            INSERT INTO platform_revenue (
                                source_type, amount, reference_id, user_id, description
                            ) VALUES ($1, $2, $3::uuid, $4, $5)
                            """,
                            'funding_fee',
                            fee,
                            row['id'],
                            user_id,
                            f"Fee for {method['type']} deposit"
                        )
            elif method['type'] == 'bank_account':
                # ACH transfer - stays in processing
                payment_result = await payment_processor.create_bank_transfer(
                    amount=request.amount + fee,
                    bank_account_id=request.payment_method_id,
                    metadata={
                        'transaction_id': row['id'],
                        'user_id': str(user_id),
                        'type': 'deposit'
                    }
                )
                logger.info(f"Bank transfer initiated: {payment_result.get('id')} - Simulation: {payment_result.get('simulation', False)}")
        except Exception as e:
            logger.error(f"Payment processing error: {str(e)}")
            # Update transaction to failed
            await conn.execute(
                "UPDATE funding_transactions SET status = 'failed' WHERE id = $1::uuid",
                row['id']
            )

        return FundingTransaction(
            id=row['id'],
            type=row['type'],
            method=row['method'],
            amount=row['amount'],
            fee=row['fee'],
            status=row['status'],
            created_at=row['created_at'],
            completed_at=None,
            expected_arrival=row['expected_arrival'],
            payment_method_id=request.payment_method_id,
        )


@router.post("/withdraw")
async def create_withdrawal(
    request: WithdrawalRequest,
    user_id: UUID = Depends(get_current_user_id),
):
    """Create withdrawal transaction in database"""
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        # Verify payment method
        method = await conn.fetchrow(
            "SELECT type, is_verified FROM payment_methods WHERE id = $1::uuid AND user_id = $2 AND is_active = true",
            request.payment_method_id,
            user_id,
        )

        if not method:
            raise HTTPException(status_code=404, detail="Payment method not found")

        # Check available cash from accounts table
        account = await conn.fetchrow(
            "SELECT cash FROM accounts WHERE user_id = $1 AND is_active = true LIMIT 1",
            user_id,
        )

        if not account or account['cash'] < request.amount:
            raise HTTPException(status_code=400, detail="Insufficient funds")

        # Calculate fee for withdrawal
        fee = payment_processor.calculate_fee(
            request.amount,
            method['type'],
            'standard'
        )
        expected_arrival = datetime.utcnow() + timedelta(days=3)

        # Insert transaction
        row = await conn.fetchrow(
            """
            INSERT INTO funding_transactions (
                user_id, type, method, amount, fee, status,
                payment_method_id, expected_arrival
            ) VALUES ($1, 'withdrawal', 'standard', $2, $3, 'processing', $4::uuid, $5)
            RETURNING id::text, type, method, amount, fee, status, created_at, expected_arrival
            """,
            user_id,
            request.amount,
            fee,
            request.payment_method_id,
            expected_arrival,
        )

        # Deduct from cash (amount + fee)
        await conn.execute(
            "UPDATE accounts SET cash = cash - $1 WHERE user_id = $2 AND is_active = true",
            request.amount + fee,
            user_id,
        )

        # Process withdrawal with payment processor
        try:
            withdrawal_result = await payment_processor.process_withdrawal(
                amount=request.amount,
                payment_method_type=method['type'],
                payment_method_id=request.payment_method_id,
                metadata={
                    'transaction_id': row['id'],
                    'user_id': str(user_id),
                    'type': 'withdrawal'
                }
            )
            logger.info(f"Withdrawal initiated: {withdrawal_result.get('id')} - Simulation: {withdrawal_result.get('simulation', False)}")
        except Exception as e:
            logger.error(f"Withdrawal processing error: {str(e)}")
            # Refund the amount on failure
            await conn.execute(
                "UPDATE accounts SET cash = cash + $1 WHERE user_id = $2 AND is_active = true",
                request.amount + fee,
                user_id,
            )
            await conn.execute(
                "UPDATE funding_transactions SET status = 'failed' WHERE id = $1::uuid",
                row['id']
            )
            raise HTTPException(status_code=500, detail="Withdrawal processing failed") from e

        return FundingTransaction(
            id=row['id'],
            type=row['type'],
            method=row['method'],
            amount=row['amount'],
            fee=row['fee'],
            status=row['status'],
            created_at=row['created_at'],
            completed_at=None,
            expected_arrival=row['expected_arrival'],
            payment_method_id=request.payment_method_id,
        )


@router.delete("/transactions/{transaction_id}")
async def cancel_funding_transaction(
    transaction_id: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """Cancel a pending funding transaction - RULES COMPLIANT"""
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        # Verify transaction belongs to user and is cancellable
        txn = await conn.fetchrow(
            "SELECT status FROM funding_transactions WHERE id = $1::uuid AND user_id = $2",
            transaction_id,
            user_id,
        )

        if not txn:
            raise HTTPException(status_code=404, detail="Transaction not found")

        if txn['status'] not in ('pending', 'processing'):
            raise HTTPException(status_code=400, detail="Transaction cannot be cancelled")

        # Update status to cancelled
        await conn.execute(
            "UPDATE funding_transactions SET status = 'cancelled' WHERE id = $1::uuid",
            transaction_id,
        )

        return {"message": "Transaction cancelled successfully"}


@router.get("/transactions/{transaction_id}/receipt")
async def download_receipt(
    transaction_id: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """Download PDF receipt for a transaction - RULES COMPLIANT"""
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        # Fetch transaction data from database
        transaction = await conn.fetchrow(
            """
            SELECT
                id::text,
                user_id,
                type,
                method,
                amount,
                fee,
                status,
                payment_method_id::text,
                created_at,
                completed_at,
                expected_arrival
            FROM funding_transactions
            WHERE id = $1::uuid AND user_id = $2
            """,
            transaction_id,
            user_id,
        )

        if not transaction:
            raise HTTPException(status_code=404, detail="Transaction not found")

        # Fetch user data
        user = await conn.fetchrow(
            "SELECT id, full_name, email FROM users WHERE id = $1",
            user_id,
        )

        # Fetch payment method data
        payment_method = await conn.fetchrow(
            """
            SELECT
                id::text,
                type,
                name,
                last_four,
                bank_name,
                card_brand
            FROM payment_methods
            WHERE id = $1::uuid
            """,
            transaction['payment_method_id'],
        )

        if not payment_method:
            # Create a default payment method if not found
            payment_method = {
                'id': transaction['payment_method_id'],
                'type': 'unknown',
                'name': 'Payment Method',
                'last_four': '****',
                'bank_name': None,
                'card_brand': None,
            }

        # Convert to dicts for receipt generator
        transaction_data = dict(transaction)
        user_data = dict(user) if user else {'full_name': 'Unknown', 'email': 'N/A'}
        payment_method_data = dict(payment_method)

        try:
            logger.info(f"Generating PDF receipt for transaction {transaction_id}")
            # Generate PDF receipt
            pdf_buffer = await ReceiptGenerator.generate_receipt(
                transaction_data,
                user_data,
                payment_method_data
            )

            logger.info(f"PDF receipt generated successfully for transaction {transaction_id}")
            # Return as streaming response
            return StreamingResponse(
                pdf_buffer,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"attachment; filename=receipt_{transaction_id}.pdf",
                    "Content-Type": "application/pdf"
                }
            )
        except Exception as e:
            logger.error(f"Failed to generate PDF receipt for transaction {transaction_id}: {str(e)}", exc_info=True)
            # Fallback to text receipt if PDF generation fails
            try:
                text_receipt = ReceiptGenerator.generate_simple_text_receipt(
                    transaction_data,
                    user_data,
                    payment_method_data
                )

                return StreamingResponse(
                    iter([text_receipt.encode()]),
                    media_type="text/plain",
                    headers={
                        "Content-Disposition": f"attachment; filename=receipt_{transaction_id}.txt"
                    }
                )
            except Exception as fallback_error:
                logger.error(f"Text receipt fallback also failed: {str(fallback_error)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate receipt: {str(e)}"
                ) from fallback_error


@router.post("/payment-methods/{payment_method_id}/verify/initiate")
async def initiate_payment_verification(
    payment_method_id: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Initiate verification for a payment method
    Returns verification status and next steps
    """
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        # Get payment method details
        method = await conn.fetchrow(
            """
            SELECT type, bank_name, account_type, routing_number,
                   card_brand, paypal_email, mpesa_phone, mpesa_country,
                   cashapp_tag, crypto_address, crypto_network
            FROM payment_methods
            WHERE id = $1::uuid AND user_id = $2 AND is_active = true
            """,
            payment_method_id,
            user_id,
        )

        if not method:
            raise HTTPException(status_code=404, detail="Payment method not found")

        # Prepare metadata
        metadata = dict(method)

        try:
            result = await PaymentVerificationService.initiate_verification(
                UUID(payment_method_id),
                method['type'],
                metadata
            )
            return result
        except VerificationError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error(f"Verification initiation error: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to initiate verification") from e


@router.post("/payment-methods/{payment_method_id}/verify/complete")
async def complete_payment_verification(
    payment_method_id: str,
    verification_data: dict[str, Any],
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Complete verification with user-provided data (amounts, codes, etc.)
    """
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        # Verify ownership
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM payment_methods WHERE id = $1::uuid AND user_id = $2)",
            payment_method_id,
            user_id,
        )

        if not exists:
            raise HTTPException(status_code=404, detail="Payment method not found")

        try:
            result = await PaymentVerificationService.complete_verification(
                UUID(payment_method_id),
                verification_data
            )
            return result
        except VerificationError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error(f"Verification completion error: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to complete verification") from e


@router.get("/payment-methods/{payment_method_id}/verification-status")
async def get_verification_status(
    payment_method_id: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """Get current verification status for a payment method"""
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        # Get payment method and verification status
        method = await conn.fetchrow(
            """
            SELECT
                pm.verification_status,
                pm.is_verified,
                pm.verified_at,
                pm.verification_error,
                pv.verification_type,
                pv.attempt_count,
                pv.expires_at
            FROM payment_methods pm
            LEFT JOIN payment_verification pv ON pm.id = pv.payment_method_id
            WHERE pm.id = $1::uuid AND pm.user_id = $2
            """,
            payment_method_id,
            user_id,
        )

        if not method:
            raise HTTPException(status_code=404, detail="Payment method not found")

        return {
            "status": method['verification_status'],
            "is_verified": method['is_verified'],
            "verified_at": method['verified_at'].isoformat() if method['verified_at'] else None,
            "verification_type": method['verification_type'],
            "attempt_count": method['attempt_count'] or 0,
            "expires_at": method['expires_at'].isoformat() if method['expires_at'] else None,
            "error": method['verification_error'],
        }


@router.get("/transactions/{transaction_id}/status")
async def get_transaction_status(
    transaction_id: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Get real-time transaction status
    Used for polling transaction progress
    """
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                status,
                completed_at,
                expected_arrival,
                notes,
                failed_reason
            FROM funding_transactions
            WHERE id = $1::uuid AND user_id = $2
            """,
            transaction_id,
            user_id,
        )

        if not row:
            raise HTTPException(status_code=404, detail="Transaction not found")

        return {
            "status": row['status'],
            "completed_at": row['completed_at'].isoformat() if row['completed_at'] else None,
            "expected_arrival": row['expected_arrival'].isoformat() if row['expected_arrival'] else None,
            "notes": row['notes'],
            "failed_reason": row['failed_reason'] if row['status'] == 'failed' else None,
        }


@router.post("/admin/settlement/run")
async def run_settlement(
    current_user: User = Depends(get_current_active_user),
):
    """
    Manually trigger settlement cycle - ADMIN ONLY
    Clears pending transactions that have reached their expected arrival time
    """
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Admin access required")

    result = await transaction_settlement.run_settlement_cycle()
    return {
        "message": "Settlement cycle completed",
        "result": result
    }


# ============================================================================
# PLAID BANK LINKING (REAL ACH TRANSFERS)
# ============================================================================

class PlaidLinkTokenRequest(BaseModel):
    """Request to create a Plaid Link token."""
    pass  # No parameters needed, user_id comes from auth


class PlaidExchangeTokenRequest(BaseModel):
    """Request to exchange public token for access token."""
    public_token: str = Field(..., description="Public token from Plaid Link")
    account_id: str = Field(..., description="Selected account ID from Plaid")


@router.post("/plaid/link-token")
async def create_plaid_link_token(
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Create a Plaid Link token for bank account connection.
    
    Returns a link_token to initialize Plaid Link in the frontend.
    
    **Flow:**
    1. Frontend calls this endpoint
    2. Backend returns link_token
    3. Frontend opens Plaid Link with token
    4. User logs into their bank
    5. Frontend receives public_token
    6. Frontend calls /plaid/exchange-token
    """
    from cift.services.plaid_service import plaid_service, PlaidServiceError
    
    if not plaid_service.is_available:
        raise HTTPException(
            status_code=503,
            detail="Bank linking is currently unavailable. Please configure PLAID_CLIENT_ID and PLAID_SECRET."
        )
    
    try:
        result = await plaid_service.create_link_token(str(user_id))
        return {
            "link_token": result["link_token"],
            "expiration": result["expiration"],
        }
    except PlaidServiceError as e:
        logger.error(f"Plaid link token error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/plaid/exchange-token")
async def exchange_plaid_token(
    request: PlaidExchangeTokenRequest,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Exchange Plaid public token for access token and save bank account.
    
    Called after user completes Plaid Link flow.
    
    **Flow:**
    1. Exchange public_token for access_token
    2. Get bank account details (routing, account number)
    3. Save payment method to database
    4. Mark as verified (instant verification via bank login)
    """
    from cift.services.plaid_service import plaid_service, PlaidServiceError
    
    if not plaid_service.is_available:
        raise HTTPException(status_code=503, detail="Bank linking unavailable")
    
    try:
        # Exchange token
        token_result = await plaid_service.exchange_public_token(request.public_token)
        access_token = token_result["access_token"]
        item_id = token_result["item_id"]
        
        # Get auth data (account/routing numbers)
        auth_data = await plaid_service.get_auth_data(access_token)
        
        # Find the selected account
        account = None
        for acc in auth_data["accounts"]:
            if acc["account_id"] == request.account_id:
                account = acc
                break
        
        if not account:
            raise HTTPException(status_code=400, detail="Selected account not found")
        
        # Save to payment_methods table
        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            # Store the payment method
            method_id = await conn.fetchval("""
                INSERT INTO payment_methods (
                    user_id, type, status, last_four,
                    bank_name, account_type, routing_number,
                    is_verified, verified_at, is_active, is_default,
                    external_id, metadata
                ) VALUES (
                    $1, 'bank_account', 'verified', $2,
                    $3, $4, $5,
                    true, NOW(), true, false,
                    $6, $7
                ) RETURNING id::text
            """,
                user_id,
                account["mask"],  # Last 4 digits
                auth_data.get("institution_id", "Unknown Bank"),
                account.get("subtype", "checking"),
                account["routing_number"],
                item_id,
                {
                    "plaid_access_token": access_token,
                    "plaid_account_id": request.account_id,
                    "account_name": account["name"],
                }
            )
            
            return {
                "payment_method_id": method_id,
                "bank_name": account.get("official_name", account["name"]),
                "account_type": account.get("subtype", "checking"),
                "last_four": account["mask"],
                "verified": True,
            }
            
    except PlaidServiceError as e:
        logger.error(f"Plaid exchange error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/plaid/deposit")
async def initiate_plaid_deposit(
    payment_method_id: str,
    amount: Decimal,
    user_id: UUID = Depends(get_current_user_id),
    current_user: User = Depends(get_current_active_user),
):
    """
    Initiate a real ACH deposit via Plaid Transfer API.
    
    Pulls money from user's linked bank account.
    """
    from cift.services.plaid_service import plaid_service, PlaidServiceError
    
    if not plaid_service.is_available:
        raise HTTPException(status_code=503, detail="ACH transfers unavailable")
    
    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        # Get payment method with Plaid credentials
        method = await conn.fetchrow("""
            SELECT metadata FROM payment_methods
            WHERE id = $1::uuid AND user_id = $2 AND is_active = true AND type = 'bank_account'
        """, payment_method_id, user_id)
        
        if not method:
            raise HTTPException(status_code=404, detail="Payment method not found")
        
        metadata = method["metadata"] or {}
        access_token = metadata.get("plaid_access_token")
        account_id = metadata.get("plaid_account_id")
        
        if not access_token or not account_id:
            raise HTTPException(status_code=400, detail="Payment method not properly linked")
        
        # Get user's legal name for ACH
        user_name = f"{current_user.first_name or ''} {current_user.last_name or ''}".strip()
        if not user_name:
            user_name = current_user.email.split("@")[0]
        
        try:
            result = await plaid_service.initiate_ach_deposit(
                access_token=access_token,
                account_id=account_id,
                amount=amount,
                user_name=user_name,
                description="CIFT Markets Deposit",
            )
            
            if not result.get("success"):
                raise HTTPException(status_code=400, detail=result.get("reason", "Transfer failed"))
            
            # Create funding transaction record
            tx_id = await conn.fetchval("""
                INSERT INTO funding_transactions (
                    user_id, type, method, amount, fee, status,
                    payment_method_id, expected_arrival, external_id, notes
                ) VALUES (
                    $1, 'deposit', 'ach', $2, 0, 'processing',
                    $3::uuid, $4, $5, $6
                ) RETURNING id::text
            """,
                user_id,
                amount,
                payment_method_id,
                datetime.utcnow() + timedelta(days=3),  # ACH takes 1-3 business days
                result.get("transfer_id"),
                f"Plaid Transfer: {result.get('status')}"
            )
            
            return {
                "transaction_id": tx_id,
                "status": "processing",
                "transfer_id": result.get("transfer_id"),
                "expected_arrival": (datetime.utcnow() + timedelta(days=3)).isoformat(),
                "message": "ACH deposit initiated. Funds will arrive in 1-3 business days."
            }
            
        except PlaidServiceError as e:
            logger.error(f"Plaid deposit error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STRIPE CARD PAYMENTS (Credit/Debit Cards with Autofill Security)
# ============================================================================

class StripeSetupIntentRequest(BaseModel):
    """Request to create a Stripe SetupIntent for secure card saving."""
    pass


class StripeCardSaveRequest(BaseModel):
    """Request to save a card after SetupIntent confirmation."""
    setup_intent_id: str = Field(..., description="Stripe SetupIntent ID")
    payment_method_id: str = Field(..., description="Stripe PaymentMethod ID from frontend")


@router.post("/stripe/setup-intent")
async def create_stripe_setup_intent(
    user_id: UUID = Depends(get_current_user_id),
    current_user: User = Depends(get_current_active_user),
):
    """
    Create a Stripe SetupIntent for secure card collection.
    
    **Security Features:**
    - Card data NEVER touches our servers (Stripe Elements handles collection)
    - PCI-DSS compliant (Stripe handles compliance)
    - Supports autofill via Stripe Elements
    - 3D Secure authentication when required
    
    **Frontend Flow:**
    1. Call this endpoint to get client_secret
    2. Initialize Stripe Elements with publishable_key
    3. User enters card (or uses browser autofill)
    4. Call stripe.confirmSetupIntent() on frontend
    5. Call /stripe/save-card with the result
    """
    from cift.services.payment_config import PaymentConfig
    
    stripe_config = PaymentConfig.get_config('stripe')
    if not stripe_config:
        raise HTTPException(
            status_code=503,
            detail="Card payments not configured. Set STRIPE_SECRET_KEY in environment."
        )
    
    import httpx
    
    url = "https://api.stripe.com/v1/setup_intents"
    headers = {
        "Authorization": f"Bearer {stripe_config.get('secret_key')}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    
    data = {
        "usage": "off_session",  # Allow charging later without customer present
        "metadata[user_id]": str(user_id),
        "metadata[platform]": "CIFT Markets",
    }
    
    # Get or create Stripe customer
    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        stripe_customer_id = await conn.fetchval(
            "SELECT stripe_customer_id FROM users WHERE id = $1", user_id
        )
        
        if not stripe_customer_id:
            # Create Stripe customer
            customer_response = await httpx.AsyncClient().post(
                "https://api.stripe.com/v1/customers",
                headers=headers,
                data={
                    "email": current_user.email,
                    "metadata[user_id]": str(user_id),
                }
            )
            if customer_response.status_code == 200:
                stripe_customer_id = customer_response.json()["id"]
                await conn.execute(
                    "UPDATE users SET stripe_customer_id = $1 WHERE id = $2",
                    stripe_customer_id, user_id
                )
        
        if stripe_customer_id:
            data["customer"] = stripe_customer_id
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, data=data, timeout=30.0)
        
        if response.status_code != 200:
            logger.error(f"Stripe SetupIntent error: {response.text}")
            raise HTTPException(status_code=500, detail="Failed to initialize card setup")
        
        result = response.json()
        
        return {
            "client_secret": result["client_secret"],
            "setup_intent_id": result["id"],
            "publishable_key": stripe_config.get("publishable_key"),
        }


@router.post("/stripe/save-card")
async def save_stripe_card(
    request: StripeCardSaveRequest,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Save a card after successful SetupIntent confirmation.
    
    Called by frontend after stripe.confirmSetupIntent() succeeds.
    """
    from cift.services.payment_config import PaymentConfig
    
    stripe_config = PaymentConfig.get_config('stripe')
    if not stripe_config:
        raise HTTPException(status_code=503, detail="Card payments not configured")
    
    import httpx
    
    headers = {
        "Authorization": f"Bearer {stripe_config.get('secret_key')}",
    }
    
    # Retrieve the SetupIntent to get card details
    async with httpx.AsyncClient() as client:
        si_response = await client.get(
            f"https://api.stripe.com/v1/setup_intents/{request.setup_intent_id}",
            headers=headers,
            timeout=30.0
        )
        
        if si_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Invalid SetupIntent")
        
        setup_intent = si_response.json()
        
        if setup_intent["status"] != "succeeded":
            raise HTTPException(status_code=400, detail="Card setup not completed")
        
        # Get payment method details
        pm_response = await client.get(
            f"https://api.stripe.com/v1/payment_methods/{request.payment_method_id}",
            headers=headers,
            timeout=30.0
        )
        
        if pm_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Invalid payment method")
        
        pm = pm_response.json()
        card = pm.get("card", {})
    
    # Save to database (NEVER store full card number - only Stripe token)
    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        method_id = await conn.fetchval("""
            INSERT INTO payment_methods (
                user_id, type, status, last_four,
                card_brand, card_exp_month, card_exp_year,
                is_verified, verified_at, is_active, is_default,
                account_number_encrypted, metadata
            ) VALUES (
                $1, 'credit_card', 'verified', $2,
                $3, $4, $5,
                true, NOW(), true, false,
                $6, $7
            ) RETURNING id::text
        """,
            user_id,
            card.get("last4"),
            card.get("brand"),
            card.get("exp_month"),
            card.get("exp_year"),
            request.payment_method_id,  # Store Stripe PM ID, NOT the card number
            {
                "stripe_setup_intent_id": request.setup_intent_id,
                "funding": card.get("funding"),  # credit, debit, prepaid
                "country": card.get("country"),
            }
        )
        
        return {
            "payment_method_id": method_id,
            "card_brand": card.get("brand"),
            "last_four": card.get("last4"),
            "exp_month": card.get("exp_month"),
            "exp_year": card.get("exp_year"),
            "verified": True,
        }


# ============================================================================
# M-PESA INTEGRATION (Kenya, Tanzania, Uganda, Rwanda)
# ============================================================================

class MpesaPhoneRequest(BaseModel):
    """Request to link an M-Pesa number."""
    phone_number: str = Field(..., description="M-Pesa phone number with country code (e.g., 254712345678)")
    country: str = Field(default="KE", description="Country code: KE, TZ, UG, RW")


class MpesaDepositRequest(BaseModel):
    """Request to initiate M-Pesa STK Push deposit."""
    payment_method_id: str
    amount: Decimal = Field(..., gt=0)


@router.post("/mpesa/link")
async def link_mpesa_number(
    request: MpesaPhoneRequest,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Link an M-Pesa number to user's account.
    
    The number will be verified via STK Push test transaction.
    """
    # Validate phone number format
    phone = request.phone_number.strip().replace("+", "").replace(" ", "")
    
    country_prefixes = {
        "KE": "254",  # Kenya
        "TZ": "255",  # Tanzania
        "UG": "256",  # Uganda
        "RW": "250",  # Rwanda
    }
    
    expected_prefix = country_prefixes.get(request.country.upper())
    if not expected_prefix:
        raise HTTPException(status_code=400, detail="Unsupported country for M-Pesa")
    
    if not phone.startswith(expected_prefix):
        raise HTTPException(
            status_code=400, 
            detail=f"Phone number must start with {expected_prefix} for {request.country}"
        )
    
    if len(phone) < 10 or len(phone) > 15:
        raise HTTPException(status_code=400, detail="Invalid phone number length")
    
    # Save to database
    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        method_id = await conn.fetchval("""
            INSERT INTO payment_methods (
                user_id, type, status, last_four,
                mpesa_phone, mpesa_country,
                is_verified, is_active, is_default
            ) VALUES (
                $1, 'mpesa', 'pending_verification', $2,
                $3, $4,
                false, true, false
            ) RETURNING id::text
        """,
            user_id,
            phone[-4:],  # Last 4 digits
            phone,
            request.country.upper(),
        )
        
        return {
            "payment_method_id": method_id,
            "phone": f"*****{phone[-4:]}",
            "country": request.country.upper(),
            "status": "pending_verification",
            "message": "M-Pesa number linked. Complete verification by making a test deposit."
        }


@router.post("/mpesa/deposit")
async def initiate_mpesa_deposit(
    request: MpesaDepositRequest,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Initiate M-Pesa STK Push deposit.
    
    User will receive a payment prompt on their phone.
    """
    from cift.services.payment_config import PaymentConfig
    from cift.services.payment_processors.mpesa import MpesaProcessor
    
    mpesa_config = PaymentConfig.get_config('mpesa')
    if not mpesa_config:
        raise HTTPException(
            status_code=503,
            detail="M-Pesa not configured. Set MPESA_CONSUMER_KEY and MPESA_CONSUMER_SECRET."
        )
    
    processor = MpesaProcessor(mpesa_config)
    
    try:
        result = await processor.process_deposit(
            user_id=user_id,
            amount=request.amount,
            payment_method_id=UUID(request.payment_method_id),
        )
        
        # Record transaction
        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            tx_id = await conn.fetchval("""
                INSERT INTO funding_transactions (
                    user_id, type, method, amount, fee, status,
                    payment_method_id, expected_arrival, external_id, notes
                ) VALUES (
                    $1, 'deposit', 'mpesa', $2, $3, $4,
                    $5::uuid, $6, $7, $8
                ) RETURNING id::text
            """,
                user_id,
                request.amount,
                result.get("fee", 0),
                result.get("status", "pending"),
                request.payment_method_id,
                result.get("estimated_arrival"),
                result.get("transaction_id"),
                "M-Pesa STK Push initiated"
            )
            
            # Mark payment method as verified if first successful transaction
            await conn.execute("""
                UPDATE payment_methods
                SET is_verified = true, verification_status = 'verified', verified_at = NOW()
                WHERE id = $1::uuid AND is_verified = false
            """, request.payment_method_id)
        
        return {
            "transaction_id": tx_id,
            "status": result.get("status"),
            "message": "Check your phone for the M-Pesa payment prompt",
            "checkout_request_id": result.get("transaction_id"),
        }
        
    except Exception as e:
        logger.error(f"M-Pesa deposit error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PAYPAL INTEGRATION
# ============================================================================

class PayPalLinkRequest(BaseModel):
    """Request to link PayPal account."""
    email: str = Field(..., description="PayPal email address")


class PayPalDepositRequest(BaseModel):
    """Request to initiate PayPal deposit."""
    payment_method_id: str
    amount: Decimal = Field(..., gt=0)
    return_url: str = Field(..., description="URL to redirect after PayPal approval")
    cancel_url: str = Field(..., description="URL to redirect if cancelled")


@router.post("/paypal/link")
async def link_paypal_account(
    request: PayPalLinkRequest,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Link a PayPal account to user's profile.
    
    Verification happens when user completes first transaction.
    """
    # Basic email validation
    if "@" not in request.email or "." not in request.email:
        raise HTTPException(status_code=400, detail="Invalid email format")
    
    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        method_id = await conn.fetchval("""
            INSERT INTO payment_methods (
                user_id, type, status, last_four,
                paypal_email,
                is_verified, is_active, is_default
            ) VALUES (
                $1, 'paypal', 'pending_verification', $2,
                $3,
                false, true, false
            ) RETURNING id::text
        """,
            user_id,
            request.email[-4:],
            request.email,
        )
        
        return {
            "payment_method_id": method_id,
            "email": f"***{request.email[3:]}",
            "status": "pending_verification",
            "message": "PayPal linked. Complete a deposit to verify."
        }


@router.post("/paypal/deposit")
async def initiate_paypal_deposit(
    request: PayPalDepositRequest,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Initiate PayPal deposit.
    
    Returns a URL for user to approve payment on PayPal.
    """
    from cift.services.payment_config import PaymentConfig
    from cift.services.payment_processors.paypal import PayPalProcessor
    
    paypal_config = PaymentConfig.get_config('paypal')
    if not paypal_config:
        raise HTTPException(
            status_code=503,
            detail="PayPal not configured. Set PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET."
        )
    
    processor = PayPalProcessor(paypal_config)
    
    try:
        result = await processor.process_deposit(
            user_id=user_id,
            amount=request.amount,
            payment_method_id=UUID(request.payment_method_id),
            metadata={
                "return_url": request.return_url,
                "cancel_url": request.cancel_url,
            }
        )
        
        # Record transaction
        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            tx_id = await conn.fetchval("""
                INSERT INTO funding_transactions (
                    user_id, type, method, amount, fee, status,
                    payment_method_id, expected_arrival, external_id, notes
                ) VALUES (
                    $1, 'deposit', 'paypal', $2, $3, $4,
                    $5::uuid, $6, $7, $8
                ) RETURNING id::text
            """,
                user_id,
                request.amount,
                result.get("fee", 0),
                "pending",
                request.payment_method_id,
                result.get("estimated_arrival"),
                result.get("transaction_id"),
                "PayPal order created"
            )
        
        return {
            "transaction_id": tx_id,
            "order_id": result.get("transaction_id"),
            "approval_url": result.get("redirect_url"),
            "status": "pending_approval",
            "message": "Redirect user to approval_url to complete payment"
        }
        
    except Exception as e:
        logger.error(f"PayPal deposit error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/paypal/capture/{order_id}")
async def capture_paypal_payment(
    order_id: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Capture PayPal payment after user approval.
    
    Called after user returns from PayPal approval flow.
    """
    from cift.services.payment_config import PaymentConfig
    from cift.services.payment_processors.paypal import PayPalProcessor
    
    paypal_config = PaymentConfig.get_config('paypal')
    if not paypal_config:
        raise HTTPException(status_code=503, detail="PayPal not configured")
    
    processor = PayPalProcessor(paypal_config)
    
    try:
        result = await processor.capture_payment(order_id)
        
        if result.get("status") == "completed":
            # Update transaction
            pool = await get_postgres_pool()
            async with pool.acquire() as conn:
                # Update funding transaction
                await conn.execute("""
                    UPDATE funding_transactions
                    SET status = 'completed', completed_at = NOW()
                    WHERE external_id = $1 AND user_id = $2
                """, order_id, user_id)
                
                # Get amount and credit user
                row = await conn.fetchrow("""
                    SELECT amount FROM funding_transactions
                    WHERE external_id = $1 AND user_id = $2
                """, order_id, user_id)
                
                if row:
                    await conn.execute("""
                        UPDATE accounts SET cash = cash + $1 WHERE user_id = $2
                    """, row["amount"], user_id)
                    
                    # Mark PayPal as verified
                    await conn.execute("""
                        UPDATE payment_methods pm
                        SET is_verified = true, verification_status = 'verified', verified_at = NOW()
                        FROM funding_transactions ft
                        WHERE ft.external_id = $1 AND ft.payment_method_id = pm.id
                    """, order_id)
        
        return {
            "status": result.get("status"),
            "message": "Payment captured successfully" if result.get("status") == "completed" else "Payment pending"
        }
        
    except Exception as e:
        logger.error(f"PayPal capture error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CRYPTOCURRENCY INTEGRATION (Bitcoin, Ethereum)
# ============================================================================

class CryptoWalletRequest(BaseModel):
    """Request to link a crypto wallet."""
    address: str = Field(..., description="Wallet address")
    network: str = Field(..., description="Network: bitcoin, ethereum")


@router.post("/crypto/link")
async def link_crypto_wallet(
    request: CryptoWalletRequest,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Link a cryptocurrency wallet for deposits/withdrawals.
    
    Supported networks: Bitcoin (BTC), Ethereum (ETH)
    """
    network = request.network.lower()
    address = request.address.strip()
    
    # Validate address format
    if network == "bitcoin":
        # Bitcoin: starts with 1, 3, or bc1
        if not (address.startswith("1") or address.startswith("3") or address.startswith("bc1")):
            raise HTTPException(status_code=400, detail="Invalid Bitcoin address format")
        if len(address) < 26 or len(address) > 62:
            raise HTTPException(status_code=400, detail="Invalid Bitcoin address length")
    elif network == "ethereum":
        # Ethereum: starts with 0x and is 42 characters
        if not address.startswith("0x") or len(address) != 42:
            raise HTTPException(status_code=400, detail="Invalid Ethereum address format")
        try:
            int(address[2:], 16)  # Validate hex
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Ethereum address")
    else:
        raise HTTPException(status_code=400, detail="Unsupported network. Use 'bitcoin' or 'ethereum'")
    
    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        method_id = await conn.fetchval("""
            INSERT INTO payment_methods (
                user_id, type, status, last_four,
                crypto_address, crypto_network,
                is_verified, is_active, is_default
            ) VALUES (
                $1, 'crypto_wallet', 'verified', $2,
                $3, $4,
                true, true, false
            ) RETURNING id::text
        """,
            user_id,
            address[-4:],
            address,
            network,
        )
        
        return {
            "payment_method_id": method_id,
            "address": f"{address[:6]}...{address[-4:]}",
            "network": network,
            "verified": True,
        }


@router.get("/crypto/deposit-address")
async def get_crypto_deposit_address(
    network: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Get platform's deposit address for receiving crypto.
    
    User sends crypto to this address, then it's credited to their account.
    """
    from cift.services.payment_config import PaymentConfig
    
    crypto_config = PaymentConfig.get_config('crypto')
    if not crypto_config:
        raise HTTPException(
            status_code=503,
            detail="Crypto deposits not configured"
        )
    
    deposit_addresses = crypto_config.get("deposit_addresses", {})
    address = deposit_addresses.get(network.lower())
    
    if not address:
        raise HTTPException(status_code=400, detail=f"Deposits not available for {network}")
    
    return {
        "network": network.lower(),
        "deposit_address": address,
        "confirmations_required": crypto_config.get("confirmations_required", {}).get(network.lower(), 3),
        "message": f"Send {network.upper()} to this address. Credits after confirmations."
    }


@router.post("/crypto/withdraw")
async def initiate_crypto_withdrawal(
    payment_method_id: str,
    amount_usd: Decimal,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Initiate cryptocurrency withdrawal.
    
    Converts USD amount to crypto and sends to user's wallet.
    """
    from cift.services.payment_config import PaymentConfig
    from cift.services.payment_processors.crypto import CryptoProcessor
    
    crypto_config = PaymentConfig.get_config('crypto')
    if not crypto_config:
        raise HTTPException(status_code=503, detail="Crypto withdrawals not configured")
    
    processor = CryptoProcessor(crypto_config)
    
    # Check user balance
    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        balance = await conn.fetchval(
            "SELECT cash FROM accounts WHERE user_id = $1", user_id
        )
        
        if not balance or balance < amount_usd:
            raise HTTPException(status_code=400, detail="Insufficient funds")
    
    try:
        result = await processor.process_withdrawal(
            user_id=user_id,
            amount=amount_usd,
            payment_method_id=UUID(payment_method_id),
        )
        
        # Record transaction and deduct balance
        async with pool.acquire() as conn:
            tx_id = await conn.fetchval("""
                INSERT INTO funding_transactions (
                    user_id, type, method, amount, fee, status,
                    payment_method_id, expected_arrival, external_id, notes
                ) VALUES (
                    $1, 'withdrawal', 'crypto', $2, $3, $4,
                    $5::uuid, $6, $7, $8
                ) RETURNING id::text
            """,
                user_id,
                amount_usd,
                result.get("fee", 0),
                result.get("status", "processing"),
                payment_method_id,
                result.get("estimated_arrival"),
                result.get("transaction_id"),
                f"Crypto withdrawal: {result.get('crypto_amount')} {result.get('network', 'crypto')}"
            )
            
            # Deduct from balance (hold until confirmed)
            await conn.execute(
                "UPDATE accounts SET cash = cash - $1 WHERE user_id = $2",
                amount_usd, user_id
            )
        
        return {
            "transaction_id": tx_id,
            "status": result.get("status"),
            "crypto_amount": result.get("crypto_amount"),
            "network": result.get("network"),
            "tx_hash": result.get("transaction_id"),
            "message": "Withdrawal initiated. Check blockchain for confirmation."
        }
        
    except Exception as e:
        logger.error(f"Crypto withdrawal error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
