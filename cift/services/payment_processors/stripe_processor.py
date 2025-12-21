"""
Stripe Payment Processor - RULES COMPLIANT
Integrates with Stripe for card payments (credit/debit cards) with autofill support
"""
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

import httpx

from cift.core.database import get_postgres_pool
from cift.services.payment_processors.base import PaymentProcessor, PaymentProcessorError


class StripeProcessor(PaymentProcessor):
    """
    Stripe payment integration for card payments

    Configuration required:
        - secret_key: Stripe Secret Key
        - publishable_key: Stripe Publishable Key (for frontend)
        - webhook_secret: Stripe Webhook Secret
        - environment: 'sandbox' or 'production'
    """

    API_BASE_URL = "https://api.stripe.com/v1"
    API_VERSION = "2023-10-16"

    def _validate_config(self) -> None:
        """Validate Stripe configuration"""
        required = ['secret_key', 'publishable_key']

        for key in required:
            if key not in self.config:
                raise PaymentProcessorError(f"Missing Stripe configuration: {key}")

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Make authenticated request to Stripe API

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without base URL)
            data: Request payload

        Returns:
            API response as dict

        Raises:
            PaymentProcessorError: If request fails
        """
        url = f"{self.API_BASE_URL}/{endpoint}"

        headers = {
            "Authorization": f"Bearer {self.config['secret_key']}",
            "Stripe-Version": self.API_VERSION,
            "Content-Type": "application/x-www-form-urlencoded"
        }

        try:
            async with httpx.AsyncClient() as client:
                if method.upper() == "GET":
                    response = await client.get(url, headers=headers, params=data, timeout=30.0)
                elif method.upper() == "POST":
                    response = await client.post(url, headers=headers, data=data, timeout=30.0)
                elif method.upper() == "DELETE":
                    response = await client.delete(url, headers=headers, timeout=30.0)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                response.raise_for_status()
                return response.json()

        except httpx.HTTPError as e:
            error_msg = f"Stripe API request failed: {str(e)}"
            try:
                error_data = e.response.json()
                if 'error' in error_data:
                    error_msg = error_data['error'].get('message', error_msg)
            except:
                pass

            raise PaymentProcessorError(error_msg)

    async def _fetch_payment_method(self, payment_method_id: UUID) -> dict[str, Any]:
        """Fetch payment method details from database"""
        pool = await get_postgres_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    id::text,
                    type,
                    card_brand,
                    card_exp_month,
                    card_exp_year,
                    last_four,
                    is_verified,
                    account_number_encrypted
                FROM payment_methods
                WHERE id = $1
                """,
                payment_method_id
            )

            if not row:
                raise PaymentProcessorError("Payment method not found")

            if row['type'] not in ['debit_card', 'credit_card']:
                raise PaymentProcessorError("Payment method is not a card")

            return dict(row)

    async def _create_payment_intent(
        self,
        amount: Decimal,
        currency: str,
        payment_method: str,
        metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Create a Stripe PaymentIntent

        Args:
            amount: Amount in dollars
            currency: Currency code (e.g., 'usd')
            payment_method: Stripe payment method ID
            metadata: Transaction metadata

        Returns:
            PaymentIntent object
        """
        # Convert amount to cents (Stripe uses smallest currency unit)
        amount_cents = int(amount * 100)

        data = {
            "amount": amount_cents,
            "currency": currency.lower(),
            "payment_method": payment_method,
            "confirm": "true",  # Immediately attempt to confirm
            "metadata[user_id]": metadata.get('user_id', ''),
            "metadata[transaction_type]": metadata.get('transaction_type', 'deposit'),
            "metadata[platform]": "CIFT Markets",
            "description": f"CIFT Markets {metadata.get('transaction_type', 'deposit')}",
        }

        return await self._make_request("POST", "payment_intents", data)

    async def _create_customer(self, user_id: UUID, email: str) -> str:
        """
        Create a Stripe Customer

        Args:
            user_id: User UUID
            email: User email

        Returns:
            Stripe Customer ID
        """
        data = {
            "email": email,
            "metadata[user_id]": str(user_id),
            "metadata[platform]": "CIFT Markets"
        }

        customer = await self._make_request("POST", "customers", data)
        return customer['id']

    async def _attach_payment_method_to_customer(
        self,
        payment_method_id: str,
        customer_id: str
    ) -> dict[str, Any]:
        """
        Attach a payment method to a Stripe customer

        Args:
            payment_method_id: Stripe PaymentMethod ID
            customer_id: Stripe Customer ID

        Returns:
            Updated PaymentMethod object
        """
        data = {
            "customer": customer_id
        }

        return await self._make_request(
            "POST",
            f"payment_methods/{payment_method_id}/attach",
            data
        )

    async def process_deposit(
        self,
        user_id: UUID,
        amount: Decimal,
        payment_method_id: UUID,
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Process card deposit via Stripe

        This creates a PaymentIntent and charges the card
        """
        # Fetch payment method from database
        payment_method = await self._fetch_payment_method(payment_method_id)

        # Get stored Stripe payment method ID
        stripe_pm_id = payment_method.get('account_number_encrypted')  # Stores Stripe PM ID

        if not stripe_pm_id:
            raise PaymentProcessorError(
                "Stripe payment method not found. Please re-add your card."
            )

        # Calculate fee
        fee = await self.calculate_fee(amount, 'deposit', payment_method['type'])

        # Create payment intent
        metadata_dict = {
            'user_id': str(user_id),
            'payment_method_id': str(payment_method_id),
            'transaction_type': 'deposit'
        }

        if metadata:
            metadata_dict.update(metadata)

        try:
            payment_intent = await self._create_payment_intent(
                amount=amount,
                currency='usd',
                payment_method=stripe_pm_id,
                metadata=metadata_dict
            )

            # Check payment intent status
            status_map = {
                'succeeded': 'completed',
                'processing': 'processing',
                'requires_action': 'pending',  # Needs 3DS or additional auth
                'requires_payment_method': 'failed',
                'canceled': 'cancelled',
                'requires_confirmation': 'pending'
            }

            internal_status = status_map.get(payment_intent['status'], 'processing')

            result = {
                'transaction_id': payment_intent['id'],
                'status': internal_status,
                'fee': fee,
                'estimated_arrival': datetime.now() + timedelta(minutes=1),
                'additional_data': {
                    'payment_intent_id': payment_intent['id'],
                    'status': payment_intent['status'],
                    'amount_received': payment_intent.get('amount_received', 0) / 100,
                }
            }

            # If requires action (e.g., 3DS), include the client secret
            if payment_intent['status'] == 'requires_action':
                result['redirect_url'] = None  # Frontend will handle 3DS with client_secret
                result['additional_data']['client_secret'] = payment_intent['client_secret']
                result['additional_data']['next_action'] = payment_intent.get('next_action')

            return result

        except PaymentProcessorError as e:
            # Check if it's a card decline
            error_msg = str(e).lower()
            if any(word in error_msg for word in ['declined', 'insufficient', 'invalid']):
                return {
                    'transaction_id': None,
                    'status': 'failed',
                    'fee': fee,
                    'estimated_arrival': None,
                    'additional_data': {
                        'error': str(e),
                        'failure_reason': 'card_declined'
                    }
                }
            raise

    async def process_withdrawal(
        self,
        user_id: UUID,
        amount: Decimal,
        payment_method_id: UUID,
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Process card withdrawal (refund) via Stripe

        Note: For withdrawals to cards, you typically use Stripe Connect transfers
        or external bank transfers. Card withdrawals are not standard.

        This method creates a pending withdrawal that will be processed via
        bank transfer or other means.
        """
        # Fetch payment method
        payment_method = await self._fetch_payment_method(payment_method_id)

        # Calculate fee
        fee = await self.calculate_fee(amount, 'withdrawal', payment_method['type'])

        # For card withdrawals, we typically don't send money back to cards
        # Instead, we'd use bank account transfers via Stripe Connect
        # This is a placeholder for that flow

        return {
            'transaction_id': f"pending_withdrawal_{datetime.now().timestamp()}",
            'status': 'processing',
            'fee': fee,
            'estimated_arrival': datetime.now() + timedelta(days=3),
            'additional_data': {
                'message': 'Card withdrawals are processed as bank transfers',
                'processing_method': 'bank_transfer'
            }
        }

    async def verify_payment_method(
        self,
        payment_method_id: UUID,
        verification_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Verify card payment method

        For Stripe, cards are verified during first successful payment
        Alternatively, can use SetupIntent for verification without charge
        """
        payment_method = await self._fetch_payment_method(payment_method_id)
        stripe_pm_id = payment_method.get('account_number_encrypted')

        if not stripe_pm_id:
            return {
                'verified': False,
                'message': 'Stripe payment method not found',
                'additional_data': {}
            }

        try:
            # Retrieve payment method from Stripe to verify it exists
            pm = await self._make_request("GET", f"payment_methods/{stripe_pm_id}")

            if pm and pm.get('card'):
                return {
                    'verified': True,
                    'message': 'Card verified successfully',
                    'additional_data': {
                        'brand': pm['card']['brand'],
                        'last4': pm['card']['last4'],
                        'exp_month': pm['card']['exp_month'],
                        'exp_year': pm['card']['exp_year']
                    }
                }
            else:
                return {
                    'verified': False,
                    'message': 'Unable to verify card',
                    'additional_data': {}
                }

        except PaymentProcessorError:
            return {
                'verified': False,
                'message': 'Card verification failed',
                'additional_data': {}
            }

    async def get_transaction_status(
        self,
        external_transaction_id: str
    ) -> dict[str, Any]:
        """
        Query Stripe transaction status

        Args:
            external_transaction_id: Stripe PaymentIntent ID
        """
        try:
            payment_intent = await self._make_request(
                "GET",
                f"payment_intents/{external_transaction_id}"
            )

            status_map = {
                'succeeded': 'completed',
                'processing': 'processing',
                'requires_action': 'pending',
                'requires_payment_method': 'failed',
                'canceled': 'cancelled'
            }

            internal_status = status_map.get(payment_intent['status'], 'processing')

            result = {
                'status': internal_status,
                'completed_at': None,
                'failure_reason': None,
                'additional_data': payment_intent
            }

            if internal_status == 'completed':
                # Convert timestamp to datetime
                result['completed_at'] = datetime.fromtimestamp(
                    payment_intent.get('created', 0)
                )
            elif internal_status == 'failed':
                result['failure_reason'] = payment_intent.get('last_payment_error', {}).get('message')

            return result

        except PaymentProcessorError as e:
            raise PaymentProcessorError(f"Failed to query Stripe transaction: {str(e)}")

    async def calculate_fee(
        self,
        amount: Decimal,
        transaction_type: str,
        payment_method_type: str
    ) -> Decimal:
        """
        Calculate Stripe processing fee

        Stripe standard pricing (US):
        - 2.9% + $0.30 per successful card charge

        Adjust based on your Stripe pricing agreement
        """
        # Standard Stripe fee: 2.9% + $0.30
        fee_percent = Decimal('0.029')  # 2.9%
        fixed_fee = Decimal('0.30')

        fee = (amount * fee_percent) + fixed_fee

        return fee

    async def refund_transaction(
        self,
        external_transaction_id: str,
        amount: Decimal | None = None,
        reason: str | None = None
    ) -> dict[str, Any]:
        """
        Refund a Stripe transaction

        Args:
            external_transaction_id: Stripe PaymentIntent ID
            amount: Amount to refund (None for full refund)
            reason: Reason for refund

        Returns:
            Refund details
        """
        data = {
            "payment_intent": external_transaction_id,
        }

        if amount:
            # Convert to cents
            data["amount"] = int(amount * 100)

        if reason:
            data["reason"] = reason

        try:
            refund = await self._make_request("POST", "refunds", data)

            return {
                'refund_id': refund['id'],
                'status': refund['status'],
                'amount': Decimal(refund['amount']) / 100,
                'created': datetime.fromtimestamp(refund['created'])
            }

        except PaymentProcessorError as e:
            raise PaymentProcessorError(f"Stripe refund failed: {str(e)}")

    def _handle_webhook(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Handle Stripe webhook events

        Stripe sends webhooks for various events:
        - payment_intent.succeeded
        - payment_intent.payment_failed
        - charge.refunded
        - etc.
        """
        event_type = payload.get('type')
        data = payload.get('data', {}).get('object', {})

        if event_type == 'payment_intent.succeeded':
            return {
                'event': 'payment_success',
                'transaction_id': data['id'],
                'status': 'completed',
                'amount': Decimal(data['amount']) / 100,
                'additional_data': data
            }

        elif event_type == 'payment_intent.payment_failed':
            return {
                'event': 'payment_failed',
                'transaction_id': data['id'],
                'status': 'failed',
                'failure_reason': data.get('last_payment_error', {}).get('message'),
                'additional_data': data
            }

        elif event_type == 'charge.refunded':
            return {
                'event': 'refund_processed',
                'transaction_id': data.get('payment_intent'),
                'status': 'refunded',
                'amount': Decimal(data['amount_refunded']) / 100,
                'additional_data': data
            }

        return {
            'event': event_type,
            'additional_data': data
        }

    async def create_setup_intent(self, user_id: UUID) -> dict[str, Any]:
        """
        Create a SetupIntent for saving card without charging

        This is used for adding cards with proper SCA (Strong Customer Authentication)

        Returns:
            SetupIntent with client_secret for frontend
        """
        data = {
            "usage": "off_session",
            "metadata[user_id]": str(user_id),
            "metadata[platform]": "CIFT Markets"
        }

        setup_intent = await self._make_request("POST", "setup_intents", data)

        return {
            'setup_intent_id': setup_intent['id'],
            'client_secret': setup_intent['client_secret'],
            'status': setup_intent['status']
        }
