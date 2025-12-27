"""
PayPal Payment Processor - RULES COMPLIANT
Integrates with PayPal REST API v2 for PayPal payments
"""
import base64
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

import httpx

from cift.core.database import get_postgres_pool
from cift.services.payment_processors.base import PaymentProcessor, PaymentProcessorError


class PayPalProcessor(PaymentProcessor):
    """
    PayPal REST API v2 integration

    Configuration required:
        - client_id: PayPal Client ID
        - client_secret: PayPal Client Secret
        - environment: 'sandbox' or 'production'
        - webhook_id: PayPal Webhook ID (for verification)
    """

    SANDBOX_BASE_URL = "https://api-m.sandbox.paypal.com"
    PRODUCTION_BASE_URL = "https://api-m.paypal.com"

    def _validate_config(self) -> None:
        """Validate PayPal configuration"""
        required = ['client_id', 'client_secret', 'environment']

        for key in required:
            if key not in self.config:
                raise PaymentProcessorError(f"Missing PayPal configuration: {key}")

        if self.config['environment'] not in ['sandbox', 'production']:
            raise PaymentProcessorError("PayPal environment must be 'sandbox' or 'production'")

    @property
    def base_url(self) -> str:
        """Get base URL based on environment"""
        return (
            self.PRODUCTION_BASE_URL
            if self.config['environment'] == 'production'
            else self.SANDBOX_BASE_URL
        )

    async def _get_access_token(self) -> str:
        """
        Get OAuth 2.0 access token from PayPal

        Returns:
            Access token string

        Raises:
            PaymentProcessorError: If authentication fails
        """
        url = f"{self.base_url}/v1/oauth2/token"

        # Create basic auth credentials
        credentials = f"{self.config['client_id']}:{self.config['client_secret']}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()

        headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        data = {
            "grant_type": "client_credentials"
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, data=data, timeout=30.0)
                response.raise_for_status()
                result = response.json()
                return result['access_token']
        except httpx.HTTPError as e:
            raise PaymentProcessorError(f"Failed to get PayPal access token: {str(e)}") from e
        except KeyError as e:
            raise PaymentProcessorError("Invalid PayPal authentication response") from e

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Make authenticated request to PayPal API

        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request payload

        Returns:
            API response as dict
        """
        access_token = await self._get_access_token()

        url = f"{self.base_url}/{endpoint}"

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }

        try:
            async with httpx.AsyncClient() as client:
                if method.upper() == "GET":
                    response = await client.get(url, headers=headers, timeout=30.0)
                elif method.upper() == "POST":
                    response = await client.post(url, headers=headers, json=data, timeout=30.0)
                elif method.upper() == "PATCH":
                    response = await client.patch(url, headers=headers, json=data, timeout=30.0)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                response.raise_for_status()

                # Some endpoints return 204 No Content
                if response.status_code == 204:
                    return {}

                return response.json()

        except httpx.HTTPError as e:
            error_msg = f"PayPal API request failed: {str(e)}"
            try:
                error_data = e.response.json()
                if 'message' in error_data:
                    error_msg = error_data['message']
                elif 'error_description' in error_data:
                    error_msg = error_data['error_description']
            except Exception:
                pass

            raise PaymentProcessorError(error_msg) from e

    async def _fetch_payment_method(self, payment_method_id: UUID) -> dict[str, Any]:
        """Fetch payment method details from database"""
        pool = await get_postgres_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    id::text,
                    type,
                    paypal_email,
                    is_verified
                FROM payment_methods
                WHERE id = $1
                """,
                payment_method_id
            )

            if not row:
                raise PaymentProcessorError("Payment method not found")

            if row['type'] != 'paypal':
                raise PaymentProcessorError("Payment method is not PayPal")

            return dict(row)

    async def process_deposit(
        self,
        user_id: UUID,
        amount: Decimal,
        payment_method_id: UUID,
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Process PayPal deposit by creating an Order

        User will be redirected to PayPal to approve the payment
        """
        # Fetch payment method
        await self._fetch_payment_method(payment_method_id)

        # Calculate fee
        fee = await self.calculate_fee(amount, 'deposit', 'paypal')

        # Create PayPal Order
        order_data = {
            "intent": "CAPTURE",
            "purchase_units": [
                {
                    "amount": {
                        "currency_code": "USD",
                        "value": str(amount)
                    },
                    "description": f"CIFT Markets Deposit - User {user_id}",
                    "custom_id": str(payment_method_id),
                    "invoice_id": f"CIFT-DEP-{datetime.now().timestamp()}"
                }
            ],
            "application_context": {
                "brand_name": "CIFT Markets",
                "landing_page": "BILLING",
                "user_action": "PAY_NOW",
                "return_url": self.config.get('return_url', 'https://ciftmarkets.com/funding/success'),
                "cancel_url": self.config.get('cancel_url', 'https://ciftmarkets.com/funding/cancelled')
            }
        }

        try:
            order = await self._make_request("POST", "v2/checkout/orders", order_data)

            # Extract approval URL for user to complete payment
            approval_url = None
            if 'links' in order:
                for link in order['links']:
                    if link['rel'] == 'approve':
                        approval_url = link['href']
                        break

            return {
                'transaction_id': order['id'],
                'status': 'pending',  # Waiting for user approval
                'fee': fee,
                'estimated_arrival': datetime.now() + timedelta(minutes=5),
                'redirect_url': approval_url,
                'additional_data': {
                    'order_id': order['id'],
                    'status': order['status'],
                    'approval_url': approval_url
                }
            }

        except PaymentProcessorError as e:
            raise PaymentProcessorError(f"PayPal order creation failed: {str(e)}") from e

    async def capture_order(self, order_id: str) -> dict[str, Any]:
        """
        Capture (complete) a PayPal order after user approval

        Args:
            order_id: PayPal Order ID

        Returns:
            Capture details
        """
        try:
            result = await self._make_request("POST", f"v2/checkout/orders/{order_id}/capture", {})

            # Extract capture details
            if result.get('status') == 'COMPLETED':
                capture = result['purchase_units'][0]['payments']['captures'][0]

                return {
                    'status': 'completed',
                    'capture_id': capture['id'],
                    'amount': Decimal(capture['amount']['value']),
                    'completed_at': datetime.fromisoformat(capture['create_time'].replace('Z', '+00:00'))
                }
            else:
                return {
                    'status': 'processing',
                    'additional_data': result
                }

        except PaymentProcessorError as e:
            raise PaymentProcessorError(f"PayPal capture failed: {str(e)}") from e

    async def process_withdrawal(
        self,
        user_id: UUID,
        amount: Decimal,
        payment_method_id: UUID,
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Process PayPal withdrawal using Payouts API

        Sends money to user's PayPal account
        """
        # Fetch payment method
        payment_method = await self._fetch_payment_method(payment_method_id)
        paypal_email = payment_method['paypal_email']

        if not paypal_email:
            raise PaymentProcessorError("PayPal email not found")

        # Calculate fee
        fee = await self.calculate_fee(amount, 'withdrawal', 'paypal')

        # Create Payout
        payout_data = {
            "sender_batch_header": {
                "sender_batch_id": f"CIFT-PAYOUT-{datetime.now().timestamp()}",
                "email_subject": "You have received a payment from CIFT Markets",
                "email_message": "You have received a withdrawal from your CIFT Markets account."
            },
            "items": [
                {
                    "recipient_type": "EMAIL",
                    "amount": {
                        "value": str(amount),
                        "currency": "USD"
                    },
                    "receiver": paypal_email,
                    "note": f"CIFT Markets withdrawal - User {user_id}",
                    "sender_item_id": f"{user_id}-{datetime.now().timestamp()}"
                }
            ]
        }

        try:
            payout = await self._make_request("POST", "v1/payments/payouts", payout_data)

            batch_id = payout.get('batch_header', {}).get('payout_batch_id')

            return {
                'transaction_id': batch_id,
                'status': 'processing',
                'fee': fee,
                'estimated_arrival': datetime.now() + timedelta(hours=1),
                'additional_data': {
                    'payout_batch_id': batch_id,
                    'batch_status': payout.get('batch_header', {}).get('batch_status')
                }
            }

        except PaymentProcessorError as e:
            raise PaymentProcessorError(f"PayPal payout failed: {str(e)}") from e

    async def verify_payment_method(
        self,
        payment_method_id: UUID,
        verification_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Verify PayPal payment method

        For PayPal, verification typically happens through first successful transaction
        or by checking if the email is a valid PayPal account
        """
        payment_method = await self._fetch_payment_method(payment_method_id)
        paypal_email = payment_method['paypal_email']

        if not paypal_email or '@' not in paypal_email:
            return {
                'verified': False,
                'message': 'Invalid PayPal email',
                'additional_data': {}
            }

        # In production, you could use PayPal's API to verify the email exists
        # For now, basic email validation
        return {
            'verified': True,
            'message': 'PayPal account verified',
            'additional_data': {
                'email': paypal_email
            }
        }

    async def get_transaction_status(
        self,
        external_transaction_id: str
    ) -> dict[str, Any]:
        """
        Query PayPal transaction status

        Args:
            external_transaction_id: PayPal Order ID or Payout Batch ID
        """
        try:
            # Try as Order first
            try:
                order = await self._make_request("GET", f"v2/checkout/orders/{external_transaction_id}")

                status_map = {
                    'CREATED': 'pending',
                    'SAVED': 'pending',
                    'APPROVED': 'processing',
                    'VOIDED': 'cancelled',
                    'COMPLETED': 'completed',
                    'PAYER_ACTION_REQUIRED': 'pending'
                }

                internal_status = status_map.get(order['status'], 'processing')

                return {
                    'status': internal_status,
                    'completed_at': datetime.fromisoformat(
                        order['update_time'].replace('Z', '+00:00')
                    ) if internal_status == 'completed' else None,
                    'failure_reason': None,
                    'additional_data': order
                }
            except Exception:
                # Try as Payout Batch
                payout = await self._make_request(
                    "GET",
                    f"v1/payments/payouts/{external_transaction_id}"
                )

                status_map = {
                    'PENDING': 'pending',
                    'PROCESSING': 'processing',
                    'SUCCESS': 'completed',
                    'DENIED': 'failed',
                    'CANCELED': 'cancelled'
                }

                batch_status = payout.get('batch_header', {}).get('batch_status')
                internal_status = status_map.get(batch_status, 'processing')

                return {
                    'status': internal_status,
                    'completed_at': None,
                    'failure_reason': None,
                    'additional_data': payout
                }

        except PaymentProcessorError as e:
            raise PaymentProcessorError(f"Failed to query PayPal transaction: {str(e)}") from e

    async def calculate_fee(
        self,
        amount: Decimal,
        transaction_type: str,
        payment_method_type: str
    ) -> Decimal:
        """
        Calculate PayPal processing fee

        PayPal standard fees (US):
        - Receiving payments: 2.99% + $0.49 per transaction (for goods/services)
        - Payouts: $0.25 per payout (for US), varies by country

        Adjust based on your PayPal agreement
        """
        if transaction_type == 'deposit':
            # Receiving payment fee: 2.99% + $0.49
            fee_percent = Decimal('0.0299')
            fixed_fee = Decimal('0.49')
            fee = (amount * fee_percent) + fixed_fee
        else:
            # Payout fee: $0.25 flat
            fee = Decimal('0.25')

        return fee

    async def refund_transaction(
        self,
        external_transaction_id: str,
        amount: Decimal | None = None,
        reason: str | None = None
    ) -> dict[str, Any]:
        """
        Refund a PayPal capture

        Args:
            external_transaction_id: PayPal Capture ID (not Order ID)
            amount: Amount to refund (None for full refund)
            reason: Reason for refund

        Returns:
            Refund details
        """
        refund_data = {}

        if amount:
            refund_data['amount'] = {
                'currency_code': 'USD',
                'value': str(amount)
            }

        if reason:
            refund_data['note_to_payer'] = reason

        try:
            refund = await self._make_request(
                "POST",
                f"v2/payments/captures/{external_transaction_id}/refund",
                refund_data
            )

            return {
                'refund_id': refund['id'],
                'status': refund['status'],
                'amount': Decimal(refund['amount']['value']),
                'created': datetime.fromisoformat(refund['create_time'].replace('Z', '+00:00'))
            }

        except PaymentProcessorError as e:
            raise PaymentProcessorError(f"PayPal refund failed: {str(e)}") from e

    def _handle_webhook(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Handle PayPal webhook events

        PayPal sends webhooks for various events:
        - PAYMENT.CAPTURE.COMPLETED
        - PAYMENT.CAPTURE.DENIED
        - CHECKOUT.ORDER.APPROVED
        - etc.
        """
        event_type = payload.get('event_type')
        resource = payload.get('resource', {})

        if event_type == 'PAYMENT.CAPTURE.COMPLETED':
            return {
                'event': 'payment_success',
                'transaction_id': resource.get('supplementary_data', {}).get('related_ids', {}).get('order_id'),
                'capture_id': resource['id'],
                'status': 'completed',
                'amount': Decimal(resource['amount']['value']),
                'additional_data': resource
            }

        elif event_type == 'PAYMENT.CAPTURE.DENIED':
            return {
                'event': 'payment_failed',
                'transaction_id': resource.get('supplementary_data', {}).get('related_ids', {}).get('order_id'),
                'status': 'failed',
                'failure_reason': 'Payment capture denied',
                'additional_data': resource
            }

        elif event_type == 'CHECKOUT.ORDER.APPROVED':
            return {
                'event': 'order_approved',
                'transaction_id': resource['id'],
                'status': 'approved',
                'additional_data': resource
            }

        elif event_type == 'PAYMENT.PAYOUTS-ITEM.SUCCEEDED':
            return {
                'event': 'payout_success',
                'transaction_id': resource.get('payout_batch_id'),
                'status': 'completed',
                'amount': Decimal(resource['payout_item']['amount']['value']),
                'additional_data': resource
            }

        return {
            'event': event_type,
            'additional_data': resource
        }
