"""
Payment Processor Service - RULES COMPLIANT
Unified payment processing facade that routes to appropriate processor
Supports: Stripe (cards), M-Pesa, PayPal, Crypto (Bitcoin/Ethereum)
"""

from decimal import Decimal
from typing import Any
from uuid import UUID

from cift.core.logging import logger
from cift.services.payment_config import PaymentConfig
from cift.services.payment_processors import PaymentProcessorError, get_payment_processor


class PaymentProcessor:
    """
    Unified payment processor facade - RULES COMPLIANT

    Routes payment requests to appropriate processor based on payment method type
    Supports: M-Pesa, Stripe, PayPal, Crypto
    """

    def __init__(self):
        """Initialize payment processor - loads config from environment"""
        self.enabled = True  # Always enabled
        logger.info("Payment processor initialized")

        # Log which payment methods are configured
        available = PaymentConfig.get_available_payment_methods()
        if available:
            logger.info(f"Configured payment methods: {', '.join(available)}")
        else:
            logger.warning("No payment methods configured - using simulation mode")

    def _get_processor(self, payment_method_type: str):
        """
        Get the appropriate payment processor for a payment method type

        Args:
            payment_method_type: Type of payment method

        Returns:
            Payment processor instance
        """
        config = PaymentConfig.get_config_for_payment_type(payment_method_type)

        if not config or not PaymentConfig.is_payment_type_configured(payment_method_type):
            logger.warning(
                f"Payment method {payment_method_type} not configured - using simulation mode"
            )
            return None

        try:
            return get_payment_processor(payment_method_type, config)
        except PaymentProcessorError as e:
            logger.error(f"Failed to get processor for {payment_method_type}: {str(e)}")
            return None

    async def create_payment_intent(
        self,
        amount: Decimal,
        currency: str = "usd",
        payment_method_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create payment intent for card payments (routes to Stripe)

        Args:
            amount: Amount in dollars
            currency: Currency code (default: usd)
            payment_method_id: Payment method UUID from database
            metadata: Additional metadata to attach

        Returns:
            Dict with payment intent details
        """
        processor = self._get_processor("credit_card")

        if not processor:
            # Simulation mode
            logger.info("Stripe not configured - simulating card payment")
            return {
                "id": "pi_simulated",
                "status": "succeeded",
                "amount": int(amount * 100),
                "currency": currency,
                "simulation": True,
            }

        try:
            # Use new Stripe processor
            user_id = UUID(metadata.get("user_id")) if metadata and "user_id" in metadata else None
            result = await processor.process_deposit(
                user_id=user_id,
                amount=amount,
                payment_method_id=UUID(payment_method_id) if payment_method_id else None,
                metadata=metadata,
            )

            return {
                "id": result.get("transaction_id"),
                "status": result.get("status"),
                "amount": int(amount * 100),
                "currency": currency,
                "simulation": False,
            }
        except PaymentProcessorError as e:
            logger.error(f"Card payment failed: {str(e)}")
            raise RuntimeError(f"Card payment error: {str(e)}") from e

    async def create_bank_transfer(
        self, amount: Decimal, bank_account_id: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Create ACH bank transfer via Alpaca

        Args:
            amount: Amount in dollars
            bank_account_id: Payment method UUID from database
            metadata: Additional metadata

        Returns:
            Dict with transfer details
        """
        processor = self._get_processor("bank_account")

        if not processor:
            # Simulation mode
            logger.info(
                f"Alpaca not configured - simulating bank transfer: ${amount} to {bank_account_id}"
            )
            return {
                "id": f"ba_sim_{bank_account_id[:8]}",
                "status": "pending",
                "amount": int(amount * 100),
                "simulation": True,
                "expected_arrival_days": 3,
            }

        try:
            user_id = UUID(metadata.get("user_id")) if metadata and "user_id" in metadata else None
            result = await processor.process_deposit(
                user_id=user_id,
                amount=amount,
                payment_method_id=UUID(bank_account_id),
                metadata=metadata,
            )
            return result
        except Exception as e:
            logger.error(f"Bank transfer failed: {str(e)}")
            raise RuntimeError(f"Bank transfer error: {str(e)}") from e

    async def create_brokerage_account(self, user_details: dict[str, Any]) -> dict[str, Any]:
        """
        Create a brokerage account for the user (Alpaca)

        Args:
            user_details: KYC details

        Returns:
            Dict with account details
        """
        processor = self._get_processor("bank_account")  # Alpaca is mapped to bank_account
        if not processor:
            logger.info("Alpaca not configured - simulating account creation")
            return {"id": "sim_alpaca_id", "status": "ACTIVE", "simulation": True}

        if hasattr(processor, "create_account"):
            return await processor.create_account(user_details)
        else:
            raise NotImplementedError("Processor does not support account creation")

    async def link_external_account(
        self,
        user_id: UUID,
        payment_type: str,
        account_details: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Link an external account via the appropriate processor

        Args:
            user_id: User UUID
            payment_type: 'bank_account', etc.
            account_details: Account details
            metadata: Additional metadata

        Returns:
            Dict with linked account details
        """
        processor = self._get_processor(payment_type)
        if not processor:
            logger.info(f"Processor for {payment_type} not configured - simulating link")
            return {"id": "sim_linked_id", "status": "verified", "simulation": True}

        return await processor.link_account(user_id, account_details, metadata)

    async def verify_micro_deposits(
        self, bank_account_id: str, amounts: tuple[int, int]
    ) -> dict[str, Any]:
        """
        Verify bank account with micro-deposit amounts (simulated)

        Args:
            bank_account_id: Bank account to verify
            amounts: Tuple of two micro-deposit amounts in cents

        Returns:
            Dict with verification result
        """
        # Bank verification simulated
        logger.info(f"Bank account verification: {bank_account_id}, amounts: {amounts}")
        return {"verified": True, "simulation": True}

    async def process_withdrawal(
        self,
        amount: Decimal,
        payment_method_type: str,
        payment_method_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Process withdrawal to user's payment method

        Args:
            amount: Amount in dollars
            payment_method_type: Type of payment method
            payment_method_id: Payment method UUID from database
            metadata: Additional metadata

        Returns:
            Dict with withdrawal details
        """
        processor = self._get_processor(payment_method_type)

        if not processor:
            # Simulation mode
            logger.info(f"Simulating {payment_method_type} withdrawal: ${amount}")
            return {
                "id": f"withdrawal_sim_{payment_method_id[:8]}",
                "status": "processing",
                "amount": float(amount),
                "simulation": True,
                "expected_completion_days": 2,
            }

        try:
            # Use appropriate processor
            user_id = UUID(metadata.get("user_id")) if metadata and "user_id" in metadata else None
            result = await processor.process_withdrawal(
                user_id=user_id,
                amount=amount,
                payment_method_id=UUID(payment_method_id),
                metadata=metadata,
            )

            return {
                "id": result.get("transaction_id"),
                "status": result.get("status"),
                "amount": float(amount),
                "simulation": False,
                "expected_completion_days": 2,
            }
        except PaymentProcessorError as e:
            logger.error(f"Withdrawal failed ({payment_method_type}): {str(e)}")
            raise RuntimeError(f"Withdrawal error: {str(e)}") from e

    async def get_transaction_status(
        self, external_id: str, transaction_type: str = "payment_intent"
    ) -> dict[str, Any]:
        """
        Check status of external transaction (simulated)

        Args:
            external_id: External transaction ID
            transaction_type: Type of transaction

        Returns:
            Dict with current status
        """
        logger.info(f"Checking transaction status: {external_id}")
        return {"status": "completed", "simulation": True}

    def calculate_fee(
        self, amount: Decimal, payment_method_type: str, transfer_type: str = "standard"
    ) -> Decimal:
        """
        Calculate transaction fee using appropriate processor

        Args:
            amount: Transaction amount
            payment_method_type: Type of payment method
            transfer_type: 'instant' or 'standard'

        Returns:
            Decimal: Fee amount
        """
        # Fallback fee calculation if processor not available
        fee_schedule = {
            "bank_account": (
                Decimal("0.00") if transfer_type == "standard" else amount * Decimal("0.015")
            ),
            "debit_card": amount * Decimal("0.029") + Decimal("0.30"),
            "credit_card": amount * Decimal("0.029") + Decimal("0.30"),
            "paypal": amount * Decimal("0.0299") + Decimal("0.49"),
            "mpesa": amount * Decimal("0.025"),  # 2.5%
            "crypto_wallet": Decimal("5.00"),  # Flat fee
        }

        return fee_schedule.get(payment_method_type, Decimal("0.00"))


# Global instance
payment_processor = PaymentProcessor()
