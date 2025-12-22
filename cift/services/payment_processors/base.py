"""
Base Payment Processor - RULES COMPLIANT
Abstract base class for all payment processors
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any
from uuid import UUID


class PaymentProcessorError(Exception):
    """Base exception for payment processor errors"""

    pass


class PaymentProcessor(ABC):
    """
    Abstract base class for payment processors
    All payment integrations must extend this class
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize payment processor with configuration

        Args:
            config: Configuration dictionary with API keys, endpoints, etc.
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate required configuration parameters"""
        pass

    @abstractmethod
    async def process_deposit(
        self,
        user_id: UUID,
        amount: Decimal,
        payment_method_id: UUID,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Process a deposit transaction

        Args:
            user_id: User UUID
            amount: Amount to deposit
            payment_method_id: Payment method UUID
            metadata: Additional metadata

        Returns:
            Dict with transaction details
        """
        pass

    async def link_account(
        self, user_id: UUID, account_details: dict[str, Any], metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Link an external account (optional implementation)

        Args:
            user_id: User UUID
            account_details: Account details (numbers, tokens, etc)
            metadata: Additional metadata

        Returns:
            Dict with linked account details (e.g. external ID)
        """
        return {}

    @abstractmethod
    async def process_withdrawal(
        self,
        user_id: UUID,
        amount: Decimal,
        payment_method_id: UUID,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Process a withdrawal transaction

        Args:
            user_id: User UUID
            amount: Amount to withdraw
            payment_method_id: Payment method UUID
            metadata: Additional transaction metadata

        Returns:
            Dict containing:
                - transaction_id: External processor transaction ID
                - status: Transaction status
                - fee: Processing fee amount
                - estimated_arrival: Estimated completion datetime
                - additional_data: Any processor-specific data

        Raises:
            PaymentProcessorError: If processing fails
        """
        pass

    @abstractmethod
    async def verify_payment_method(
        self, payment_method_id: UUID, verification_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Verify a payment method (e.g., micro-deposits, SMS code, etc.)

        Args:
            payment_method_id: Payment method UUID
            verification_data: Verification details (amounts, codes, etc.)

        Returns:
            Dict containing:
                - verified: Boolean indicating success
                - message: User-friendly message
                - additional_data: Any processor-specific data

        Raises:
            PaymentProcessorError: If verification fails
        """
        pass

    @abstractmethod
    async def get_transaction_status(self, external_transaction_id: str) -> dict[str, Any]:
        """
        Query transaction status from external processor

        Args:
            external_transaction_id: Processor's transaction ID

        Returns:
            Dict containing:
                - status: Current transaction status
                - completed_at: Completion datetime if completed
                - failure_reason: Reason if failed
                - additional_data: Any processor-specific data

        Raises:
            PaymentProcessorError: If query fails
        """
        pass

    @abstractmethod
    async def calculate_fee(
        self, amount: Decimal, transaction_type: str, payment_method_type: str
    ) -> Decimal:
        """
        Calculate processing fee for a transaction

        Args:
            amount: Transaction amount
            transaction_type: 'deposit' or 'withdrawal'
            payment_method_type: Type of payment method

        Returns:
            Processing fee amount
        """
        pass

    async def refund_transaction(
        self, external_transaction_id: str, amount: Decimal | None = None, reason: str | None = None
    ) -> dict[str, Any]:
        """
        Refund a transaction (optional, not all processors support this)

        Args:
            external_transaction_id: Processor's transaction ID
            amount: Amount to refund (None for full refund)
            reason: Reason for refund

        Returns:
            Dict containing refund details

        Raises:
            NotImplementedError: If processor doesn't support refunds
            PaymentProcessorError: If refund fails
        """
        raise NotImplementedError("Refunds not supported by this processor")

    def _handle_webhook(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Handle webhook callbacks from payment processor (optional)

        Args:
            payload: Webhook payload

        Returns:
            Dict containing parsed webhook data

        Raises:
            NotImplementedError: If webhooks not supported
        """
        raise NotImplementedError("Webhooks not supported by this processor")
