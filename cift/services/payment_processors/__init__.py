"""
Payment Processors Package - RULES COMPLIANT
Central module for all payment processor integrations
"""
from typing import Any, Dict

from cift.services.payment_processors.alpaca_processor import AlpacaProcessor
from cift.services.payment_processors.base import PaymentProcessor, PaymentProcessorError
from cift.services.payment_processors.crypto import CryptoProcessor
from cift.services.payment_processors.mpesa import MpesaProcessor
from cift.services.payment_processors.paypal import PayPalProcessor
from cift.services.payment_processors.stripe_processor import StripeProcessor

__all__ = [
    'PaymentProcessor',
    'PaymentProcessorError',
    'MpesaProcessor',
    'StripeProcessor',
    'PayPalProcessor',
    'CryptoProcessor',
    'AlpacaProcessor',
    'get_payment_processor'
]


def get_payment_processor(
    payment_method_type: str,
    config: dict[str, Any]
) -> PaymentProcessor:
    """
    Factory function to get the appropriate payment processor

    Args:
        payment_method_type: Type of payment method
            ('mpesa', 'debit_card', 'credit_card', 'paypal', 'crypto_wallet', 'bank_account')
        config: Configuration dictionary for the processor

    Returns:
        Initialized payment processor instance

    Raises:
        PaymentProcessorError: If payment method type is unsupported

    Example:
        >>> config = {'consumer_key': '...', 'consumer_secret': '...', ...}
        >>> processor = get_payment_processor('mpesa', config)
        >>> result = await processor.process_deposit(user_id, amount, payment_method_id)
    """
    processor_map = {
        'mpesa': MpesaProcessor,
        'debit_card': StripeProcessor,
        'credit_card': StripeProcessor,
        'paypal': PayPalProcessor,
        'crypto_wallet': CryptoProcessor,
        'bank_account': AlpacaProcessor
    }

    processor_class = processor_map.get(payment_method_type)

    if not processor_class:
        raise PaymentProcessorError(
            f"Unsupported payment method type: {payment_method_type}. "
            f"Supported types: {', '.join(processor_map.keys())}"
        )

    return processor_class(config)
