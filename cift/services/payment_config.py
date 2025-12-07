"""
Payment Processor Configuration - RULES COMPLIANT
Centralized configuration management for payment processors
"""
import os
from typing import Dict, Any


class PaymentConfig:
    """
    Centralized payment processor configuration
    
    Loads configuration from environment variables
    """
    
    @staticmethod
    def get_mpesa_config() -> Dict[str, Any]:
        """
        Get M-Pesa configuration
        
        Required environment variables:
        - MPESA_CONSUMER_KEY
        - MPESA_CONSUMER_SECRET
        - MPESA_BUSINESS_SHORT_CODE
        - MPESA_PASSKEY
        - MPESA_ENVIRONMENT (sandbox or production)
        - MPESA_CALLBACK_URL
        """
        return {
            'consumer_key': os.getenv('MPESA_CONSUMER_KEY', ''),
            'consumer_secret': os.getenv('MPESA_CONSUMER_SECRET', ''),
            'business_short_code': os.getenv('MPESA_BUSINESS_SHORT_CODE', ''),
            'passkey': os.getenv('MPESA_PASSKEY', ''),
            'environment': os.getenv('MPESA_ENVIRONMENT', 'sandbox'),
            'callback_url': os.getenv('MPESA_CALLBACK_URL', 'http://localhost:8000/api/v1/webhooks/mpesa'),
            'initiator_name': os.getenv('MPESA_INITIATOR_NAME', 'apiuser'),
            'security_credential': os.getenv('MPESA_SECURITY_CREDENTIAL', ''),
            'timeout_url': os.getenv('MPESA_TIMEOUT_URL', 'http://localhost:8000/api/v1/webhooks/mpesa/timeout')
        }
    
    @staticmethod
    def get_stripe_config() -> Dict[str, Any]:
        """
        Get Stripe configuration
        
        Required environment variables:
        - STRIPE_SECRET_KEY
        - STRIPE_PUBLISHABLE_KEY
        - STRIPE_WEBHOOK_SECRET (optional)
        """
        return {
            'secret_key': os.getenv('STRIPE_SECRET_KEY', ''),
            'publishable_key': os.getenv('STRIPE_PUBLISHABLE_KEY', ''),
            'webhook_secret': os.getenv('STRIPE_WEBHOOK_SECRET', ''),
            'environment': 'production' if 'sk_live' in os.getenv('STRIPE_SECRET_KEY', '') else 'sandbox'
        }
    
    @staticmethod
    def get_paypal_config() -> Dict[str, Any]:
        """
        Get PayPal configuration
        
        Required environment variables:
        - PAYPAL_CLIENT_ID
        - PAYPAL_CLIENT_SECRET
        - PAYPAL_ENVIRONMENT (sandbox or production)
        - PAYPAL_WEBHOOK_ID (optional)
        """
        return {
            'client_id': os.getenv('PAYPAL_CLIENT_ID', ''),
            'client_secret': os.getenv('PAYPAL_CLIENT_SECRET', ''),
            'environment': os.getenv('PAYPAL_ENVIRONMENT', 'sandbox'),
            'webhook_id': os.getenv('PAYPAL_WEBHOOK_ID', ''),
            'return_url': os.getenv('PAYPAL_RETURN_URL', 'http://localhost:3000/funding/success'),
            'cancel_url': os.getenv('PAYPAL_CANCEL_URL', 'http://localhost:3000/funding/cancelled')
        }
    
    @staticmethod
    def get_crypto_config() -> Dict[str, Any]:
        """
        Get Cryptocurrency configuration
        
        Required environment variables:
        - CRYPTO_BTC_DEPOSIT_ADDRESS
        - CRYPTO_ETH_DEPOSIT_ADDRESS
        - CRYPTO_BTC_HOT_WALLET_KEY (encrypted)
        - CRYPTO_ETH_HOT_WALLET_KEY (encrypted)
        - CRYPTO_CONFIRMATIONS_REQUIRED (optional, defaults to 3 for BTC, 12 for ETH)
        - ETHERSCAN_API_KEY (optional)
        """
        return {
            'deposit_addresses': {
                'bitcoin': os.getenv('CRYPTO_BTC_DEPOSIT_ADDRESS', ''),
                'ethereum': os.getenv('CRYPTO_ETH_DEPOSIT_ADDRESS', '')
            },
            'hot_wallet_private_keys': {
                'bitcoin': os.getenv('CRYPTO_BTC_HOT_WALLET_KEY', ''),
                'ethereum': os.getenv('CRYPTO_ETH_HOT_WALLET_KEY', '')
            },
            'confirmations_required': int(os.getenv('CRYPTO_CONFIRMATIONS_REQUIRED', '3')),
            'blockchain_explorer_api_key': os.getenv('ETHERSCAN_API_KEY', ''),
            'btc_node_url': os.getenv('CRYPTO_BTC_NODE_URL', ''),
            'eth_node_url': os.getenv('CRYPTO_ETH_NODE_URL', '')
        }

    @staticmethod
    def get_alpaca_config() -> Dict[str, Any]:
        """
        Get Alpaca configuration for Bank Transfers
        
        Required environment variables:
        - ALPACA_API_KEY
        - ALPACA_SECRET_KEY
        - ALPACA_BASE_URL (optional)
        """
        return {
            'api_key': os.getenv('ALPACA_API_KEY', ''),
            'secret_key': os.getenv('ALPACA_SECRET_KEY', ''),
            'base_url': os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        }
    
    @staticmethod
    def get_config_for_payment_type(payment_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific payment type
        
        Args:
            payment_type: Payment method type
            
        Returns:
            Configuration dictionary
        """
        config_map = {
            'mpesa': PaymentConfig.get_mpesa_config,
            'debit_card': PaymentConfig.get_stripe_config,
            'credit_card': PaymentConfig.get_stripe_config,
            'paypal': PaymentConfig.get_paypal_config,
            'crypto_wallet': PaymentConfig.get_crypto_config,
            'bank_account': PaymentConfig.get_alpaca_config
        }
        
        config_func = config_map.get(payment_type)
        
        if not config_func:
            return {}
        
        return config_func()
    
    @staticmethod
    def is_payment_type_configured(payment_type: str) -> bool:
        """
        Check if a payment type is properly configured
        
        Args:
            payment_type: Payment method type
            
        Returns:
            True if configured with required credentials
        """
        config = PaymentConfig.get_config_for_payment_type(payment_type)
        
        if not config:
            return False
        
        # Check if essential keys are present and non-empty
        if payment_type == 'mpesa':
            return bool(
                config.get('consumer_key') and
                config.get('consumer_secret') and
                config.get('business_short_code')
            )
        elif payment_type in ['debit_card', 'credit_card']:
            return bool(
                config.get('secret_key') and
                config.get('publishable_key')
            )
        elif payment_type == 'paypal':
            return bool(
                config.get('client_id') and
                config.get('client_secret')
            )
        elif payment_type == 'crypto_wallet':
            return bool(
                config.get('deposit_addresses', {}).get('bitcoin') or
                config.get('deposit_addresses', {}).get('ethereum')
            )
        elif payment_type == 'bank_account':
            return bool(
                config.get('api_key') and
                config.get('secret_key')
            )
        
        return False
    
    @staticmethod
    def get_available_payment_methods() -> list[str]:
        """
        Get list of payment methods that are properly configured
        
        Returns:
            List of available payment method types
        """
        all_types = ['mpesa', 'debit_card', 'credit_card', 'paypal', 'crypto_wallet', 'bank_account']
        
        return [
            payment_type
            for payment_type in all_types
            if PaymentConfig.is_payment_type_configured(payment_type)
        ]
