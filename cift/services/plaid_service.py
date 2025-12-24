"""
Plaid Integration Service - REAL Bank Account Verification & ACH Transfers

This service provides real bank account linking, verification, and ACH transfers
via Plaid's APIs. Used for actual money movement between user bank accounts
and the platform's omnibus account.

Required Environment Variables:
- PLAID_CLIENT_ID: Your Plaid client ID
- PLAID_SECRET: Your Plaid secret (sandbox/development/production)
- PLAID_ENV: Environment (sandbox, development, production)

Get your keys at: https://dashboard.plaid.com/developers/keys
"""

import os
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from cift.core.logging import logger

# Try to import plaid, handle gracefully if not installed
try:
    import plaid
    from plaid.api import plaid_api
    from plaid.model.country_code import CountryCode
    from plaid.model.link_token_create_request import LinkTokenCreateRequest
    from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
    from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
    from plaid.model.auth_get_request import AuthGetRequest
    from plaid.model.accounts_get_request import AccountsGetRequest
    from plaid.model.transfer_authorization_create_request import TransferAuthorizationCreateRequest
    from plaid.model.transfer_create_request import TransferCreateRequest
    from plaid.model.transfer_type import TransferType
    from plaid.model.transfer_network import TransferNetwork
    from plaid.model.ach_class import ACHClass
    from plaid.model.transfer_user_in_request import TransferUserInRequest
    from plaid.model.products import Products
    PLAID_AVAILABLE = True
except ImportError:
    PLAID_AVAILABLE = False
    logger.warning("Plaid SDK not installed. Run: pip install plaid-python")


class PlaidServiceError(Exception):
    """Raised when Plaid operations fail"""
    pass


class PlaidService:
    """
    Real Plaid integration for bank account verification and ACH transfers.
    
    Features:
    - Link Token creation for Plaid Link
    - Instant bank verification (via bank login)
    - Micro-deposit verification (fallback)
    - ACH deposits (user bank -> platform)
    - ACH withdrawals (platform -> user bank)
    """
    
    # Environment mapping
    ENV_MAP = {
        'sandbox': 'sandbox',
        'development': 'development', 
        'production': 'production',
    }
    
    def __init__(self):
        """Initialize Plaid client with environment configuration."""
        self.client_id = os.getenv('PLAID_CLIENT_ID', '')
        self.secret = os.getenv('PLAID_SECRET', '')
        self.env = os.getenv('PLAID_ENV', 'sandbox').lower()
        
        self._available = False
        self._client = None
        
        if not PLAID_AVAILABLE:
            logger.warning("Plaid SDK not available - bank linking disabled")
            return
            
        if not self.client_id or not self.secret:
            logger.warning("Plaid credentials not configured - bank linking disabled")
            logger.info("Set PLAID_CLIENT_ID and PLAID_SECRET in .env")
            return
            
        self._initialize_client()
    
    def _initialize_client(self):
        """Create Plaid API client."""
        try:
            # Map environment to Plaid host
            if self.env == 'production':
                host = plaid.Environment.Production
            elif self.env == 'development':
                host = plaid.Environment.Development
            else:
                host = plaid.Environment.Sandbox
            
            configuration = plaid.Configuration(
                host=host,
                api_key={
                    'clientId': self.client_id,
                    'secret': self.secret,
                }
            )
            
            api_client = plaid.ApiClient(configuration)
            self._client = plaid_api.PlaidApi(api_client)
            self._available = True
            
            logger.info(f"Plaid service initialized (env: {self.env})")
            
        except Exception as e:
            logger.error(f"Failed to initialize Plaid client: {e}")
            self._available = False
    
    @property
    def is_available(self) -> bool:
        """Check if Plaid service is configured and available."""
        return self._available
    
    async def create_link_token(
        self,
        user_id: str,
        client_name: str = "CIFT Markets"
    ) -> dict[str, Any]:
        """
        Create a Link token for Plaid Link initialization.
        
        This token is used by the frontend to open Plaid Link,
        which guides users through connecting their bank account.
        
        Args:
            user_id: Unique user identifier
            client_name: Name shown in Plaid Link
            
        Returns:
            Dict with link_token and expiration
        """
        if not self._available:
            raise PlaidServiceError("Plaid service not available")
        
        try:
            request = LinkTokenCreateRequest(
                user=LinkTokenCreateRequestUser(
                    client_user_id=str(user_id)
                ),
                client_name=client_name,
                products=[Products("auth"), Products("transfer")],
                country_codes=[CountryCode("US")],
                language="en",
            )
            
            response = self._client.link_token_create(request)
            
            return {
                'link_token': response['link_token'],
                'expiration': response['expiration'],
                'request_id': response['request_id'],
            }
            
        except plaid.ApiException as e:
            logger.error(f"Plaid link token error: {e}")
            raise PlaidServiceError(f"Failed to create link token: {e}")
    
    async def exchange_public_token(self, public_token: str) -> dict[str, Any]:
        """
        Exchange a public token for an access token.
        
        Called after user completes Plaid Link flow.
        The access token is used for all subsequent API calls for this bank.
        
        Args:
            public_token: Public token from Plaid Link callback
            
        Returns:
            Dict with access_token and item_id
        """
        if not self._available:
            raise PlaidServiceError("Plaid service not available")
        
        try:
            request = ItemPublicTokenExchangeRequest(
                public_token=public_token
            )
            
            response = self._client.item_public_token_exchange(request)
            
            return {
                'access_token': response['access_token'],
                'item_id': response['item_id'],
            }
            
        except plaid.ApiException as e:
            logger.error(f"Plaid token exchange error: {e}")
            raise PlaidServiceError(f"Failed to exchange token: {e}")
    
    async def get_auth_data(self, access_token: str) -> dict[str, Any]:
        """
        Get bank account and routing numbers.
        
        Used to link bank account to brokerage (Alpaca) for ACH transfers.
        
        Args:
            access_token: Plaid access token for the bank
            
        Returns:
            Dict with account details including routing/account numbers
        """
        if not self._available:
            raise PlaidServiceError("Plaid service not available")
        
        try:
            request = AuthGetRequest(access_token=access_token)
            response = self._client.auth_get(request)
            
            accounts = []
            for account in response['accounts']:
                # Find matching ACH numbers
                ach_numbers = None
                for num in response['numbers']['ach']:
                    if num['account_id'] == account['account_id']:
                        ach_numbers = num
                        break
                
                if ach_numbers:
                    accounts.append({
                        'account_id': account['account_id'],
                        'name': account['name'],
                        'official_name': account.get('official_name'),
                        'type': account['type'],
                        'subtype': account.get('subtype'),
                        'mask': account['mask'],  # Last 4 digits
                        'routing_number': ach_numbers['routing'],
                        'account_number': ach_numbers['account'],
                        'wire_routing': ach_numbers.get('wire_routing'),
                    })
            
            return {
                'item_id': response['item']['item_id'],
                'institution_id': response['item'].get('institution_id'),
                'accounts': accounts,
            }
            
        except plaid.ApiException as e:
            logger.error(f"Plaid auth error: {e}")
            raise PlaidServiceError(f"Failed to get auth data: {e}")
    
    async def initiate_ach_deposit(
        self,
        access_token: str,
        account_id: str,
        amount: Decimal,
        user_name: str,
        description: str = "CIFT Deposit",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """
        Initiate an ACH deposit (pull money from user's bank).
        
        This creates a debit transfer that pulls money from the user's
        linked bank account to the platform's bank account.
        
        Args:
            access_token: Plaid access token
            account_id: Plaid account ID to debit
            amount: Amount to transfer
            user_name: User's legal name (required for ACH)
            description: Transfer description
            idempotency_key: Unique key to prevent duplicates
            
        Returns:
            Dict with transfer details and status
        """
        if not self._available:
            raise PlaidServiceError("Plaid service not available")
        
        try:
            # Step 1: Create transfer authorization (risk check)
            auth_request = TransferAuthorizationCreateRequest(
                access_token=access_token,
                account_id=account_id,
                type=TransferType("debit"),
                network=TransferNetwork("ach"),
                amount=str(amount),
                ach_class=ACHClass("ppd"),  # Personal payment
                user=TransferUserInRequest(
                    legal_name=user_name,
                ),
            )
            
            auth_response = self._client.transfer_authorization_create(auth_request)
            
            authorization = auth_response['authorization']
            if authorization['decision'] != 'approved':
                return {
                    'success': False,
                    'status': 'rejected',
                    'reason': authorization.get('decision_rationale', {}).get('description', 'Transfer not authorized'),
                    'authorization_id': authorization['id'],
                }
            
            # Step 2: Create the actual transfer
            transfer_request = TransferCreateRequest(
                access_token=access_token,
                account_id=account_id,
                authorization_id=authorization['id'],
                amount=str(amount),
                description=description,
            )
            
            if idempotency_key:
                transfer_request.idempotency_key = idempotency_key
            
            transfer_response = self._client.transfer_create(transfer_request)
            transfer = transfer_response['transfer']
            
            return {
                'success': True,
                'transfer_id': transfer['id'],
                'status': transfer['status'],  # 'pending' -> 'posted' -> 'settled'
                'amount': transfer['amount'],
                'created': transfer['created'],
                'authorization_id': authorization['id'],
            }
            
        except plaid.ApiException as e:
            logger.error(f"Plaid deposit error: {e}")
            raise PlaidServiceError(f"Failed to initiate deposit: {e}")
    
    async def initiate_ach_withdrawal(
        self,
        access_token: str,
        account_id: str,
        amount: Decimal,
        user_name: str,
        description: str = "CIFT Withdrawal",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """
        Initiate an ACH withdrawal (push money to user's bank).
        
        This creates a credit transfer that pushes money from the platform's
        bank account to the user's linked bank account.
        
        Args:
            access_token: Plaid access token
            account_id: Plaid account ID to credit
            amount: Amount to transfer
            user_name: User's legal name
            description: Transfer description
            idempotency_key: Unique key to prevent duplicates
            
        Returns:
            Dict with transfer details and status
        """
        if not self._available:
            raise PlaidServiceError("Plaid service not available")
        
        try:
            # Step 1: Create transfer authorization
            auth_request = TransferAuthorizationCreateRequest(
                access_token=access_token,
                account_id=account_id,
                type=TransferType("credit"),  # Push to user's bank
                network=TransferNetwork("ach"),
                amount=str(amount),
                ach_class=ACHClass("ppd"),
                user=TransferUserInRequest(
                    legal_name=user_name,
                ),
            )
            
            auth_response = self._client.transfer_authorization_create(auth_request)
            
            authorization = auth_response['authorization']
            if authorization['decision'] != 'approved':
                return {
                    'success': False,
                    'status': 'rejected',
                    'reason': authorization.get('decision_rationale', {}).get('description', 'Transfer not authorized'),
                }
            
            # Step 2: Create the transfer
            transfer_request = TransferCreateRequest(
                access_token=access_token,
                account_id=account_id,
                authorization_id=authorization['id'],
                amount=str(amount),
                description=description,
            )
            
            if idempotency_key:
                transfer_request.idempotency_key = idempotency_key
            
            transfer_response = self._client.transfer_create(transfer_request)
            transfer = transfer_response['transfer']
            
            return {
                'success': True,
                'transfer_id': transfer['id'],
                'status': transfer['status'],
                'amount': transfer['amount'],
                'created': transfer['created'],
            }
            
        except plaid.ApiException as e:
            logger.error(f"Plaid withdrawal error: {e}")
            raise PlaidServiceError(f"Failed to initiate withdrawal: {e}")


# Global instance
plaid_service = PlaidService()
