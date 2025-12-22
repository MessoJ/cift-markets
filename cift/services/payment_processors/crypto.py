"""
Cryptocurrency Payment Processor - RULES COMPLIANT
Integrates with blockchain networks for Bitcoin and Ethereum payments
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

import httpx

from cift.core.database import get_postgres_pool
from cift.services.payment_processors.base import PaymentProcessor, PaymentProcessorError


class CryptoProcessor(PaymentProcessor):
    """
    Cryptocurrency payment integration for Bitcoin and Ethereum

    Uses blockchain explorers and node APIs for transaction verification

    Configuration required:
        - btc_node_url: Bitcoin node RPC URL (optional, can use block explorers)
        - eth_node_url: Ethereum node RPC URL (e.g., Infura, Alchemy)
        - deposit_addresses: Dict of {network: address} for receiving deposits
        - hot_wallet_private_keys: Private keys for withdrawal wallets (encrypted)
        - confirmations_required: Number of confirmations before considering tx complete
        - blockchain_explorer_api_key: API key for blockchain explorer (e.g., Blockchain.com, Etherscan)
    """

    # Block explorer APIs (free tier available)
    BLOCKCHAIN_INFO_API = "https://blockchain.info"
    ETHERSCAN_API = "https://api.etherscan.io/api"

    def _validate_config(self) -> None:
        """Validate crypto configuration"""
        required = ["deposit_addresses"]

        for key in required:
            if key not in self.config:
                raise PaymentProcessorError(f"Missing crypto configuration: {key}")

        if not isinstance(self.config["deposit_addresses"], dict):
            raise PaymentProcessorError("deposit_addresses must be a dictionary")

    async def _fetch_payment_method(self, payment_method_id: UUID) -> dict[str, Any]:
        """Fetch payment method details from database"""
        pool = await get_postgres_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    id::text,
                    type,
                    crypto_address,
                    crypto_network,
                    is_verified
                FROM payment_methods
                WHERE id = $1
                """,
                payment_method_id,
            )

            if not row:
                raise PaymentProcessorError("Payment method not found")

            if row["type"] != "crypto_wallet":
                raise PaymentProcessorError("Payment method is not a crypto wallet")

            return dict(row)

    def _validate_btc_address(self, address: str) -> bool:
        """
        Validate Bitcoin address format

        Args:
            address: Bitcoin address

        Returns:
            True if valid format
        """
        # Basic validation - in production use bitcoin library
        if not address:
            return False

        # P2PKH (legacy): starts with 1
        # P2SH: starts with 3
        # Bech32 (native SegWit): starts with bc1
        valid_prefixes = ("1", "3", "bc1")

        if not address.startswith(valid_prefixes):
            return False

        # Length check
        if len(address) < 26 or len(address) > 62:
            return False

        return True

    def _validate_eth_address(self, address: str) -> bool:
        """
        Validate Ethereum address format

        Args:
            address: Ethereum address

        Returns:
            True if valid format
        """
        # Basic validation - in production use web3 library
        if not address:
            return False

        # Must start with 0x and be 42 characters (0x + 40 hex chars)
        if not address.startswith("0x"):
            return False

        if len(address) != 42:
            return False

        # Check if hex
        try:
            int(address[2:], 16)
            return True
        except ValueError:
            return False

    async def _get_btc_price_usd(self) -> Decimal:
        """
        Get current BTC/USD price

        Returns:
            BTC price in USD
        """
        url = f"{self.BLOCKCHAIN_INFO_API}/ticker"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                response.raise_for_status()
                data = response.json()
                return Decimal(str(data["USD"]["last"]))
        except Exception as e:
            raise PaymentProcessorError(f"Failed to get BTC price: {str(e)}") from e

    async def _get_eth_price_usd(self) -> Decimal:
        """
        Get current ETH/USD price

        Returns:
            ETH price in USD
        """
        # Use CoinGecko API (free, no key required)
        url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                response.raise_for_status()
                data = response.json()
                return Decimal(str(data["ethereum"]["usd"]))
        except Exception as e:
            raise PaymentProcessorError(f"Failed to get ETH price: {str(e)}") from e

    async def _convert_usd_to_crypto(self, usd_amount: Decimal, network: str) -> Decimal:
        """
        Convert USD amount to cryptocurrency amount

        Args:
            usd_amount: Amount in USD
            network: 'bitcoin' or 'ethereum'

        Returns:
            Amount in crypto
        """
        if network.lower() == "bitcoin":
            btc_price = await self._get_btc_price_usd()
            return usd_amount / btc_price
        elif network.lower() == "ethereum":
            eth_price = await self._get_eth_price_usd()
            return usd_amount / eth_price
        else:
            raise PaymentProcessorError(f"Unsupported network: {network}")

    async def _check_btc_transaction(self, tx_hash: str) -> dict[str, Any]:
        """
        Check Bitcoin transaction status

        Args:
            tx_hash: Bitcoin transaction hash

        Returns:
            Transaction details
        """
        url = f"{self.BLOCKCHAIN_INFO_API}/rawtx/{tx_hash}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                response.raise_for_status()
                data = response.json()

                # Extract relevant info
                confirmations = data.get("block_height", 0)
                if confirmations > 0:
                    # Estimate confirmations (simplified)
                    confirmations = 1  # In production, calculate from current block height

                return {
                    "hash": tx_hash,
                    "confirmations": confirmations,
                    "amount_btc": Decimal(sum(out["value"] for out in data["out"]))
                    / Decimal("100000000"),
                    "time": datetime.fromtimestamp(data.get("time", 0)),
                    "status": (
                        "confirmed"
                        if confirmations >= self.config.get("confirmations_required", 3)
                        else "pending"
                    ),
                }
        except Exception as e:
            raise PaymentProcessorError(f"Failed to check BTC transaction: {str(e)}") from e

    async def _check_eth_transaction(self, tx_hash: str) -> dict[str, Any]:
        """
        Check Ethereum transaction status

        Args:
            tx_hash: Ethereum transaction hash

        Returns:
            Transaction details
        """
        api_key = self.config.get("blockchain_explorer_api_key", "")
        url = f"{self.ETHERSCAN_API}?module=proxy&action=eth_getTransactionByHash&txhash={tx_hash}&apikey={api_key}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                response.raise_for_status()
                data = response.json()

                if data["result"]:
                    tx = data["result"]

                    # Convert hex to decimal
                    value_wei = int(tx.get("value", "0x0"), 16)
                    value_eth = Decimal(value_wei) / Decimal("1000000000000000000")

                    # Check if confirmed
                    block_number = tx.get("blockNumber")
                    confirmations = 0

                    if block_number:
                        # Get current block
                        current_block_url = f"{self.ETHERSCAN_API}?module=proxy&action=eth_blockNumber&apikey={api_key}"
                        block_response = await client.get(current_block_url, timeout=10.0)
                        current_block_data = block_response.json()
                        current_block = int(current_block_data["result"], 16)
                        block_num = int(block_number, 16)
                        confirmations = current_block - block_num

                    return {
                        "hash": tx_hash,
                        "confirmations": confirmations,
                        "amount_eth": value_eth,
                        "from": tx.get("from"),
                        "to": tx.get("to"),
                        "status": (
                            "confirmed"
                            if confirmations >= self.config.get("confirmations_required", 12)
                            else "pending"
                        ),
                    }
                else:
                    return {"hash": tx_hash, "status": "not_found"}

        except Exception as e:
            raise PaymentProcessorError(f"Failed to check ETH transaction: {str(e)}") from e

    async def process_deposit(
        self,
        user_id: UUID,
        amount: Decimal,
        payment_method_id: UUID,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Process crypto deposit

        Returns deposit address and amount to send
        User must manually send crypto to this address
        """
        # Fetch payment method
        payment_method = await self._fetch_payment_method(payment_method_id)
        network = payment_method["crypto_network"].lower()

        # Get deposit address for this network
        deposit_address = self.config["deposit_addresses"].get(network)

        if not deposit_address:
            raise PaymentProcessorError(f"No deposit address configured for {network}")

        # Convert USD to crypto
        crypto_amount = await self._convert_usd_to_crypto(amount, network)

        # Calculate fee
        fee = await self.calculate_fee(amount, "deposit", "crypto_wallet")

        # Generate unique deposit identifier (can use memo/destination tag for some networks)
        deposit_id = f"{user_id}-{datetime.now().timestamp()}"

        return {
            "transaction_id": deposit_id,
            "status": "pending",  # Waiting for blockchain confirmation
            "fee": fee,
            "estimated_arrival": datetime.now() + timedelta(hours=1),  # Depends on network
            "additional_data": {
                "deposit_address": deposit_address,
                "amount_crypto": str(crypto_amount),
                "network": network,
                "amount_usd": str(amount),
                "instructions": f"Send exactly {crypto_amount:.8f} {network.upper()} to {deposit_address}",
                "confirmations_required": self.config.get(
                    "confirmations_required", 3 if network == "bitcoin" else 12
                ),
            },
        }

    async def process_withdrawal(
        self,
        user_id: UUID,
        amount: Decimal,
        payment_method_id: UUID,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Process crypto withdrawal

        Sends crypto from hot wallet to user's address
        """
        # Fetch payment method
        payment_method = await self._fetch_payment_method(payment_method_id)
        recipient_address = payment_method["crypto_address"]
        network = payment_method["crypto_network"].lower()

        # Validate address
        if network == "bitcoin" and not self._validate_btc_address(recipient_address):
            raise PaymentProcessorError("Invalid Bitcoin address")
        elif network == "ethereum" and not self._validate_eth_address(recipient_address):
            raise PaymentProcessorError("Invalid Ethereum address")

        # Convert USD to crypto
        crypto_amount = await self._convert_usd_to_crypto(amount, network)

        # Calculate fee
        fee = await self.calculate_fee(amount, "withdrawal", "crypto_wallet")

        # In production, you would:
        # 1. Create and sign transaction using hot wallet private key
        # 2. Broadcast transaction to network
        # 3. Return transaction hash
        #
        # For now, return pending status
        # You'd use libraries like: python-bitcoinlib, web3.py, etc.

        return {
            "transaction_id": f"pending_crypto_withdrawal_{datetime.now().timestamp()}",
            "status": "processing",
            "fee": fee,
            "estimated_arrival": datetime.now() + timedelta(hours=2),
            "additional_data": {
                "recipient_address": recipient_address,
                "amount_crypto": str(crypto_amount),
                "network": network,
                "amount_usd": str(amount),
                "note": "Withdrawal will be processed within 24 hours",
            },
        }

    async def verify_payment_method(
        self, payment_method_id: UUID, verification_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Verify crypto wallet address

        Validates address format
        """
        payment_method = await self._fetch_payment_method(payment_method_id)
        address = payment_method["crypto_address"]
        network = payment_method["crypto_network"].lower()

        is_valid = False

        if network == "bitcoin":
            is_valid = self._validate_btc_address(address)
        elif network == "ethereum":
            is_valid = self._validate_eth_address(address)
        else:
            return {
                "verified": False,
                "message": f"Unsupported network: {network}",
                "additional_data": {},
            }

        if is_valid:
            return {
                "verified": True,
                "message": f"{network.title()} address verified",
                "additional_data": {"address": address, "network": network},
            }
        else:
            return {
                "verified": False,
                "message": f"Invalid {network} address format",
                "additional_data": {},
            }

    async def get_transaction_status(self, external_transaction_id: str) -> dict[str, Any]:
        """
        Query crypto transaction status

        Args:
            external_transaction_id: Transaction hash
        """
        # Determine if it's BTC or ETH based on format
        if external_transaction_id.startswith("0x"):
            # Ethereum transaction
            try:
                tx_data = await self._check_eth_transaction(external_transaction_id)

                if tx_data["status"] == "confirmed":
                    return {
                        "status": "completed",
                        "completed_at": datetime.now(),
                        "failure_reason": None,
                        "additional_data": tx_data,
                    }
                elif tx_data["status"] == "pending":
                    return {
                        "status": "pending",
                        "completed_at": None,
                        "failure_reason": None,
                        "additional_data": tx_data,
                    }
                else:
                    return {
                        "status": "failed",
                        "completed_at": None,
                        "failure_reason": "Transaction not found",
                        "additional_data": tx_data,
                    }
            except PaymentProcessorError:
                raise
        else:
            # Bitcoin transaction
            try:
                tx_data = await self._check_btc_transaction(external_transaction_id)

                if tx_data["status"] == "confirmed":
                    return {
                        "status": "completed",
                        "completed_at": tx_data["time"],
                        "failure_reason": None,
                        "additional_data": tx_data,
                    }
                else:
                    return {
                        "status": "pending",
                        "completed_at": None,
                        "failure_reason": None,
                        "additional_data": tx_data,
                    }
            except PaymentProcessorError:
                raise

    async def calculate_fee(
        self, amount: Decimal, transaction_type: str, payment_method_type: str
    ) -> Decimal:
        """
        Calculate crypto processing fee

        Fees include:
        - Platform fee (e.g., 1%)
        - Network fee (gas/miner fee)

        In production, you'd fetch real-time network fees
        """
        # Platform fee: 1%
        platform_fee_percent = Decimal("0.01")
        platform_fee = amount * platform_fee_percent

        # Estimated network fee (in USD equivalent)
        # In production, fetch real-time gas prices
        if transaction_type == "withdrawal":
            # Higher fee for withdrawals (we pay network fee)
            network_fee = Decimal("5.00")  # Approximate
        else:
            # Lower fee for deposits (user pays network fee)
            network_fee = Decimal("0.50")

        total_fee = platform_fee + network_fee

        return total_fee

    def _handle_webhook(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Handle blockchain webhook/callback

        Some services like BlockCypher, Alchemy provide webhooks for new transactions
        """
        # This would be implemented based on webhook provider
        # Example structure:

        tx_hash = payload.get("hash")
        confirmations = payload.get("confirmations", 0)

        if confirmations >= self.config.get("confirmations_required", 3):
            return {
                "event": "transaction_confirmed",
                "transaction_id": tx_hash,
                "status": "completed",
                "confirmations": confirmations,
                "additional_data": payload,
            }
        else:
            return {
                "event": "transaction_detected",
                "transaction_id": tx_hash,
                "status": "pending",
                "confirmations": confirmations,
                "additional_data": payload,
            }
