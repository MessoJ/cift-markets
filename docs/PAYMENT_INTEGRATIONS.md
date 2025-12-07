# Payment Integrations - Production Ready

## Overview

CIFT Markets now includes comprehensive, production-ready payment processing integrations for multiple payment methods:

- **Credit/Debit Cards** - Stripe integration with 3DS support
- **M-Pesa** - Safaricom Daraja API (STK Push & B2C)
- **PayPal** - REST API v2 (Orders & Payouts)
- **Cryptocurrency** - Bitcoin & Ethereum support

## Architecture

### Payment Processor Framework

All payment integrations follow a unified architecture using the Abstract Base Class pattern:

```
cift/services/payment_processors/
├── base.py              # Abstract base class
├── stripe_processor.py  # Card payments
├── mpesa.py            # M-Pesa (Kenya, Tanzania, Uganda, Rwanda)
├── paypal.py           # PayPal payments
├── crypto.py           # Bitcoin & Ethereum
└── __init__.py         # Factory function
```

### Key Components

1. **`PaymentProcessor` (base.py)** - Abstract base class defining the interface
2. **`get_payment_processor()`** - Factory function to get appropriate processor
3. **`PaymentConfig`** - Centralized configuration management
4. **`payment_processor`** - Global facade instance for backward compatibility

## Configuration

### Environment Variables

Copy `.env.example.payments` to `.env` and configure your credentials:

```bash
# M-Pesa (Daraja API)
MPESA_CONSUMER_KEY=your_key
MPESA_CONSUMER_SECRET=your_secret
MPESA_BUSINESS_SHORT_CODE=174379
MPESA_PASSKEY=your_passkey
MPESA_ENVIRONMENT=sandbox  # or 'production'
MPESA_CALLBACK_URL=https://yourdomain.com/api/v1/webhooks/mpesa

# Stripe
STRIPE_SECRET_KEY=sk_test_...  # or sk_live_...
STRIPE_PUBLISHABLE_KEY=pk_test_...  # or pk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...

# PayPal
PAYPAL_CLIENT_ID=your_client_id
PAYPAL_CLIENT_SECRET=your_secret
PAYPAL_ENVIRONMENT=sandbox  # or 'production'

# Cryptocurrency
CRYPTO_BTC_DEPOSIT_ADDRESS=your_btc_address
CRYPTO_ETH_DEPOSIT_ADDRESS=0x_your_eth_address
CRYPTO_CONFIRMATIONS_REQUIRED=3
ETHERSCAN_API_KEY=your_api_key
```

### Getting API Credentials

#### M-Pesa (Safaricom Daraja)
1. Visit: https://developer.safaricom.co.ke
2. Create an app in the sandbox or production environment
3. Get Consumer Key, Consumer Secret, and Passkey
4. Configure your Paybill/Till number

#### Stripe
1. Visit: https://dashboard.stripe.com/apikeys
2. Get Secret Key and Publishable Key
3. Create webhook endpoint for async notifications
4. Test in sandbox mode first (keys start with `sk_test_`)

#### PayPal
1. Visit: https://developer.paypal.com/dashboard/
2. Create REST API app
3. Get Client ID and Client Secret
4. Test in sandbox environment first

#### Cryptocurrency
1. **Bitcoin**: Generate a new address from your wallet
2. **Ethereum**: Create address (e.g., via MetaMask, Infura, Alchemy)
3. **Explorer APIs**: Get API key from Etherscan.io for transaction verification
4. **Node Access**: Optional - use Infura or Alchemy for direct blockchain access

## Usage

### Basic Example

```python
from cift.services.payment_processors import get_payment_processor
from cift.services.payment_config import PaymentConfig

# Get configuration for payment type
config = PaymentConfig.get_mpesa_config()

# Get processor instance
processor = get_payment_processor('mpesa', config)

# Process deposit
result = await processor.process_deposit(
    user_id=user_uuid,
    amount=Decimal('100.00'),
    payment_method_id=payment_method_uuid,
    metadata={'order_id': 'ORDER123'}
)

# Result contains:
# - transaction_id: External processor transaction ID
# - status: 'pending', 'processing', 'completed', 'failed'
# - fee: Processing fee amount
# - estimated_arrival: Expected completion datetime
# - additional_data: Processor-specific data
```

### Using the Facade

For backward compatibility, use the global `payment_processor` instance:

```python
from cift.services.payment_processor import payment_processor

# Calculate fee
fee = payment_processor.calculate_fee(
    amount=Decimal('100.00'),
    payment_method_type='mpesa',
    transfer_type='standard'
)

# Process withdrawal
result = await payment_processor.process_withdrawal(
    amount=Decimal('50.00'),
    payment_method_type='paypal',
    payment_method_id='pm_uuid',
    metadata={'user_id': 'user_uuid'}
)
```

## Payment Methods

### 1. M-Pesa Integration

**Features:**
- STK Push (Lipa Na M-Pesa Online) for deposits
- B2C (Business to Customer) for withdrawals
- Real-time transaction status checking
- Webhook/callback support
- Multi-country support (KE, TZ, UG, RW)

**Flow:**
1. User initiates deposit → STK Push sent to phone
2. User enters M-Pesa PIN on phone
3. Callback received → Transaction completed
4. Funds credited to account

**Test Phone Numbers (Sandbox):**
- Kenya: 254708374149
- Test amounts: Any amount

### 2. Stripe Card Payments

**Features:**
- Payment Intents API with SCA (Strong Customer Authentication)
- 3D Secure (3DS) support
- SetupIntent for saving cards without charging
- Automatic card brand detection
- Refund support

**Flow:**
1. Create SetupIntent for new card (or use saved card)
2. Frontend collects card details using Stripe Elements
3. Backend creates PaymentIntent with card
4. 3DS challenge if required
5. Payment completed → Funds credited

**Test Cards:**
- Success: 4242 4242 4242 4242
- 3DS Required: 4000 0025 0000 3155
- Declined: 4000 0000 0000 0002

### 3. PayPal Integration

**Features:**
- Orders API for deposits (user redirected to PayPal)
- Payouts API for withdrawals
- Webhook notifications
- Refund support

**Flow - Deposit:**
1. Create Order → Get approval URL
2. Redirect user to PayPal
3. User approves payment
4. Capture Order → Funds credited

**Flow - Withdrawal:**
1. Create Payout batch
2. PayPal processes within 1 hour
3. User receives funds in PayPal account

### 4. Cryptocurrency

**Features:**
- Bitcoin support
- Ethereum support
- Address validation
- Blockchain transaction verification
- Configurable confirmation requirements

**Flow - Deposit:**
1. Generate deposit address
2. User sends crypto to address
3. Monitor blockchain for confirmations
4. Credit account after N confirmations

**Flow - Withdrawal:**
1. Create withdrawal transaction
2. Sign and broadcast to network
3. Monitor for confirmations
4. Mark complete

**Confirmation Requirements:**
- Bitcoin: 3-6 confirmations (~30-60 minutes)
- Ethereum: 12-35 confirmations (~3-9 minutes)

## Fee Structure

### Default Fees

| Payment Method | Deposit Fee | Withdrawal Fee |
|---------------|-------------|----------------|
| Bank Account (ACH) | $0.00 | $0.00 |
| Credit/Debit Card | 2.9% + $0.30 | Not supported* |
| PayPal | 2.99% + $0.49 | $0.25 |
| M-Pesa | 2.5% | 2.5% |
| Cryptocurrency | $0.50 + network fee | $5.00 + network fee |

*Card withdrawals typically require bank transfers

### Adjusting Fees

Fees are calculated by each processor's `calculate_fee()` method. To adjust:

1. Update processor's fee calculation logic
2. Or modify `PaymentProcessor.calculate_fee()` in `payment_processor.py`

## Security Best Practices

### API Key Management

1. **Never commit API keys to version control**
2. Use environment variables for all credentials
3. Rotate keys regularly
4. Use sandbox keys for development/testing
5. Restrict API key permissions (e.g., read-only for analytics)

### Webhook Security

1. **Verify webhook signatures:**
   - Stripe: Verify using webhook secret
   - PayPal: Verify using webhook ID
   - M-Pesa: Validate source IP

2. **Implement idempotency:**
   - Store processed webhook IDs
   - Prevent duplicate processing

### PCI Compliance

1. **Never store full card numbers**
   - Use Stripe's tokenization
   - Store only last 4 digits

2. **Use Stripe Elements for card collection**
   - Card data never touches your servers
   - Automatic PCI compliance

### Crypto Security

1. **Hot Wallet Security:**
   - Encrypt private keys at rest
   - Use environment variables (encrypted)
   - Consider hardware security modules (HSMs)

2. **Cold Storage:**
   - Keep majority of funds in cold wallets
   - Only maintain operational amounts in hot wallets

## Webhook Endpoints

### M-Pesa Callback

```
POST /api/v1/webhooks/mpesa
```

Receives STK Push and B2C callbacks.

### Stripe Webhooks

```
POST /api/v1/webhooks/stripe
```

Events:
- `payment_intent.succeeded`
- `payment_intent.payment_failed`
- `charge.refunded`

### PayPal Webhooks

```
POST /api/v1/webhooks/paypal
```

Events:
- `PAYMENT.CAPTURE.COMPLETED`
- `PAYMENT.CAPTURE.DENIED`
- `CHECKOUT.ORDER.APPROVED`
- `PAYMENT.PAYOUTS-ITEM.SUCCEEDED`

## Error Handling

All processors raise `PaymentProcessorError` for failures:

```python
from cift.services.payment_processors import PaymentProcessorError

try:
    result = await processor.process_deposit(...)
except PaymentProcessorError as e:
    # Handle error
    logger.error(f"Payment failed: {str(e)}")
    # Return user-friendly error message
```

## Testing

### Unit Tests

```bash
# Run payment processor tests
pytest tests/services/test_payment_processors.py -v
```

### Integration Tests

1. **Use sandbox/test environments**
2. **Test all payment flows:**
   - Successful deposits
   - Failed deposits
   - Successful withdrawals
   - Failed withdrawals
   - Refunds
   - Webhooks

3. **Test error scenarios:**
   - Insufficient funds
   - Invalid payment methods
   - Network timeouts
   - Invalid API keys

### Manual Testing Checklist

- [ ] Add payment method for each type
- [ ] Initiate deposit for each method
- [ ] Complete payment flow
- [ ] Verify funds credited correctly
- [ ] Initiate withdrawal for each method
- [ ] Verify funds deducted and sent
- [ ] Test receipt download
- [ ] Test payment method verification
- [ ] Test error handling (invalid amounts, etc.)

## Monitoring

### Key Metrics

1. **Success Rates:**
   - Deposit success rate by payment method
   - Withdrawal success rate by payment method

2. **Processing Times:**
   - Average time to complete deposit
   - Average time to complete withdrawal

3. **Fees Collected:**
   - Total fees by payment method
   - Fee percentage of volume

4. **Errors:**
   - Failed transactions by reason
   - Payment processor downtime

### Logging

All payment operations are logged with structured logging:

```python
logger.info("M-Pesa deposit initiated", extra={
    'user_id': str(user_id),
    'amount': float(amount),
    'payment_method_id': str(payment_method_id),
    'transaction_id': external_id
})
```

## Troubleshooting

### Common Issues

#### M-Pesa

**Issue:** STK Push not received on phone
- Verify phone number format (254XXXXXXXXX)
- Check if phone is M-Pesa registered
- Ensure sufficient balance for test

**Issue:** "Invalid Access Token"
- Check consumer key/secret
- Verify environment (sandbox vs production)
- Regenerate credentials if needed

#### Stripe

**Issue:** "Payment failed - card declined"
- User's card may have insufficient funds
- Card may be restricted for online payments
- Try different test card

**Issue:** "No such payment_method"
- Payment method may have been detached
- Re-add payment method

#### PayPal

**Issue:** "Order not approved"
- User cancelled at PayPal
- PayPal account may have insufficient funds
- Try sandbox account

#### Cryptocurrency

**Issue:** "Transaction not found"
- Wait for blockchain propagation (~10 seconds)
- Verify transaction hash is correct
- Check if transaction was actually broadcast

**Issue:** "Insufficient confirmations"
- Wait for more blocks
- Reduce confirmation requirements for testing

## Production Checklist

Before going live:

- [ ] Switch all APIs to production mode
- [ ] Update all API keys to production keys
- [ ] Configure production webhook URLs (HTTPS required)
- [ ] Set up monitoring and alerting
- [ ] Test with real payment methods (small amounts)
- [ ] Implement rate limiting
- [ ] Set up backup/failover for hot wallets (crypto)
- [ ] Configure proper error notifications
- [ ] Review and approve fee schedule
- [ ] Verify PCI compliance (for cards)
- [ ] Set up fraud detection rules
- [ ] Document runbook for common issues
- [ ] Train support team on payment flows

## Support

For issues specific to payment processors:

- **M-Pesa:** https://developer.safaricom.co.ke/support
- **Stripe:** https://support.stripe.com
- **PayPal:** https://developer.paypal.com/support
- **Blockchain:** Check respective blockchain explorer forums

For CIFT Markets payment integration issues, contact the development team.

## Future Enhancements

Potential additions:

1. **Additional Payment Methods:**
   - Cash App
   - Venmo
   - Apple Pay / Google Pay
   - Bank transfers (ACH, SEPA, FPS)

2. **Features:**
   - Recurring payments / subscriptions
   - Split payments
   - Multi-currency support
   - Payment plans / installments
   - Gift cards / vouchers

3. **Optimization:**
   - Smart routing (choose cheapest processor)
   - Retry logic for failed payments
   - Batch processing for withdrawals
   - Payment method recommendations

## License

Proprietary - CIFT Markets

---

**Last Updated:** 2025-11-15  
**Version:** 1.0.0
