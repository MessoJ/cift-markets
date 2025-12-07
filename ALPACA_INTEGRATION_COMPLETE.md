# Alpaca Integration Complete

## Overview
The backend has been updated to support "Real Money" transactions via Alpaca for bank transfers (ACH). This implementation is end-to-end, covering configuration, service logic, and API integration.

## Components Implemented

### 1. Alpaca Processor (`cift/services/payment_processors/alpaca_processor.py`)
- Implements `PaymentProcessor` interface.
- Handles `process_deposit` for ACH transfers.
- Implements `link_account` to create ACH relationships via Alpaca API.
- Supports both Sandbox (Paper) and Production environments.

### 2. Configuration (`cift/services/payment_config.py`)
- Added `get_alpaca_config` to load `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`.
- Registered `bank_account` type to use Alpaca configuration.
- Updated `is_payment_type_configured` to check for Alpaca credentials.

### 3. Service Layer (`cift/services/payment_processor.py`)
- Updated `create_bank_transfer` to use the configured `bank_account` processor (Alpaca) instead of simulation.
- Added `link_external_account` method to the facade to support linking bank accounts.

### 4. API Layer (`cift/api/routes/funding.py`)
- Updated `add_payment_method` to automatically call `link_external_account` when a user adds a bank account.
- If Alpaca returns `APPROVED` immediately (e.g. in simulation or specific flows), the payment method is auto-verified.

## Environment Variables Required
To enable real money handling, set the following in your `.env` file:

```env
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # or https://api.alpaca.markets for production
```

## Security
- API keys are loaded from environment variables.
- Secrets are never hardcoded.
- HTTPS is enforced for all API calls.

## Next Steps
- Obtain production keys from Alpaca.
- Set up webhooks to handle asynchronous ACH status updates (e.g. `returned`, `settled`).
