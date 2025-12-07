# Real Money Onboarding & Alpaca Integration

## Overview
To support real money deposits and withdrawals, we have implemented a **Brokerage as a Service** model using Alpaca.

## Key Concepts
1.  **No Separate Login**: Users do *not* need to create an account at Alpaca.com.
2.  **Sub-Accounts**: CIFT creates a sub-account for each user under our Master Broker Account.
3.  **KYC Requirement**: To create this sub-account, we must collect and submit the user's Identity Data (Name, Address, DOB, SSN).

## Implementation Details

### 1. Database Updates
- Added `alpaca_account_id` to `users` table.
- Added `kyc_status` to `users` table.

### 2. New API Endpoints
- `POST /api/v1/onboarding/submit`: Submits the user's KYC profile to Alpaca to create the brokerage account.
    - **Input**: Uses data from `kyc_profiles` (First Name, Last Name, Address, DOB, SSN).
    - **Output**: Returns `account_id` and `status` (e.g., `ACTIVE`, `PENDING`).
    - **Action**: Calls `AlpacaProcessor.create_brokerage_account`.

### 3. Payment Processor Updates
- **AlpacaProcessor**: Added `create_account` method to handle the `POST /v1/accounts` call to Alpaca.
- **PaymentProcessor (Facade)**: Added `create_brokerage_account` to route the request.

## User Flow
1.  **User Registration**: User signs up (Email/Password).
2.  **KYC Profile**: User fills out profile (Address, DOB, SSN) via Frontend.
3.  **Submission**: User clicks "Submit Application".
    - Frontend calls `POST /api/v1/onboarding/submit`.
    - Backend creates Alpaca Account.
    - `alpaca_account_id` is saved to `users` table.
4.  **Funding**:
    - User links Bank Account (`POST /api/v1/funding/payment-methods`).
    - Backend uses `alpaca_account_id` to create ACH Relationship.
    - User deposits funds (`POST /api/v1/funding/deposit`).

## Next Steps
- **Frontend**: Ensure the KYC form calls the new `/submit` endpoint.
- **Webhooks**: Listen for `account.updated` events from Alpaca to handle KYC approval/rejection asynchronously.
