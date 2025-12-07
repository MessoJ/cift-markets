# Funding Page - Future Enhancements & Features

To further elevate the Funding section to an "Industry Standard" or "Bloomberg-level" experience, consider implementing the following features and UI/UX improvements.

## 1. Advanced Features

### 🔄 Recurring Deposits (Auto-Invest)
- **Feature**: Allow users to set up automatic weekly/monthly deposits.
- **UI**: A "Schedule" toggle in the Deposit tab.
- **Value**: Increases user retention and assets under management (AUM).

### 💱 Multi-Currency Support
- **Feature**: Allow holding balances in EUR, GBP, JPY, etc.
- **UI**: A currency selector in the header summary cards.
- **Value**: Essential for international markets or forex trading.

### 🏦 Wire Instructions Modal
- **Feature**: A dedicated view for "Large Transfers" (> $50k) that displays:
    - Beneficiary Name/Address
    - Bank Name/Address
    - SWIFT/BIC & IBAN
    - Specific "Memo" field instructions (critical for tracking).
- **UI**: A "Printable PDF" button for these instructions.

### 🔒 2FA for Withdrawals
- **Feature**: Require a Time-based One-Time Password (TOTP) or SMS code before finalizing a withdrawal.
- **UI**: A modal popup after clicking "Withdraw" asking for the 6-digit code.
- **Value**: Critical security trust signal.

### 📄 Tax Documents Center
- **Feature**: A section to download 1099-B, 1099-DIV, or monthly statements.
- **UI**: A small link or sub-tab in "History" for "Documents".

## 2. UI/UX Improvements

### ⚡ Real-Time Status Updates
- **Tech**: Use WebSockets to update transaction status from "Processing" to "Completed" without refreshing.
- **UI**: A toast notification: *"Your deposit of $5,000 has cleared and is ready for trading."*

### 🔗 Plaid / Stripe Integration
- **Feature**: Replace the manual "Add Payment Method" form with a real bank linker (Plaid Link).
- **UI**: The familiar "Select your bank" modal.

### 📊 Funding Analytics
- **Feature**: A small chart in the History tab showing "Net Deposits over Time".
- **UI**: A sparkline or bar chart above the transaction list.

### 📱 Mobile Optimization
- **Improvement**: Ensure the "Credit Card" visuals in `PaymentMethodsTab` stack beautifully on mobile screens.
- **Action**: Verify touch targets for the "Quick Amount" buttons are at least 44px.

## 3. Immediate "Low Hanging Fruit"

- **"Copy" Buttons**: Add small copy icons next to Transaction IDs and Reference Numbers.
- **Tooltips**: Add hover tooltips to "Buying Power" explaining why it might differ from "Cash" (e.g., margin leverage).
- **Cancel Button**: Allow users to cancel a "Pending" deposit if it hasn't been processed yet.
