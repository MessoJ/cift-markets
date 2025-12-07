import { createSignal, Show, For } from 'solid-js';
import { X, CheckCircle, AlertCircle } from 'lucide-solid';
import { apiClient } from '../../../lib/api/client';
import { PaymentMethodLogo } from '../../../components/PaymentMethodLogo';

interface AddPaymentMethodModalProps {
  onClose: () => void;
  onSuccess: () => void;
}

type PaymentType = 'bank_account' | 'debit_card' | 'credit_card' | 'paypal' | 'mpesa' | 'crypto_wallet' | 'cashapp';
type SubmitState = 'idle' | 'submitting' | 'success' | 'error';

export function AddPaymentMethodModal(props: AddPaymentMethodModalProps) {
  const [selectedType, setSelectedType] = createSignal<PaymentType | null>(null);
  const [error, setError] = createSignal<string | null>(null);
  const [submitState, setSubmitState] = createSignal<SubmitState>('idle');

  // Bank Account Fields
  const [bankName, setBankName] = createSignal('');
  const [accountType, setAccountType] = createSignal<'checking' | 'savings'>('checking');
  const [accountNumber, setAccountNumber] = createSignal('');
  const [routingNumber, setRoutingNumber] = createSignal('');

  // Card Fields
  const [cardNumber, setCardNumber] = createSignal('');
  const [cardBrand, setCardBrand] = createSignal('');
  const [cardExpMonth, setCardExpMonth] = createSignal('');
  const [cardExpYear, setCardExpYear] = createSignal('');
  const [cardCVV, setCardCVV] = createSignal('');

  // PayPal Fields
  const [paypalEmail, setPaypalEmail] = createSignal('');

  // M-Pesa Fields
  const [mpesaPhone, setMpesaPhone] = createSignal('');
  const [mpesaCountry, setMpesaCountry] = createSignal('KE');

  // Crypto Fields
  const [cryptoAddress, setCryptoAddress] = createSignal('');
  const [cryptoNetwork, setCryptoNetwork] = createSignal('bitcoin');

  // Cash App Fields
  const [cashappTag, setCashappTag] = createSignal('');

  const paymentTypes = [
    { value: 'bank_account' as PaymentType, label: 'Bank Account', logo: 'bank' as const, color: 'primary' },
    { value: 'debit_card' as PaymentType, label: 'Debit Card', logo: 'visa' as const, color: 'accent' },
    { value: 'credit_card' as PaymentType, label: 'Credit Card', logo: 'mastercard' as const, color: 'success' },
    { value: 'paypal' as PaymentType, label: 'PayPal', logo: 'paypal' as const, color: 'info' },
    { value: 'cashapp' as PaymentType, label: 'Cash App', logo: 'cashapp' as const, color: 'success' },
    { value: 'mpesa' as PaymentType, label: 'M-Pesa', logo: 'mpesa' as const, color: 'success' },
    { value: 'crypto_wallet' as PaymentType, label: 'Crypto Wallet', logo: 'bitcoin' as const, color: 'warning' },
  ];

  const detectCardBrand = (number: string) => {
    const cleaned = number.replace(/\s/g, '');
    
    // Visa: starts with 4
    if (/^4/.test(cleaned)) return 'Visa';
    
    // Mastercard: 51-55, 2221-2720
    if (/^5[1-5]/.test(cleaned) || /^2(2[2-9][0-9]|[3-6][0-9]{2}|7[0-1][0-9]|720)/.test(cleaned)) {
      return 'Mastercard';
    }
    
    // American Express: 34, 37
    if (/^3[47]/.test(cleaned)) return 'American Express';
    
    // Discover: 6011, 622126-622925, 644-649, 65
    if (/^6011|^64[4-9]|^65|^622(1(2[6-9]|[3-9][0-9])|[2-8][0-9]{2}|9([0-1][0-9]|2[0-5]))/.test(cleaned)) {
      return 'Discover';
    }
    
    // Diners Club: 36, 38, 300-305
    if (/^3(0[0-5]|[68])/.test(cleaned)) return 'Diners Club';
    
    // JCB: 3528-3589
    if (/^35(2[89]|[3-8][0-9])/.test(cleaned)) return 'JCB';
    
    return '';
  };

  const handleCardNumberChange = (value: string) => {
    // Format card number with spaces
    const cleaned = value.replace(/\s/g, '');
    const formatted = cleaned.replace(/(\d{4})/g, '$1 ').trim();
    setCardNumber(formatted);
    setCardBrand(detectCardBrand(cleaned));
  };

  const handleSubmit = async (e: Event) => {
    e.preventDefault();
    const type = selectedType();
    if (!type) return;

    setError(null);
    setSubmitState('submitting');

    try {
      const method: any = { type };

      if (type === 'bank_account') {
        if (!bankName() || !accountNumber() || !routingNumber()) {
          setError('Please fill in all bank account fields');
          setSubmitState('error');
          return;
        }
        method.bank_name = bankName();
        method.account_type = accountType();
        method.account_number = accountNumber();
        method.routing_number = routingNumber();
      } else if (type === 'debit_card' || type === 'credit_card') {
        if (!cardNumber() || !cardExpMonth() || !cardExpYear() || !cardCVV()) {
          setError('Please fill in all card fields');
          setSubmitState('error');
          return;
        }
        method.card_number = cardNumber().replace(/\s/g, '');
        method.card_brand = cardBrand();
        method.card_exp_month = parseInt(cardExpMonth());
        method.card_exp_year = parseInt(cardExpYear());
        method.card_cvv = cardCVV();
      } else if (type === 'paypal') {
        if (!paypalEmail()) {
          setError('Please enter your PayPal email');
          setSubmitState('error');
          return;
        }
        method.paypal_email = paypalEmail();
      } else if (type === 'cashapp') {
        if (!cashappTag()) {
          setError('Please enter your Cash App $Cashtag');
          setSubmitState('error');
          return;
        }
        method.cashapp_tag = cashappTag();
      } else if (type === 'mpesa') {
        if (!mpesaPhone()) {
          setError('Please enter your M-Pesa phone number');
          setSubmitState('error');
          return;
        }
        method.mpesa_phone = mpesaPhone();
        method.mpesa_country = mpesaCountry();
      } else if (type === 'crypto_wallet') {
        if (!cryptoAddress()) {
          setError('Please enter your crypto wallet address');
          setSubmitState('error');
          return;
        }
        method.crypto_address = cryptoAddress();
        method.crypto_network = cryptoNetwork();
      }

      const result = await apiClient.addPaymentMethod(method);
      
      // Initiate verification automatically
      try {
        const verificationResult = await apiClient.initiatePaymentVerification(result.id);
        
        if (verificationResult.status === 'verified') {
          // Instant verification (e.g., cards)
          setSubmitState('success');
          setTimeout(() => {
            props.onSuccess();
            props.onClose();
          }, 2000);
        } else {
          // Requires user action (e.g., bank micro-deposits, M-Pesa STK push)
          setSubmitState('success');
          setError(null);
          setTimeout(() => {
            props.onSuccess();
            props.onClose();
          }, 2000);
        }
      } catch (verificationError: any) {
        // Payment method added but verification failed to initiate
        console.error('Verification initiation failed:', verificationError);
        setSubmitState('success');
        setTimeout(() => {
          props.onSuccess();
          props.onClose();
        }, 2000);
      }
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || err.message || 'Failed to add payment method';
      setError(errorMsg);
      setSubmitState('error');
    }
  };

  return (
    <div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-2 sm:p-4 overflow-y-auto">
      <div class="bg-terminal-900 border border-terminal-750 rounded-lg max-w-2xl w-full p-4 sm:p-6 my-4 sm:my-8">
        <div class="flex justify-between items-center mb-6">
          <h3 class="text-lg font-bold text-white">Add Payment Method</h3>
          <button
            onClick={props.onClose}
            class="text-gray-400 hover:text-white transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Payment Type Selection */}
        <Show when={!selectedType()}>
          <div class="grid grid-cols-2 md:grid-cols-3 gap-3">
            <For each={paymentTypes}>
              {(type) => (
                <button
                  onClick={() => setSelectedType(type.value)}
                  class="p-4 bg-terminal-850 border border-terminal-750 hover:border-terminal-600 rounded flex flex-col items-center gap-3 transition-all hover:scale-105 group"
                >
                  <div class="transform group-hover:scale-110 transition-transform">
                    <PaymentMethodLogo type={type.logo} size={48} />
                  </div>
                  <span class="text-white text-sm font-semibold">{type.label}</span>
                </button>
              )}
            </For>
          </div>
        </Show>

        {/* Payment Method Forms */}
        <Show when={selectedType()}>
          <form onSubmit={handleSubmit} class="space-y-4">
            {/* Bank Account Form */}
            <Show when={selectedType() === 'bank_account'}>
              <div class="space-y-3">
                <div>
                  <label class="block text-sm font-semibold text-gray-300 mb-1">Bank Name</label>
                  <input
                    type="text"
                    value={bankName()}
                    onInput={(e) => setBankName(e.currentTarget.value)}
                    placeholder="Chase, Bank of America, etc."
                    class="w-full px-3 py-2 bg-terminal-850 border border-terminal-750 rounded text-white placeholder-gray-500 focus:outline-none focus:border-primary-500"
                  />
                </div>
                <div>
                  <label class="block text-sm font-semibold text-gray-300 mb-1">Account Type</label>
                  <select
                    value={accountType()}
                    onChange={(e) => setAccountType(e.currentTarget.value as 'checking' | 'savings')}
                    class="w-full px-3 py-2 bg-terminal-850 border border-terminal-750 rounded text-white focus:outline-none focus:border-primary-500"
                  >
                    <option value="checking">Checking</option>
                    <option value="savings">Savings</option>
                  </select>
                </div>
                <div>
                  <label class="block text-sm font-semibold text-gray-300 mb-1">Account Number</label>
                  <input
                    type="text"
                    value={accountNumber()}
                    onInput={(e) => setAccountNumber(e.currentTarget.value)}
                    placeholder="Account number"
                    class="w-full px-3 py-2 bg-terminal-850 border border-terminal-750 rounded text-white placeholder-gray-500 focus:outline-none focus:border-primary-500"
                  />
                </div>
                <div>
                  <label class="block text-sm font-semibold text-gray-300 mb-1">Routing Number</label>
                  <input
                    type="text"
                    value={routingNumber()}
                    onInput={(e) => setRoutingNumber(e.currentTarget.value)}
                    placeholder="9-digit routing number"
                    maxLength={9}
                    class="w-full px-3 py-2 bg-terminal-850 border border-terminal-750 rounded text-white placeholder-gray-500 focus:outline-none focus:border-primary-500"
                  />
                </div>
              </div>
            </Show>

            {/* Card Form (Debit/Credit) */}
            <Show when={selectedType() === 'debit_card' || selectedType() === 'credit_card'}>
              <div class="space-y-3">
                <div>
                  <label class="block text-sm font-semibold text-gray-300 mb-1">Card Number</label>
                  <div class="relative">
                    <input
                      type="text"
                      value={cardNumber()}
                      onInput={(e) => handleCardNumberChange(e.currentTarget.value)}
                      placeholder="1234 5678 9012 3456"
                      maxLength={19}
                      class="w-full px-3 py-2 pr-12 bg-terminal-850 border border-terminal-750 rounded text-white placeholder-gray-500 focus:outline-none focus:border-accent-500"
                    />
                    <Show when={cardBrand()}>
                      <div class="absolute right-2 top-1/2 -translate-y-1/2">
                        <PaymentMethodLogo 
                          type={cardBrand().toLowerCase().replace(' ', '') as any} 
                          size={32}
                        />
                      </div>
                    </Show>
                  </div>
                  <Show when={cardBrand()}>
                    <div class="flex items-center gap-2 mt-1">
                      <p class="text-xs text-gray-400">Detected: {cardBrand()}</p>
                    </div>
                  </Show>
                </div>
                <div class="grid grid-cols-3 gap-3">
                  <div>
                    <label class="block text-sm font-semibold text-gray-300 mb-1">Exp Month</label>
                    <input
                      type="text"
                      value={cardExpMonth()}
                      onInput={(e) => setCardExpMonth(e.currentTarget.value)}
                      placeholder="MM"
                      maxLength={2}
                      class="w-full px-3 py-2 bg-terminal-850 border border-terminal-750 rounded text-white placeholder-gray-500 focus:outline-none focus:border-accent-500"
                    />
                  </div>
                  <div>
                    <label class="block text-sm font-semibold text-gray-300 mb-1">Exp Year</label>
                    <input
                      type="text"
                      value={cardExpYear()}
                      onInput={(e) => setCardExpYear(e.currentTarget.value)}
                      placeholder="YYYY"
                      maxLength={4}
                      class="w-full px-3 py-2 bg-terminal-850 border border-terminal-750 rounded text-white placeholder-gray-500 focus:outline-none focus:border-accent-500"
                    />
                  </div>
                  <div>
                    <label class="block text-sm font-semibold text-gray-300 mb-1">CVV</label>
                    <input
                      type="text"
                      value={cardCVV()}
                      onInput={(e) => setCardCVV(e.currentTarget.value)}
                      placeholder="123"
                      maxLength={4}
                      class="w-full px-3 py-2 bg-terminal-850 border border-terminal-750 rounded text-white placeholder-gray-500 focus:outline-none focus:border-accent-500"
                    />
                  </div>
                </div>
              </div>
            </Show>

            {/* PayPal Form */}
            <Show when={selectedType() === 'paypal'}>
              <div>
                <label class="block text-sm font-semibold text-gray-300 mb-1">PayPal Email</label>
                <input
                  type="email"
                  value={paypalEmail()}
                  onInput={(e) => setPaypalEmail(e.currentTarget.value)}
                  placeholder="your.email@example.com"
                  class="w-full px-3 py-2 bg-terminal-850 border border-terminal-750 rounded text-white placeholder-gray-500 focus:outline-none focus:border-info-500"
                />
              </div>
            </Show>

            {/* Cash App Form */}
            <Show when={selectedType() === 'cashapp'}>
              <div>
                <label class="block text-sm font-semibold text-gray-300 mb-1">Cash App $Cashtag</label>
                <input
                  type="text"
                  value={cashappTag()}
                  onInput={(e) => setCashappTag(e.currentTarget.value)}
                  placeholder="$YourCashtag"
                  class="w-full px-3 py-2 bg-terminal-850 border border-terminal-750 rounded text-white placeholder-gray-500 focus:outline-none focus:border-success-500"
                />
                <p class="text-xs text-gray-400 mt-1">Enter your Cash App $Cashtag (e.g., $JohnDoe)</p>
              </div>
            </Show>

            {/* M-Pesa Form */}
            <Show when={selectedType() === 'mpesa'}>
              <div class="space-y-3">
                <div>
                  <label class="block text-sm font-semibold text-gray-300 mb-1">Country</label>
                  <select
                    value={mpesaCountry()}
                    onChange={(e) => setMpesaCountry(e.currentTarget.value)}
                    class="w-full px-3 py-2 bg-terminal-850 border border-terminal-750 rounded text-white focus:outline-none focus:border-success-500"
                  >
                    <option value="KE">Kenya</option>
                    <option value="TZ">Tanzania</option>
                    <option value="UG">Uganda</option>
                    <option value="RW">Rwanda</option>
                  </select>
                </div>
                <div>
                  <label class="block text-sm font-semibold text-gray-300 mb-1">Phone Number</label>
                  <input
                    type="tel"
                    value={mpesaPhone()}
                    onInput={(e) => setMpesaPhone(e.currentTarget.value)}
                    placeholder="+254712345678"
                    class="w-full px-3 py-2 bg-terminal-850 border border-terminal-750 rounded text-white placeholder-gray-500 focus:outline-none focus:border-success-500"
                  />
                </div>
              </div>
            </Show>

            {/* Crypto Wallet Form */}
            <Show when={selectedType() === 'crypto_wallet'}>
              <div class="space-y-3">
                <div>
                  <label class="block text-sm font-semibold text-gray-300 mb-1">Network</label>
                  <select
                    value={cryptoNetwork()}
                    onChange={(e) => setCryptoNetwork(e.currentTarget.value)}
                    class="w-full px-3 py-2 bg-terminal-850 border border-terminal-750 rounded text-white focus:outline-none focus:border-warning-500"
                  >
                    <option value="bitcoin">Bitcoin (BTC)</option>
                    <option value="ethereum">Ethereum (ETH)</option>
                    <option value="usdc">USD Coin (USDC)</option>
                    <option value="usdt">Tether (USDT)</option>
                    <option value="sol">Solana (SOL)</option>
                  </select>
                </div>
                <div>
                  <label class="block text-sm font-semibold text-gray-300 mb-1">Wallet Address</label>
                  <input
                    type="text"
                    value={cryptoAddress()}
                    onInput={(e) => setCryptoAddress(e.currentTarget.value)}
                    placeholder="0x... or bc1..."
                    class="w-full px-3 py-2 bg-terminal-850 border border-terminal-750 rounded text-white placeholder-gray-500 focus:outline-none focus:border-warning-500 font-mono text-sm"
                  />
                </div>
              </div>
            </Show>

            {/* Success Message */}
            <Show when={submitState() === 'success'}>
              <div class="p-4 bg-success-500/10 border border-success-500 rounded flex items-center gap-3 animate-in fade-in duration-300">
                <CheckCircle size={24} class="text-success-500" />
                <div>
                  <p class="text-success-500 font-semibold">Payment Method Added!</p>
                  <p class="text-success-400 text-sm mt-1">Your payment method has been successfully added.</p>
                </div>
              </div>
            </Show>

            {/* Error Message */}
            <Show when={submitState() === 'error' && error()}>
              <div class="p-4 bg-danger-500/10 border border-danger-500 rounded flex items-start gap-3">
                <AlertCircle size={20} class="text-danger-500 mt-0.5" />
                <div class="flex-1">
                  <p class="text-danger-500 font-semibold">Failed to Add Payment Method</p>
                  <p class="text-danger-400 text-sm mt-1">{error()}</p>
                </div>
              </div>
            </Show>

            {/* Submitting State */}
            <Show when={submitState() === 'submitting'}>
              <div class="p-4 bg-primary-500/10 border border-primary-500 rounded flex items-center gap-3">
                <div class="animate-spin rounded-full h-5 w-5 border-2 border-primary-500 border-t-transparent"></div>
                <p class="text-primary-500 font-semibold">Processing payment method...</p>
              </div>
            </Show>

            {/* Action Buttons */}
            <div class="flex gap-3 pt-2">
              <button
                type="button"
                onClick={() => setSelectedType(null)}
                disabled={submitState() === 'submitting'}
                class="flex-1 px-4 py-2 bg-terminal-850 hover:bg-terminal-800 border border-terminal-750 text-white text-sm font-semibold rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Back
              </button>
              <button
                type="submit"
                disabled={submitState() === 'submitting' || submitState() === 'success'}
                class="flex-1 px-4 py-2 bg-accent-500 hover:bg-accent-600 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-semibold rounded transition-colors"
              >
                {submitState() === 'submitting' ? 'Processing...' : submitState() === 'success' ? 'Success!' : 'Add Payment Method'}
              </button>
            </div>
          </form>
        </Show>
      </div>
    </div>
  );
}
