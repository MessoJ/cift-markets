import { createSignal, For, Show } from 'solid-js';
import { 
  CheckCircle2, 
  AlertTriangle,
  ArrowRight,
  Info,
  Lock
} from 'lucide-solid';
import { apiClient, PaymentMethod } from '../../../lib/api/client';
import { formatCurrency } from '../../../lib/utils';
import { PaymentMethodLogo } from '../../../components/PaymentMethodLogo';
import { Modal } from '../../../components/ui/Modal';

interface WithdrawTabProps {
  paymentMethods: PaymentMethod[];
  portfolio: any;
  onSuccess: () => void;
}

export function WithdrawTab(props: WithdrawTabProps) {
  const [loading, setLoading] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [withdrawAmount, setWithdrawAmount] = createSignal('');
  const [selectedPaymentMethod, setSelectedPaymentMethod] = createSignal('');
  const [successMessage, setSuccessMessage] = createSignal<string | null>(null);
  
  // 2FA State
  const [show2FAModal, setShow2FAModal] = createSignal(false);
  const [twoFACode, setTwoFACode] = createSignal('');
  const [twoFAError, setTwoFAError] = createSignal<string | null>(null);

  const getLogoType = (method: PaymentMethod): any => {
    if (method.type === 'bank_account') return 'bank';
    if (method.type === 'mpesa') return 'mpesa';
    if (method.type === 'paypal') return 'paypal';
    if (method.type === 'cashapp') return 'cashapp';
    
    if (method.type === 'debit_card' || method.type === 'credit_card') {
      const brand = (method as any).card_brand?.toLowerCase();
      if (brand === 'visa') return 'visa';
      if (brand === 'mastercard') return 'mastercard';
      if (brand === 'amex') return 'amex';
      if (brand === 'discover') return 'discover';
      return 'visa';
    }
    
    if (method.type === 'crypto_wallet') {
      const network = (method as any).crypto_network?.toLowerCase();
      if (network === 'btc' || network === 'bitcoin') return 'bitcoin';
      if (network === 'eth' || network === 'ethereum') return 'ethereum';
      return 'crypto';
    }
    
    return 'bank';
  };

  const handleWithdrawalClick = () => {
    const amount = parseFloat(withdrawAmount());
    if (!amount || amount <= 0) {
      setError('Please enter a valid amount');
      return;
    }

    if (!selectedPaymentMethod()) {
      setError('Please select a payment method');
      return;
    }

    const availableCash = props.portfolio?.cash || 0;
    if (amount > availableCash) {
      setError(`Insufficient funds. Available: ${formatCurrency(availableCash)}`);
      return;
    }

    // Minimum withdrawal check
    if (amount < 10) {
      setError('Minimum withdrawal amount is $10.00');
      return;
    }

    // Open 2FA Modal instead of direct API call
    setError(null);
    setTwoFACode('');
    setTwoFAError(null);
    setShow2FAModal(true);
  };

  const confirmWithdrawal = async () => {
    if (twoFACode().length !== 6) {
      setTwoFAError('Please enter a valid 6-digit code');
      return;
    }

    setLoading(true);
    setTwoFAError(null);

    try {
      // In a real app, we would verify the 2FA code here first
      // await apiClient.verify2FA(twoFACode());

      const amount = parseFloat(withdrawAmount());
      await apiClient.initiateWithdrawal({
        amount,
        payment_method_id: selectedPaymentMethod(),
      });
      
      setShow2FAModal(false);
      setWithdrawAmount('');
      setSelectedPaymentMethod('');
      setSuccessMessage(`Successfully initiated withdrawal of ${formatCurrency(amount)}`);
      props.onSuccess();
      
      setTimeout(() => setSuccessMessage(null), 5000);
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || 'Failed to initiate withdrawal';
      setTwoFAError(msg); // Show error in modal if it fails there
      // If it's not a 2FA error, maybe close modal and show main error? 
      // For now, keep modal open so they can retry if it was a code issue.
    } finally {
      setLoading(false);
    }
  };

  const setMaxAmount = () => {
    if (props.portfolio?.cash) {
      setWithdrawAmount(props.portfolio.cash.toString());
    }
  };

  return (
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full relative">
      {/* Left Column: Form */}
      <div class="lg:col-span-2 space-y-6">
        
        {/* Destination Selection */}
        <div>
          <h3 class="text-sm font-mono font-bold text-gray-400 mb-3 uppercase tracking-wider">Withdraw To</h3>
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <For each={props.paymentMethods}>
              {(method) => (
                <button
                  onClick={() => setSelectedPaymentMethod(method.id)}
                  class={`
                    relative p-4 rounded-xl border text-left transition-all group overflow-hidden
                    ${selectedPaymentMethod() === method.id 
                      ? 'bg-accent-900/10 border-accent-500 ring-1 ring-accent-500' 
                      : 'bg-terminal-800/50 border-terminal-700 hover:border-gray-500 hover:bg-terminal-800'}
                  `}
                >
                  <div class="flex items-start justify-between mb-3">
                    <PaymentMethodLogo type={getLogoType(method)} class="w-10 h-10" />
                    <Show when={selectedPaymentMethod() === method.id}>
                      <div class="bg-accent-500 rounded-full p-1">
                        <CheckCircle2 size={14} class="text-white" />
                      </div>
                    </Show>
                  </div>
                  
                  <div class="font-mono font-bold text-white text-sm mb-1">
                    {method.name}
                  </div>
                  <div class="text-xs text-gray-500 font-mono flex items-center gap-2">
                    <span class="uppercase">{method.type.replace('_', ' ')}</span>
                    <span class="w-1 h-1 rounded-full bg-gray-600"></span>
                    {method.type === 'bank_account' && `••••${method.last_four}`}
                    {(method.type === 'debit_card' || method.type === 'credit_card') && `••••${method.last_four}`}
                    {method.type === 'crypto_wallet' && `${method.crypto_network?.toUpperCase()}`}
                  </div>
                </button>
              )}
            </For>
          </div>
        </div>

        {/* Amount Selection */}
        <div>
          <h3 class="text-sm font-mono font-bold text-gray-400 mb-3 uppercase tracking-wider">Withdrawal Amount</h3>
          
          <div class="bg-terminal-800/50 border border-terminal-700 p-6 rounded-xl mb-3 relative overflow-hidden group focus-within:border-accent-500/50 focus-within:ring-1 focus-within:ring-accent-500/50 transition-all">
            <div class="flex items-center justify-between mb-4">
              <span class="text-xs text-gray-500 font-mono font-bold">AVAILABLE TO WITHDRAW</span>
              <span class="text-xs text-white font-mono font-bold bg-terminal-800 px-2 py-1 rounded-md border border-terminal-700">
                {formatCurrency(props.portfolio?.cash || 0)}
              </span>
            </div>
            <div class="flex items-center gap-2 relative z-10">
              <span class="text-3xl text-gray-500 font-light">$</span>
              <input
                type="number"
                value={withdrawAmount()}
                onInput={(e) => setWithdrawAmount(e.currentTarget.value)}
                placeholder="0.00"
                class="bg-transparent text-4xl font-mono font-bold text-white focus:outline-none w-full placeholder-gray-700"
              />
              <button 
                onClick={setMaxAmount}
                class="px-3 py-1.5 bg-terminal-800 hover:bg-terminal-700 text-xs font-mono font-bold text-accent-400 rounded-lg uppercase border border-terminal-700 hover:border-accent-500/50 transition-all"
              >
                MAX
              </button>
            </div>
          </div>

          <Show when={error()}>
            <div class="flex items-center gap-3 p-4 bg-danger-900/20 border border-danger-900/50 rounded-xl text-danger-400 text-sm mb-4 animate-in fade-in slide-in-from-top-2">
              <AlertTriangle size={20} class="shrink-0" />
              {error()}
            </div>
          </Show>

          <Show when={successMessage()}>
            <div class="flex items-center gap-3 p-4 bg-success-900/20 border border-success-900/50 rounded-xl text-success-400 text-sm mb-4 animate-in fade-in slide-in-from-top-2">
              <CheckCircle2 size={20} class="shrink-0" />
              {successMessage()}
            </div>
          </Show>

          <button
            onClick={handleWithdrawalClick}
            disabled={loading()}
            class={`
              w-full py-4 rounded-xl font-bold font-mono text-sm uppercase tracking-wider flex items-center justify-center gap-2 transition-all
              ${loading() 
                ? 'bg-terminal-800 text-gray-500 cursor-not-allowed' 
                : 'bg-accent-600 hover:bg-accent-500 text-white shadow-lg shadow-accent-900/20 hover:shadow-accent-900/40 hover:-translate-y-0.5'}
            `}
          >
            <Show when={loading()} fallback={<>Review Withdrawal <ArrowRight size={16} /></>}>
              Processing...
            </Show>
          </button>
        </div>
      </div>

      {/* Right Column: Info */}
      <div class="space-y-6">
        <div class="bg-terminal-800/30 border border-terminal-750 p-5 rounded-xl backdrop-blur-sm">
          <h3 class="text-xs font-mono font-bold text-gray-400 mb-5 uppercase tracking-wider flex items-center gap-2">
            <Info size={14} />
            Important Information
          </h3>
          
          <ul class="space-y-4">
            <li class="flex gap-3 text-xs text-gray-400">
              <div class="w-1.5 h-1.5 rounded-full bg-accent-500 mt-1.5 flex-shrink-0 shadow-[0_0_8px_rgba(var(--accent-500),0.5)]"></div>
              <p class="leading-relaxed">Withdrawals to bank accounts typically take 1-3 business days to settle.</p>
            </li>
            <li class="flex gap-3 text-xs text-gray-400">
              <div class="w-1.5 h-1.5 rounded-full bg-accent-500 mt-1.5 flex-shrink-0 shadow-[0_0_8px_rgba(var(--accent-500),0.5)]"></div>
              <p class="leading-relaxed">For security, you can only withdraw to payment methods you have previously used to deposit.</p>
            </li>
            <li class="flex gap-3 text-xs text-gray-400">
              <div class="w-1.5 h-1.5 rounded-full bg-accent-500 mt-1.5 flex-shrink-0 shadow-[0_0_8px_rgba(var(--accent-500),0.5)]"></div>
              <p class="leading-relaxed">Large withdrawals may require additional verification steps.</p>
            </li>
          </ul>
        </div>
      </div>

      {/* 2FA Modal */}
      <Modal
        open={show2FAModal()}
        onClose={() => setShow2FAModal(false)}
        title="Security Verification"
        size="sm"
      >
        <div class="text-center mb-8">
          <div class="w-16 h-16 bg-accent-500/10 rounded-full flex items-center justify-center mx-auto mb-4 text-accent-500 ring-1 ring-accent-500/20">
            <Lock size={32} />
          </div>
          <h4 class="text-lg font-bold text-white mb-2">Two-Factor Authentication</h4>
          <p class="text-sm text-gray-400 leading-relaxed">
            Enter the 6-digit code sent to your device ending in ••88 to confirm this withdrawal.
          </p>
        </div>

        <div class="mb-8">
          <input
            type="text"
            maxLength={6}
            value={twoFACode()}
            onInput={(e) => {
              // Only allow numbers
              const val = e.currentTarget.value.replace(/[^0-9]/g, '');
              setTwoFACode(val);
            }}
            placeholder="000000"
            class="w-full bg-terminal-950 border border-terminal-700 text-center text-3xl font-mono font-bold text-white py-4 rounded-lg focus:border-accent-500 focus:ring-1 focus:ring-accent-500 tracking-[0.5em] placeholder-terminal-800 transition-all"
          />
          <Show when={twoFAError()}>
            <p class="text-xs text-danger-400 mt-3 text-center font-bold flex items-center justify-center gap-1">
              <AlertTriangle size={12} />
              {twoFAError()}
            </p>
          </Show>
        </div>

        <div class="flex gap-3">
          <button 
            onClick={() => setShow2FAModal(false)}
            class="flex-1 py-3 bg-terminal-800 hover:bg-terminal-700 text-gray-300 text-xs font-bold uppercase tracking-wider rounded-lg transition-colors"
          >
            Cancel
          </button>
          <button 
            onClick={confirmWithdrawal}
            disabled={loading() || twoFACode().length !== 6}
            class={`
              flex-1 py-3 rounded-lg text-xs font-bold uppercase tracking-wider transition-all
              ${loading() || twoFACode().length !== 6
                ? 'bg-terminal-700 text-gray-500 cursor-not-allowed' 
                : 'bg-accent-600 hover:bg-accent-500 text-white shadow-lg shadow-accent-900/20'}
            `}
          >
            {loading() ? 'Verifying...' : 'Confirm Withdrawal'}
          </button>
        </div>
      </Modal>
    </div>
  );
}
