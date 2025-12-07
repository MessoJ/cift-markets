import { createSignal, For, Show, createMemo } from 'solid-js';
import { 
  DollarSign, 
  CheckCircle2, 
  AlertTriangle,
  ArrowRight,
  ShieldCheck,
  Zap,
  Calendar,
  FileText,
  Copy,
  Printer,
  Plus
} from 'lucide-solid';
import { apiClient, PaymentMethod, TransferLimit } from '../../../lib/api/client';
import { formatCurrency } from '../../../lib/utils';
import { PaymentMethodLogo } from '../../../components/PaymentMethodLogo';
import { Modal } from '../../../components/ui/Modal';

interface DepositTabProps {
  paymentMethods: PaymentMethod[] | undefined;
  limits: TransferLimit | null;
  onSuccess: () => void;
}

export function DepositTab(props: DepositTabProps) {
  const [loading, setLoading] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [depositAmount, setDepositAmount] = createSignal('');
  const [selectedPaymentMethod, setSelectedPaymentMethod] = createSignal('');
  const [successMessage, setSuccessMessage] = createSignal<string | null>(null);
  
  // New Features State
  const [isRecurring, setIsRecurring] = createSignal(false);
  const [recurringFrequency, setRecurringFrequency] = createSignal('monthly');
  const [showWireModal, setShowWireModal] = createSignal(false);

  // Get selected payment method details
  const selectedMethod = createMemo(() => {
    const id = selectedPaymentMethod();
    return props.paymentMethods?.find(m => m.id === id);
  });

  // Determine transfer type based on payment method
  const getTransferType = (): "standard" | "instant" => {
    const method = selectedMethod();
    if (!method) return 'standard';
    
    // Instant for cards and M-Pesa, standard for others
    if (method.type === 'debit_card' || method.type === 'credit_card' || method.type === 'mpesa') {
      return 'instant';
    }
    return 'standard';
  };

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

  const handleDeposit = async () => {
    const rawAmount = parseFloat(depositAmount());
    if (isNaN(rawAmount) || rawAmount <= 0) {
      setError('Please enter a valid amount');
      return;
    }
    // Ensure 2 decimal places
    const amount = Math.round(rawAmount * 100) / 100;

    if (!selectedPaymentMethod()) {
      setError('Please select a payment method');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccessMessage(null);
    
    try {
      const payload = {
        amount,
        payment_method_id: selectedPaymentMethod(),
        transfer_type: getTransferType(),
      };
      console.log('Submitting deposit:', payload);

      await apiClient.initiateDeposit(payload);
      
      setDepositAmount('');
      setSelectedPaymentMethod('');
      setIsRecurring(false);
      
      const recurringMsg = isRecurring() ? ` and scheduled ${recurringFrequency()} auto-deposit` : '';
      setSuccessMessage(`Successfully initiated deposit of ${formatCurrency(amount)}${recurringMsg}`);
      
      props.onSuccess();
      
      // Clear success message after 5 seconds
      setTimeout(() => setSuccessMessage(null), 5000);
    } catch (err: any) {
      console.error('Deposit failed:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to initiate deposit');
    } finally {
      setLoading(false);
    }
  };

  const quickAmounts = [100, 500, 1000, 5000];

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    // Could add a small toast here
  };

  return (
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full relative">
      {/* Left Column: Payment Method & Amount */}
      <div class="lg:col-span-2 space-y-6">
        
        {/* Payment Method Selection */}
        <div>
          <h3 class="text-sm font-mono font-bold text-gray-400 mb-3 uppercase tracking-wider">Select Payment Method</h3>
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
            
            {/* Add New Method Placeholder */}
            <button class="p-4 rounded-xl border border-dashed border-terminal-700 bg-transparent hover:bg-terminal-800/50 hover:border-gray-500 transition-colors flex flex-col items-center justify-center gap-2 text-gray-500 group min-h-[120px]">
              <div class="p-3 rounded-full bg-terminal-800 group-hover:bg-terminal-700 transition-colors">
                <Plus size={20} />
              </div>
              <span class="text-xs font-mono font-bold">ADD NEW METHOD</span>
            </button>
          </div>
        </div>

        {/* Amount Selection */}
        <div>
          <h3 class="text-sm font-mono font-bold text-gray-400 mb-3 uppercase tracking-wider">Deposit Amount</h3>
          
          <div class="bg-terminal-800/50 border border-terminal-700 p-6 rounded-xl mb-3 relative overflow-hidden group focus-within:border-accent-500/50 focus-within:ring-1 focus-within:ring-accent-500/50 transition-all">
            <div class="absolute top-0 right-0 p-4 opacity-10 group-focus-within:opacity-20 transition-opacity">
              <DollarSign size={64} />
            </div>
            <div class="flex items-center gap-2 relative z-10">
              <span class="text-3xl text-gray-500 font-light">$</span>
              <input
                type="number"
                value={depositAmount()}
                onInput={(e) => setDepositAmount(e.currentTarget.value)}
                placeholder="0.00"
                class="bg-transparent text-4xl font-mono font-bold text-white focus:outline-none w-full placeholder-gray-700"
              />
            </div>
          </div>

          <div class="flex flex-wrap gap-2 mb-6">
            <For each={quickAmounts}>
              {(amount) => (
                <button
                  onClick={() => setDepositAmount(amount.toString())}
                  class="px-4 py-2 bg-terminal-800 border border-terminal-700 rounded-lg text-xs font-mono font-bold text-gray-400 hover:bg-terminal-700 hover:text-white hover:border-gray-500 transition-all"
                >
                  {formatCurrency(amount)}
                </button>
              )}
            </For>
          </div>

          {/* Recurring Deposit Toggle */}
          <div class="mb-6 p-4 bg-terminal-800/30 border border-terminal-750 rounded-xl">
            <div class="flex items-center justify-between mb-3">
              <div class="flex items-center gap-3">
                <div class="p-2 bg-accent-500/10 rounded-lg text-accent-400">
                  <Calendar size={18} />
                </div>
                <div>
                  <span class="text-sm font-bold text-white block">Recurring Deposit</span>
                  <span class="text-xs text-gray-500">Automate your investment strategy</span>
                </div>
              </div>
              <button 
                onClick={() => setIsRecurring(!isRecurring())}
                class={`w-12 h-6 rounded-full relative transition-colors ${isRecurring() ? 'bg-accent-500' : 'bg-terminal-600'}`}
              >
                <div class={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform shadow-sm ${isRecurring() ? 'translate-x-7' : 'translate-x-1'}`}></div>
              </button>
            </div>
            
            <Show when={isRecurring()}>
              <div class="flex gap-2 mt-4 pt-4 border-t border-terminal-700/50 animate-in fade-in slide-in-from-top-2 duration-200">
                <For each={['weekly', 'bi-weekly', 'monthly']}>
                  {(freq) => (
                    <button
                      onClick={() => setRecurringFrequency(freq)}
                      class={`
                        flex-1 py-2 text-xs font-mono font-bold uppercase rounded-lg border transition-all
                        ${recurringFrequency() === freq 
                          ? 'bg-accent-500/20 border-accent-500 text-accent-400' 
                          : 'bg-terminal-900 border-terminal-700 text-gray-400 hover:border-gray-500 hover:bg-terminal-800'}
                      `}
                    >
                      {freq}
                    </button>
                  )}
                </For>
              </div>
            </Show>
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
            onClick={handleDeposit}
            disabled={loading()}
            class={`
              w-full py-4 rounded-xl font-bold font-mono text-sm uppercase tracking-wider flex items-center justify-center gap-2 transition-all
              ${loading() 
                ? 'bg-terminal-800 text-gray-500 cursor-not-allowed' 
                : 'bg-accent-600 hover:bg-accent-500 text-white shadow-lg shadow-accent-900/20 hover:shadow-accent-900/40 hover:-translate-y-0.5'}
            `}
          >
            <Show when={loading()} fallback={<>Initiate Deposit <ArrowRight size={16} /></>}>
              Processing...
            </Show>
          </button>
        </div>
      </div>

      {/* Right Column: Info & Limits */}
      <div class="space-y-6">
        <div class="bg-terminal-800/30 border border-terminal-750 p-5 rounded-xl backdrop-blur-sm">
          <h3 class="text-xs font-mono font-bold text-gray-400 mb-5 uppercase tracking-wider flex items-center gap-2">
            <ShieldCheck size={14} />
            Security & Limits
          </h3>
          
          <div class="space-y-5">
            <div>
              <div class="flex justify-between text-xs mb-2">
                <span class="text-gray-400">Daily Limit</span>
                <span class="text-white font-mono font-bold">{formatCurrency(props.limits?.daily_deposit_limit || 50000)}</span>
              </div>
              <div class="w-full bg-terminal-900 h-2 rounded-full overflow-hidden border border-terminal-800">
                <div 
                  class="bg-gradient-to-r from-success-600 to-success-400 h-full rounded-full transition-all duration-500"
                  style={{ width: `${((props.limits?.daily_deposit_limit || 50000) - (props.limits?.daily_deposit_remaining || 50000)) / (props.limits?.daily_deposit_limit || 50000) * 100}%` }}
                ></div>
              </div>
              <div class="text-[10px] text-gray-500 mt-1.5 text-right font-mono">
                {formatCurrency(props.limits?.daily_deposit_remaining || 0)} remaining
              </div>
            </div>

            <div class="pt-5 border-t border-terminal-700/50">
              <h4 class="text-xs font-bold text-white mb-3">Processing Times</h4>
              <ul class="space-y-3">
                <li class="flex justify-between text-xs">
                  <span class="text-gray-500">Bank Transfer (ACH)</span>
                  <span class="text-gray-300 font-mono">1-3 Days</span>
                </li>
                <li class="flex justify-between text-xs">
                  <span class="text-gray-500">Debit Card</span>
                  <span class="text-success-400 font-mono font-bold">Instant</span>
                </li>
                <li class="flex justify-between text-xs">
                  <span class="text-gray-500">Wire Transfer</span>
                  <span class="text-gray-300 font-mono">Same Day</span>
                </li>
                <li class="flex justify-between text-xs">
                  <span class="text-gray-500">Crypto</span>
                  <span class="text-success-400 font-mono font-bold">~10 Mins</span>
                </li>
              </ul>
            </div>
            
            {/* Wire Instructions Button */}
            <div class="pt-5 border-t border-terminal-700/50">
              <button 
                onClick={() => setShowWireModal(true)}
                class="w-full py-2.5 bg-terminal-800 hover:bg-terminal-700 text-gray-300 hover:text-white text-xs font-bold uppercase tracking-wider rounded-lg flex items-center justify-center gap-2 transition-all border border-terminal-700 hover:border-gray-600"
              >
                <FileText size={14} />
                View Wire Instructions
              </button>
            </div>
          </div>
        </div>

        <div class="bg-gradient-to-br from-accent-900/20 to-transparent border border-accent-500/20 p-5 rounded-xl relative overflow-hidden">
          <div class="absolute top-0 right-0 p-4 opacity-10">
            <Zap size={80} />
          </div>
          <div class="flex items-start gap-4 relative z-10">
            <div class="p-2.5 bg-accent-500/20 rounded-lg text-accent-400 shrink-0">
              <Zap size={20} />
            </div>
            <div>
              <h4 class="text-sm font-bold text-accent-400 mb-1.5">Instant Buying Power</h4>
              <p class="text-xs text-accent-200/70 leading-relaxed">
                Deposits up to $1,000 are available for trading immediately, even while funds are clearing.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Wire Instructions Modal */}
      <Modal
        open={showWireModal()}
        onClose={() => setShowWireModal(false)}
        title="Wire Transfer Instructions"
        size="md"
      >
        <div class="space-y-6">
          <div class="bg-warning-900/10 border border-warning-900/30 p-4 rounded-lg flex items-start gap-3">
            <AlertTriangle class="text-warning-500 flex-shrink-0 mt-0.5" size={18} />
            <p class="text-xs text-warning-200 leading-relaxed">
              Please ensure you include your unique Reference ID in the memo field. Failure to do so may result in delays or returned funds.
            </p>
          </div>

          <div class="space-y-4">
            <div class="grid grid-cols-2 gap-4">
              <div>
                <label class="text-[10px] text-gray-500 uppercase tracking-wider block mb-1.5 font-bold">Beneficiary Name</label>
                <div class="flex items-center justify-between bg-terminal-950 border border-terminal-800 p-3 rounded-lg group hover:border-gray-600 transition-colors">
                  <span class="text-sm font-mono text-white">CIFT Markets LLC</span>
                  <button onClick={() => copyToClipboard('CIFT Markets LLC')} class="text-gray-500 hover:text-white opacity-0 group-hover:opacity-100 transition-opacity"><Copy size={14} /></button>
                </div>
              </div>
              <div>
                <label class="text-[10px] text-gray-500 uppercase tracking-wider block mb-1.5 font-bold">Bank Name</label>
                <div class="flex items-center justify-between bg-terminal-950 border border-terminal-800 p-3 rounded-lg group hover:border-gray-600 transition-colors">
                  <span class="text-sm font-mono text-white">Silvergate Bank</span>
                  <button onClick={() => copyToClipboard('Silvergate Bank')} class="text-gray-500 hover:text-white opacity-0 group-hover:opacity-100 transition-opacity"><Copy size={14} /></button>
                </div>
              </div>
            </div>

            <div>
              <label class="text-[10px] text-gray-500 uppercase tracking-wider block mb-1.5 font-bold">Account Number</label>
              <div class="flex items-center justify-between bg-terminal-950 border border-terminal-800 p-3 rounded-lg group hover:border-gray-600 transition-colors">
                <span class="text-sm font-mono text-white">9876543210</span>
                <button onClick={() => copyToClipboard('9876543210')} class="text-gray-500 hover:text-white opacity-0 group-hover:opacity-100 transition-opacity"><Copy size={14} /></button>
              </div>
            </div>

            <div>
              <label class="text-[10px] text-gray-500 uppercase tracking-wider block mb-1.5 font-bold">Routing Number (ABA)</label>
              <div class="flex items-center justify-between bg-terminal-950 border border-terminal-800 p-3 rounded-lg group hover:border-gray-600 transition-colors">
                <span class="text-sm font-mono text-white">123456789</span>
                <button onClick={() => copyToClipboard('123456789')} class="text-gray-500 hover:text-white opacity-0 group-hover:opacity-100 transition-opacity"><Copy size={14} /></button>
              </div>
            </div>

            <div>
              <label class="text-[10px] text-gray-500 uppercase tracking-wider block mb-1.5 font-bold">Reference ID (Memo)</label>
              <div class="flex items-center justify-between bg-accent-900/10 border border-accent-500/30 p-3 rounded-lg group hover:border-accent-500/50 transition-colors">
                <span class="text-sm font-mono font-bold text-accent-400">CIFT-8829-USER</span>
                <button onClick={() => copyToClipboard('CIFT-8829-USER')} class="text-accent-500 hover:text-white opacity-0 group-hover:opacity-100 transition-opacity"><Copy size={14} /></button>
              </div>
            </div>
          </div>

          <div class="flex justify-end pt-4 border-t border-terminal-800">
            <button class="flex items-center gap-2 px-4 py-2 bg-terminal-800 hover:bg-terminal-700 text-white rounded-lg text-xs font-bold uppercase tracking-wider transition-colors">
              <Printer size={14} />
              Print Instructions
            </button>
          </div>
        </div>
      </Modal>
    </div>
  );
}
