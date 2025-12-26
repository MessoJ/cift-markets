import { createSignal, For, Show } from 'solid-js';
import { 
  Plus, 
  Trash2, 
  Building2, 
  CreditCard, 
  Bitcoin, 
  CheckCircle2,
  AlertCircle,
  X,
  Lock,
  ShieldCheck,
  MoreVertical
} from 'lucide-solid';
import { apiClient, PaymentMethod } from '../../../lib/api/client';
import { PaymentMethodLogo } from '../../../components/PaymentMethodLogo';
import { Modal } from '../../../components/ui/Modal';

// Helper to load Plaid script dynamically
const loadPlaidScript = () => {
  return new Promise<any>((resolve, reject) => {
    if ((window as any).Plaid) {
      resolve((window as any).Plaid);
      return;
    }
    const script = document.createElement('script');
    script.src = 'https://cdn.plaid.com/link/v2/stable/link-initialize.js';
    script.onload = () => resolve((window as any).Plaid);
    script.onerror = reject;
    document.head.appendChild(script);
  });
};

interface PaymentMethodsTabProps {
  paymentMethods: PaymentMethod[];
  onUpdate: () => void;
}

export function PaymentMethodsTab(props: PaymentMethodsTabProps) {
  const [showAddModal, setShowAddModal] = createSignal(false);
  const [deletingId, setDeletingId] = createSignal<string | null>(null);
  
  // Add Method State
  const [addType, setAddType] = createSignal<'bank_account' | 'debit_card' | 'crypto_wallet' | 'mpesa' | 'paypal'>('bank_account');
  const [formData, setFormData] = createSignal<any>({});

  const handleConnectPlaid = async () => {
    try {
      // Load Plaid script
      const Plaid = await loadPlaidScript();
      
      // Get Link Token
      const { data } = await apiClient.post('/funding/plaid/link-token');
      
      if (!data.link_token) {
        throw new Error('No link token received');
      }

      const handler = Plaid.create({
        token: data.link_token,
        onSuccess: async (public_token: string, metadata: any) => {
          try {
            // Exchange token
            await apiClient.post('/funding/plaid/exchange-token', {
              public_token,
              account_id: metadata.account?.id || metadata.account_id
            });
            
            // Close modal and refresh
            setShowAddModal(false);
            props.onUpdate();
            // alert('Bank account connected successfully!');
          } catch (err) {
            console.error('Token exchange failed', err);
            alert('Failed to link bank account. Please try again.');
          }
        },
        onExit: (err: any, metadata: any) => {
          if (err) console.error('Plaid exit', err);
        },
      });

      handler.open();
    } catch (err) {
      console.error('Plaid connection failed', err);
      alert('Failed to initialize bank connection. Please try again later.');
    }
  };
  const [adding, setAdding] = createSignal(false);
  const [addError, setAddError] = createSignal<string | null>(null);

  const handleVerify = async (id: string) => {
    try {
      const result = await apiClient.initiatePaymentVerification(id);
      
      if (result.oauth_url) {
        // Redirect for OAuth (PayPal, etc)
        window.location.href = result.oauth_url;
      } else {
        alert(result.message || 'Verification initiated. Check your email or phone.');
      }
    } catch (err: any) {
      console.error('Verification failed', err);
      alert(err.response?.data?.detail || err.message || 'Failed to initiate verification');
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to remove this payment method?')) return;
    
    setDeletingId(id);
    try {
      // FIXED: Use removePaymentMethod instead of deletePaymentMethod
      await apiClient.removePaymentMethod(id);
      props.onUpdate();
    } catch (err) {
      console.error('Failed to delete payment method', err);
    } finally {
      setDeletingId(null);
    }
  };

  const handleAddSubmit = async (e: Event) => {
    e.preventDefault();
    setAdding(true);
    setAddError(null);

    try {
      const data = {
        type: addType(),
        ...formData(),
        // Add dummy data for required fields not in form
        provider: addType() === 'bank_account' ? 'plaid' : 'stripe',
        last_four: addType() === 'bank_account' 
          ? formData().account_number?.slice(-4) || '0000'
          : formData().card_number?.slice(-4) || '0000'
      };

      await apiClient.addPaymentMethod(data);
      setShowAddModal(false);
      setFormData({});
      props.onUpdate();
    } catch (err: any) {
      setAddError(err.response?.data?.detail || err.message || 'Failed to add payment method');
    } finally {
      setAdding(false);
    }
  };

  const updateField = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
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

  return (
    <div class="h-full flex flex-col">
      {/* Header */}
      <div class="flex items-center justify-between mb-6">
        <div>
          <h3 class="text-lg font-bold text-white mb-1">Linked Accounts</h3>
          <p class="text-xs text-gray-400">Manage your payment methods for deposits and withdrawals.</p>
        </div>
        <button 
          onClick={() => setShowAddModal(true)}
          class="flex items-center gap-2 px-4 py-2 bg-accent-600 hover:bg-accent-500 text-white rounded-lg text-xs font-bold uppercase tracking-wider transition-all shadow-lg shadow-accent-900/20 hover:shadow-accent-900/40 hover:-translate-y-0.5"
        >
          <Plus size={16} />
          Add New
        </button>
      </div>

      {/* List */}
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 overflow-y-auto pb-4 custom-scrollbar">
        <For each={props.paymentMethods}>
          {(method) => (
            <div class="bg-terminal-800/50 border border-terminal-700 p-5 rounded-xl group hover:border-gray-600 transition-all relative overflow-hidden">
              <div class="absolute top-0 right-0 p-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <button 
                  onClick={() => handleDelete(method.id)}
                  disabled={deletingId() === method.id}
                  class="p-2 text-gray-500 hover:text-danger-400 hover:bg-danger-900/20 rounded-lg transition-colors"
                >
                  <Trash2 size={16} />
                </button>
              </div>

              <div class="flex items-start gap-4">
                <PaymentMethodLogo type={getLogoType(method)} class="w-12 h-12" />
                
                <div>
                  <div class="flex items-center gap-2 mb-1">
                    <h4 class="font-bold text-white text-sm">{method.name}</h4>
                    <Show when={method.is_default}>
                      <span class="px-1.5 py-0.5 bg-accent-900/30 text-accent-400 text-[10px] font-bold uppercase rounded border border-accent-500/30">Default</span>
                    </Show>
                  </div>
                  
                  <div class="text-xs text-gray-400 font-mono mb-3 flex items-center gap-2">
                    <span class="capitalize">{method.type.replace('_', ' ')}</span>
                    <span class="w-1 h-1 rounded-full bg-gray-600"></span>
                    {method.type === 'bank_account' && `••••${method.last_four}`}
                    {(method.type === 'debit_card' || method.type === 'credit_card') && `••••${method.last_four}`}
                    {method.type === 'crypto_wallet' && <span class="uppercase">{method.crypto_network}</span>}
                  </div>

                  <div class="flex items-center gap-4">
                    <Show when={method.is_verified} fallback={
                      <button 
                        onClick={() => handleVerify(method.id)}
                        class="flex items-center gap-1.5 text-[10px] text-amber-400 bg-amber-900/10 px-2 py-1 rounded border border-amber-900/30 hover:bg-amber-900/20 transition-colors"
                      >
                        <AlertCircle size={12} />
                        VERIFY NOW
                      </button>
                    }>
                      <div class="flex items-center gap-1.5 text-[10px] text-success-400 bg-success-900/10 px-2 py-1 rounded border border-success-900/30">
                        <CheckCircle2 size={12} />
                        VERIFIED
                      </div>
                    </Show>
                    <div class="flex items-center gap-1.5 text-[10px] text-gray-500">
                      <ShieldCheck size={12} />
                      SECURE
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </For>

        {/* Empty State */}
        <Show when={props.paymentMethods.length === 0}>
          <div class="col-span-1 md:col-span-2 flex flex-col items-center justify-center p-12 border border-dashed border-terminal-700 rounded-xl bg-terminal-900/30 text-center">
            <div class="w-16 h-16 bg-terminal-800 rounded-full flex items-center justify-center mb-4 text-gray-600">
              <CreditCard size={32} />
            </div>
            <h4 class="text-lg font-bold text-white mb-2">No Payment Methods</h4>
            <p class="text-sm text-gray-500 mb-6 max-w-md">
              Add a bank account, card, or crypto wallet to start funding your account.
            </p>
            <button 
              onClick={() => setShowAddModal(true)}
              class="px-6 py-2 bg-terminal-800 hover:bg-terminal-700 text-white rounded-lg text-xs font-bold uppercase tracking-wider transition-colors border border-terminal-600"
            >
              Add First Method
            </button>
          </div>
        </Show>
      </div>

      {/* Add Method Modal */}
      <Modal
        open={showAddModal()}
        onClose={() => setShowAddModal(false)}
        title="Add Payment Method"
        size="lg"
      >
        <div class="flex flex-col md:flex-row h-[500px] md:h-auto">
          {/* Sidebar */}
          <div class="w-full md:w-1/3 border-b md:border-b-0 md:border-r border-terminal-700 bg-terminal-900/50 p-4 space-y-2 overflow-y-auto">
            <div class="text-[10px] text-gray-500 uppercase tracking-wider mb-2 font-bold">Payment Methods</div>
            <button 
              onClick={() => setAddType('bank_account')}
              class={`w-full flex items-center gap-3 p-3 rounded-lg text-left transition-all ${addType() === 'bank_account' ? 'bg-accent-600 text-white shadow-lg shadow-accent-900/20' : 'text-gray-400 hover:bg-terminal-800 hover:text-white'}`}
            >
              <Building2 size={18} />
              <span class="text-sm font-bold">Bank Account</span>
            </button>
            <button 
              onClick={() => setAddType('debit_card')}
              class={`w-full flex items-center gap-3 p-3 rounded-lg text-left transition-all ${addType() === 'debit_card' ? 'bg-accent-600 text-white shadow-lg shadow-accent-900/20' : 'text-gray-400 hover:bg-terminal-800 hover:text-white'}`}
            >
              <CreditCard size={18} />
              <span class="text-sm font-bold">Debit / Credit</span>
            </button>
            <button 
              onClick={() => setAddType('crypto_wallet')}
              class={`w-full flex items-center gap-3 p-3 rounded-lg text-left transition-all ${addType() === 'crypto_wallet' ? 'bg-accent-600 text-white shadow-lg shadow-accent-900/20' : 'text-gray-400 hover:bg-terminal-800 hover:text-white'}`}
            >
              <Bitcoin size={18} />
              <span class="text-sm font-bold">Crypto Wallet</span>
            </button>
            
            <div class="border-t border-terminal-700 my-3"></div>
            <div class="text-[10px] text-gray-500 uppercase tracking-wider mb-2 font-bold">Regional / Mobile</div>
            
            <button 
              onClick={() => setAddType('mpesa')}
              class={`w-full flex items-center gap-3 p-3 rounded-lg text-left transition-all ${addType() === 'mpesa' ? 'bg-green-600 text-white shadow-lg shadow-green-900/20' : 'text-gray-400 hover:bg-terminal-800 hover:text-white'}`}
            >
              <svg class="w-[18px] h-[18px]" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
              </svg>
              <div>
                <span class="text-sm font-bold">M-Pesa</span>
                <div class="text-[10px] opacity-70">Kenya / East Africa</div>
              </div>
            </button>
            <button 
              onClick={() => setAddType('paypal')}
              class={`w-full flex items-center gap-3 p-3 rounded-lg text-left transition-all ${addType() === 'paypal' ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' : 'text-gray-400 hover:bg-terminal-800 hover:text-white'}`}
            >
              <svg class="w-[18px] h-[18px]" viewBox="0 0 24 24" fill="currentColor">
                <path d="M7.076 21.337H2.47a.641.641 0 0 1-.633-.74L4.944 3.72a.771.771 0 0 1 .76-.65h6.654c2.18 0 3.904.548 5.125 1.628 1.286 1.14 1.892 2.713 1.801 4.675-.04.877-.23 1.727-.563 2.526-.341.819-.788 1.544-1.328 2.155-.504.57-1.065 1.04-1.667 1.396-.533.315-1.189.585-1.952.803-.764.218-1.594.326-2.466.326H9.116a.77.77 0 0 0-.76.65l-.98 5.88a.641.641 0 0 1-.633.54l-.667-.312z"/>
              </svg>
              <div>
                <span class="text-sm font-bold">PayPal</span>
                <div class="text-[10px] opacity-70">Global</div>
              </div>
            </button>
          </div>

          {/* Form Area */}
          <div class="flex-1 p-6 overflow-y-auto custom-scrollbar">
            <form onSubmit={handleAddSubmit} class="space-y-5">
              
              {/* Common Fields */}
              <div>
                <label class="block text-xs font-mono text-gray-500 uppercase mb-1.5 font-bold">Account Nickname</label>
                <input 
                  type="text" 
                  placeholder="e.g. Primary Checking"
                  class="w-full bg-terminal-950 border border-terminal-700 rounded-lg px-4 py-2.5 text-white text-sm focus:border-accent-500 focus:ring-1 focus:ring-accent-500 focus:outline-none transition-all placeholder-terminal-700"
                  onInput={(e) => updateField('name', e.currentTarget.value)}
                />
              </div>

              {/* Bank Fields - Using Plaid Link for Secure Connection */}
              <Show when={addType() === 'bank_account'}>
                <div class="p-4 bg-accent-900/10 border border-accent-500/30 rounded-lg">
                  <div class="flex items-center gap-2 mb-3">
                    <ShieldCheck size={16} class="text-accent-400" />
                    <span class="text-xs text-accent-400 font-bold">SECURE BANK CONNECTION</span>
                  </div>
                  <p class="text-xs text-gray-400 mb-4">
                    Connect your bank securely via Plaid. Your login credentials are never shared with us.
                    Supports instant verification for most banks.
                  </p>
                  
                  <button
                    type="button"
                    class="w-full py-3 px-4 bg-terminal-800 hover:bg-terminal-700 border border-terminal-600 rounded-lg flex items-center justify-center gap-3 transition-all"
                    onClick={handleConnectPlaid}
                  >
                    <Building2 size={20} class="text-accent-400" />
                    <span class="text-sm font-bold text-white">Connect Bank with Plaid</span>
                  </button>
                  
                  <div class="mt-3 text-[10px] text-gray-500 flex items-center gap-1">
                    <Lock size={10} />
                    Powered by Plaid • Bank-level Security • Instant Verification
                  </div>
                </div>
                
                {/* Supported Banks */}
                <div class="flex flex-wrap gap-2 mt-4">
                  {['Chase', 'Bank of America', 'Wells Fargo', 'Citi', 'Capital One', '+ 12,000 more'].map(bank => (
                    <span class="px-2 py-1 bg-terminal-800 border border-terminal-700 rounded text-[10px] text-gray-400">{bank}</span>
                  ))}
                </div>
              </Show>

              {/* Card Fields - Using Stripe Elements for PCI Compliance */}
              <Show when={addType() === 'debit_card'}>
                <div class="p-4 bg-accent-900/10 border border-accent-500/30 rounded-lg">
                  <div class="flex items-center gap-2 mb-3">
                    <ShieldCheck size={16} class="text-accent-400" />
                    <span class="text-xs text-accent-400 font-bold">SECURE CARD ENTRY</span>
                  </div>
                  <p class="text-xs text-gray-400 mb-4">
                    Card data is collected securely via Stripe. Your card number never touches our servers.
                    Browser autofill is fully supported.
                  </p>
                  
                  {/* Stripe Elements Container */}
                  <div id="card-element" class="bg-terminal-950 border border-terminal-700 rounded-lg p-4 min-h-[44px]">
                    {/* Stripe Elements will be mounted here */}
                    <div class="text-xs text-gray-500 animate-pulse">
                      Loading secure card form...
                    </div>
                  </div>
                  
                  <div class="mt-3 text-[10px] text-gray-500 flex items-center gap-1">
                    <Lock size={10} />
                    Powered by Stripe • PCI-DSS Level 1 Compliant • 3D Secure Ready
                  </div>
                </div>
                
                {/* Note about Stripe initialization */}
                <div class="p-3 bg-terminal-800 rounded-lg border border-terminal-700">
                  <p class="text-xs text-gray-400">
                    <strong class="text-white">To enable card payments:</strong><br/>
                    1. Add Stripe publishable key to your environment<br/>
                    2. Initialize Stripe Elements on page load<br/>
                    3. Card autofill will work automatically with browser settings
                  </p>
                </div>
              </Show>

              {/* Crypto Fields */}
              <Show when={addType() === 'crypto_wallet'}>
                <div>
                  <label class="block text-xs font-mono text-gray-500 uppercase mb-1.5 font-bold">Network</label>
                  <div class="relative">
                    <select 
                      class="w-full bg-terminal-950 border border-terminal-700 rounded-lg px-4 py-2.5 text-white text-sm focus:border-accent-500 focus:ring-1 focus:ring-accent-500 focus:outline-none transition-all appearance-none cursor-pointer"
                      onChange={(e) => updateField('crypto_network', e.currentTarget.value)}
                    >
                      <option value="btc">Bitcoin (BTC)</option>
                      <option value="eth">Ethereum (ERC-20)</option>
                      <option value="sol">Solana (SOL)</option>
                    </select>
                    <div class="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-gray-500">
                      <MoreVertical size={14} class="rotate-90" />
                    </div>
                  </div>
                </div>
                <div>
                  <label class="block text-xs font-mono text-gray-500 uppercase mb-1.5 font-bold">Wallet Address</label>
                  <input 
                    type="text" 
                    required
                    placeholder="0x..."
                    class="w-full bg-terminal-950 border border-terminal-700 rounded-lg px-4 py-2.5 text-white text-sm focus:border-accent-500 focus:ring-1 focus:ring-accent-500 focus:outline-none transition-all placeholder-terminal-700 font-mono"
                    onInput={(e) => updateField('crypto_address', e.currentTarget.value)}
                  />
                </div>
              </Show>

              {/* M-Pesa Fields */}
              <Show when={addType() === 'mpesa'}>
                <div class="p-4 bg-green-900/10 border border-green-500/30 rounded-lg">
                  <div class="flex items-center gap-2 mb-3">
                    <ShieldCheck size={16} class="text-green-400" />
                    <span class="text-xs text-green-400 font-bold">M-PESA MOBILE MONEY</span>
                  </div>
                  <p class="text-xs text-gray-400 mb-4">
                    Link your M-Pesa account for instant deposits via STK Push. Withdrawals sent directly to your M-Pesa.
                  </p>
                </div>
                
                <div>
                  <label class="block text-xs font-mono text-gray-500 uppercase mb-1.5 font-bold">Phone Number</label>
                  <div class="flex gap-2">
                    <select 
                      class="w-24 bg-terminal-950 border border-terminal-700 rounded-lg px-3 py-2.5 text-white text-sm focus:border-green-500 focus:ring-1 focus:ring-green-500 focus:outline-none transition-all"
                      onChange={(e) => updateField('country_code', e.currentTarget.value)}
                    >
                      <option value="+254">🇰🇪 +254</option>
                      <option value="+255">🇹🇿 +255</option>
                      <option value="+256">🇺🇬 +256</option>
                      <option value="+250">🇷🇼 +250</option>
                    </select>
                    <input 
                      type="tel" 
                      required
                      placeholder="7XXXXXXXX"
                      class="flex-1 bg-terminal-950 border border-terminal-700 rounded-lg px-4 py-2.5 text-white text-sm focus:border-green-500 focus:ring-1 focus:ring-green-500 focus:outline-none transition-all placeholder-terminal-700 font-mono"
                      onInput={(e) => updateField('phone_number', e.currentTarget.value)}
                    />
                  </div>
                  <p class="text-[10px] text-gray-500 mt-1">Enter without leading 0 (e.g., 712345678)</p>
                </div>
                
                <div class="p-3 bg-terminal-800 rounded-lg border border-terminal-700">
                  <h5 class="text-xs font-bold text-white mb-2">How M-Pesa works:</h5>
                  <ol class="text-xs text-gray-400 space-y-1 list-decimal list-inside">
                    <li>Enter deposit amount in the app</li>
                    <li>Receive STK Push on your phone</li>
                    <li>Enter M-Pesa PIN to confirm</li>
                    <li>Funds credited instantly</li>
                  </ol>
                </div>
              </Show>

              {/* PayPal Fields */}
              <Show when={addType() === 'paypal'}>
                <div class="p-4 bg-blue-900/10 border border-blue-500/30 rounded-lg">
                  <div class="flex items-center gap-2 mb-3">
                    <ShieldCheck size={16} class="text-blue-400" />
                    <span class="text-xs text-blue-400 font-bold">PAYPAL SECURE LINK</span>
                  </div>
                  <p class="text-xs text-gray-400 mb-4">
                    Connect your PayPal account for global payments. Supports credit cards linked to your PayPal.
                  </p>
                </div>
                
                <div>
                  <label class="block text-xs font-mono text-gray-500 uppercase mb-1.5 font-bold">PayPal Email</label>
                  <input 
                    type="email" 
                    required
                    placeholder="your@email.com"
                    class="w-full bg-terminal-950 border border-terminal-700 rounded-lg px-4 py-2.5 text-white text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 focus:outline-none transition-all placeholder-terminal-700"
                    onInput={(e) => updateField('paypal_email', e.currentTarget.value)}
                  />
                </div>
                
                <div class="p-3 bg-terminal-800 rounded-lg border border-terminal-700">
                  <h5 class="text-xs font-bold text-white mb-2">How PayPal deposits work:</h5>
                  <ol class="text-xs text-gray-400 space-y-1 list-decimal list-inside">
                    <li>Enter deposit amount in the app</li>
                    <li>Redirected to PayPal for approval</li>
                    <li>Log in and confirm payment</li>
                    <li>Funds credited after capture</li>
                  </ol>
                </div>
                
                <div class="flex items-center gap-2 p-3 bg-amber-900/10 border border-amber-500/30 rounded-lg">
                  <AlertCircle size={14} class="text-amber-400 shrink-0" />
                  <p class="text-xs text-amber-400">PayPal charges fees. Check rates before depositing.</p>
                </div>
              </Show>

              <Show when={addError()}>
                <div class="p-4 bg-danger-900/20 border border-danger-900/50 rounded-lg flex items-center gap-3 text-danger-400 text-xs animate-in fade-in slide-in-from-top-2">
                  <AlertCircle size={16} class="shrink-0" />
                  {addError()}
                </div>
              </Show>

              <div class="pt-6 border-t border-terminal-700 flex items-center justify-between">
                <div class="flex items-center gap-2 text-xs text-gray-500">
                  <Lock size={12} />
                  Encrypted & Secure
                </div>
                <button 
                  type="submit"
                  disabled={adding()}
                  class={`px-6 py-2.5 bg-accent-600 hover:bg-accent-500 text-white rounded-lg text-sm font-bold uppercase tracking-wider transition-all shadow-lg shadow-accent-900/20 hover:shadow-accent-900/40 hover:-translate-y-0.5 ${adding() ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  {adding() ? 'Adding...' : 'Add Method'}
                </button>
              </div>
            </form>
          </div>
        </div>
      </Modal>
    </div>
  );
}
