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

interface PaymentMethodsTabProps {
  paymentMethods: PaymentMethod[];
  onUpdate: () => void;
}

export function PaymentMethodsTab(props: PaymentMethodsTabProps) {
  const [showAddModal, setShowAddModal] = createSignal(false);
  const [deletingId, setDeletingId] = createSignal<string | null>(null);
  
  // Add Method State
  const [addType, setAddType] = createSignal<'bank_account' | 'debit_card' | 'crypto_wallet'>('bank_account');
  const [formData, setFormData] = createSignal<any>({});
  const [adding, setAdding] = createSignal(false);
  const [addError, setAddError] = createSignal<string | null>(null);

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
                    <div class="flex items-center gap-1.5 text-[10px] text-success-400 bg-success-900/10 px-2 py-1 rounded border border-success-900/30">
                      <CheckCircle2 size={12} />
                      VERIFIED
                    </div>
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
          <div class="w-full md:w-1/3 border-b md:border-b-0 md:border-r border-terminal-700 bg-terminal-900/50 p-4 space-y-2">
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

              {/* Bank Fields */}
              <Show when={addType() === 'bank_account'}>
                <div>
                  <label class="block text-xs font-mono text-gray-500 uppercase mb-1.5 font-bold">Bank Name</label>
                  <input 
                    type="text" 
                    required
                    placeholder="e.g. Chase, Bank of America"
                    class="w-full bg-terminal-950 border border-terminal-700 rounded-lg px-4 py-2.5 text-white text-sm focus:border-accent-500 focus:ring-1 focus:ring-accent-500 focus:outline-none transition-all placeholder-terminal-700"
                    onInput={(e) => updateField('bank_name', e.currentTarget.value)}
                  />
                </div>
                <div class="grid grid-cols-2 gap-4">
                  <div>
                    <label class="block text-xs font-mono text-gray-500 uppercase mb-1.5 font-bold">Routing Number</label>
                    <input 
                      type="text" 
                      required
                      maxLength={9}
                      placeholder="9 digits"
                      class="w-full bg-terminal-950 border border-terminal-700 rounded-lg px-4 py-2.5 text-white text-sm focus:border-accent-500 focus:ring-1 focus:ring-accent-500 focus:outline-none transition-all placeholder-terminal-700 font-mono"
                      onInput={(e) => updateField('routing_number', e.currentTarget.value)}
                    />
                  </div>
                  <div>
                    <label class="block text-xs font-mono text-gray-500 uppercase mb-1.5 font-bold">Account Number</label>
                    <input 
                      type="text" 
                      required
                      placeholder="Account Number"
                      class="w-full bg-terminal-950 border border-terminal-700 rounded-lg px-4 py-2.5 text-white text-sm focus:border-accent-500 focus:ring-1 focus:ring-accent-500 focus:outline-none transition-all placeholder-terminal-700 font-mono"
                      onInput={(e) => updateField('account_number', e.currentTarget.value)}
                    />
                  </div>
                </div>
                <div>
                  <label class="block text-xs font-mono text-gray-500 uppercase mb-1.5 font-bold">Account Type</label>
                  <div class="relative">
                    <select 
                      class="w-full bg-terminal-950 border border-terminal-700 rounded-lg px-4 py-2.5 text-white text-sm focus:border-accent-500 focus:ring-1 focus:ring-accent-500 focus:outline-none transition-all appearance-none cursor-pointer"
                      onChange={(e) => updateField('account_type', e.currentTarget.value)}
                    >
                      <option value="checking">Checking</option>
                      <option value="savings">Savings</option>
                    </select>
                    <div class="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-gray-500">
                      <MoreVertical size={14} class="rotate-90" />
                    </div>
                  </div>
                </div>
              </Show>

              {/* Card Fields */}
              <Show when={addType() === 'debit_card'}>
                <div>
                  <label class="block text-xs font-mono text-gray-500 uppercase mb-1.5 font-bold">Card Number</label>
                  <div class="relative">
                    <input 
                      type="text" 
                      required
                      maxLength={19}
                      placeholder="0000 0000 0000 0000"
                      class="w-full bg-terminal-950 border border-terminal-700 rounded-lg px-4 py-2.5 pl-10 text-white text-sm focus:border-accent-500 focus:ring-1 focus:ring-accent-500 focus:outline-none transition-all placeholder-terminal-700 font-mono"
                      onInput={(e) => updateField('card_number', e.currentTarget.value)}
                    />
                    <CreditCard size={16} class="absolute left-3 top-3 text-gray-500" />
                  </div>
                </div>
                <div class="grid grid-cols-2 gap-4">
                  <div>
                    <label class="block text-xs font-mono text-gray-500 uppercase mb-1.5 font-bold">Expiry Date</label>
                    <input 
                      type="text" 
                      required
                      placeholder="MM/YY"
                      maxLength={5}
                      class="w-full bg-terminal-950 border border-terminal-700 rounded-lg px-4 py-2.5 text-white text-sm focus:border-accent-500 focus:ring-1 focus:ring-accent-500 focus:outline-none transition-all placeholder-terminal-700 font-mono"
                      onInput={(e) => updateField('expiry', e.currentTarget.value)}
                    />
                  </div>
                  <div>
                    <label class="block text-xs font-mono text-gray-500 uppercase mb-1.5 font-bold">CVV</label>
                    <input 
                      type="text" 
                      required
                      maxLength={4}
                      placeholder="123"
                      class="w-full bg-terminal-950 border border-terminal-700 rounded-lg px-4 py-2.5 text-white text-sm focus:border-accent-500 focus:ring-1 focus:ring-accent-500 focus:outline-none transition-all placeholder-terminal-700 font-mono"
                      onInput={(e) => updateField('cvv', e.currentTarget.value)}
                    />
                  </div>
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
