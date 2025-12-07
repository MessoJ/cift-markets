/**
 * ACCOUNT FUNDING PAGE
 * Professional funding interface with comprehensive drill-downs
 * Matches the high-density "Bloomberg" aesthetic of the Transactions page.
 */

import { createSignal, createEffect, Show, For } from 'solid-js';
import { 
  DollarSign, 
  ArrowUpRight, 
  ArrowDownRight, 
  CreditCard, 
  Clock, 
  AlertCircle, 
  Zap,
  Wallet,
  Activity,
  Landmark,
  FileText
} from 'lucide-solid';
import { apiClient, FundingTransaction, PaymentMethod, TransferLimit, PortfolioSummary } from '../../lib/api/client';
import { formatCurrency } from '../../lib/utils';
import { DepositTab } from './tabs/DepositTab';
import { WithdrawTab } from './tabs/WithdrawTab';
import { HistoryTab } from './tabs/HistoryTab';
import { PaymentMethodsTab } from './tabs/PaymentMethodsTab';
import { StatementsTab } from './tabs/StatementsTab';

type TabType = 'deposit' | 'withdraw' | 'history' | 'methods' | 'statements';

// Summary Card Component matching TransactionsPage
const SummaryCard = (props: { title: string; value: number; type: 'cash' | 'power' | 'pending' | 'limit'; subtext?: string }) => (
  <div class="bg-terminal-900 border border-terminal-750 p-4 rounded-sm flex-1 min-w-[200px]">
    <div class="flex items-center justify-between mb-2">
      <span class="text-xs font-mono text-gray-400 uppercase tracking-wider">{props.title}</span>
      <div class={`p-1.5 rounded-full ${
        props.type === 'cash' ? 'bg-success-900/20 text-success-400' :
        props.type === 'power' ? 'bg-accent-900/20 text-accent-400' :
        props.type === 'pending' ? 'bg-warning-900/20 text-warning-400' :
        'bg-terminal-800 text-gray-400'
      }`}>
        {props.type === 'cash' && <Wallet class="w-4 h-4" />}
        {props.type === 'power' && <Zap class="w-4 h-4" />}
        {props.type === 'pending' && <Clock class="w-4 h-4" />}
        {props.type === 'limit' && <Landmark class="w-4 h-4" />}
      </div>
    </div>
    <div class={`text-xl font-mono font-bold ${
      props.type === 'cash' ? 'text-success-400' :
      props.type === 'power' ? 'text-accent-400' :
      props.type === 'pending' ? 'text-warning-400' :
      'text-white'
    }`}>
      {formatCurrency(props.value)}
    </div>
    <Show when={props.subtext}>
      <div class="text-[10px] text-gray-500 mt-1 font-mono">
        {props.subtext}
      </div>
    </Show>
  </div>
);

export default function FundingPage() {
  const [activeTab, setActiveTab] = createSignal<TabType>('deposit');
  const [loading, setLoading] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [transactions, setTransactions] = createSignal<FundingTransaction[]>([]);
  const [paymentMethods, setPaymentMethods] = createSignal<PaymentMethod[]>([]);
  const [limits, setLimits] = createSignal<TransferLimit | null>(null);
  const [portfolio, setPortfolio] = createSignal<PortfolioSummary | null>(null);

  createEffect(() => {
    loadData();
  });

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [txns, methods, transferLimits, portfolioData] = await Promise.all([
        apiClient.getFundingTransactions({ limit: 50 }),
        apiClient.getPaymentMethods(),
        apiClient.getTransferLimits(),
        apiClient.getPortfolio(),
      ]);
      setTransactions(txns.transactions);
      setPaymentMethods(methods);
      setLimits(transferLimits);
      setPortfolio(portfolioData);
    } catch (err: any) {
      setError(err.message || 'Failed to load funding data');
    } finally {
      setLoading(false);
    }
  };

  // Calculate pending deposits
  const pendingDeposits = () => {
    return transactions()
      .filter(t => t.status === 'processing' || t.status === 'pending')
      .filter(t => t.type === 'deposit')
      .reduce((sum, t) => {
        const val = typeof t.amount === 'string' ? parseFloat(t.amount) : t.amount;
        return sum + (isNaN(val) ? 0 : val);
      }, 0);
  };

  // Calculate total deposited (lifetime)
  const totalDeposited = () => {
    return transactions()
      .filter(t => t.status === 'completed' && t.type === 'deposit')
      .reduce((sum, t) => {
        const val = typeof t.amount === 'string' ? parseFloat(t.amount) : t.amount;
        return sum + (isNaN(val) ? 0 : val);
      }, 0);
  };

  return (
    <div class="flex flex-col gap-4 p-2 md:p-4 bg-black text-white overflow-hidden md:h-full min-h-0">
      {/* Header Section */}
      <div class="flex flex-col gap-4">
        <div class="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 bg-accent-500/10 rounded-sm flex items-center justify-center border border-accent-500/20">
              <DollarSign class="text-accent-500" size={20} />
            </div>
            <div>
              <h1 class="text-xl font-bold font-mono tracking-tight text-white">FUNDING & TRANSFERS</h1>
              <p class="text-xs text-gray-500 font-mono uppercase tracking-wider">Manage liquidity and payment methods</p>
            </div>
          </div>
          
          {/* Quick Actions */}
          <div class="flex gap-3">
            <button 
              onClick={() => setActiveTab('deposit')}
              class="flex items-center gap-2 px-4 py-2 bg-accent-600 hover:bg-accent-500 text-white rounded-sm text-xs font-bold font-mono uppercase tracking-wider transition-colors shadow-lg shadow-accent-900/20"
            >
              <ArrowDownRight size={16} />
              Deposit
            </button>
            <button 
              onClick={() => setActiveTab('withdraw')}
              class="flex items-center gap-2 px-4 py-2 bg-terminal-800 hover:bg-terminal-700 text-white border border-terminal-600 rounded-sm text-xs font-bold font-mono uppercase tracking-wider transition-colors"
            >
              <ArrowUpRight size={16} />
              Withdraw
            </button>
          </div>
        </div>

        {/* Summary Cards */}
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <SummaryCard 
            title="AVAILABLE CASH" 
            value={portfolio()?.cash || 0} 
            type="cash" 
            subtext="Settled funds available for withdrawal"
          />
          <SummaryCard 
            title="BUYING POWER" 
            value={portfolio()?.buying_power || 0} 
            type="power" 
            subtext="Total margin buying power"
          />
          <SummaryCard 
            title="PENDING DEPOSITS" 
            value={pendingDeposits()} 
            type="pending" 
            subtext="Funds clearing in 1-3 days"
          />
          <SummaryCard 
            title="DAILY DEPOSIT LIMIT" 
            value={limits()?.daily_deposit_remaining || 0} 
            type="limit" 
            subtext={`of ${formatCurrency(limits()?.daily_deposit_limit || 0)} remaining`}
          />
        </div>
        
        {/* Lifetime Stats Bar */}
        <div class="pt-2 border-t border-terminal-800 flex items-center gap-8 text-[10px] font-mono text-gray-500 uppercase tracking-wider">
          <div class="flex items-center gap-2">
            <Activity size={12} />
            <span>LIFETIME DEPOSITS: <span class="text-white">{formatCurrency(totalDeposited())}</span></span>
          </div>
          <div class="flex items-center gap-2">
            <CreditCard size={12} />
            <span>ACTIVE METHODS: <span class="text-white">{paymentMethods().length}</span></span>
          </div>
          <div class="flex-1"></div>
          <div class="flex items-center gap-2 text-accent-400">
            <Zap size={12} />
            <span>INSTANT DEPOSITS ENABLED</span>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div class="flex-1 flex flex-col min-h-0 bg-terminal-900 border border-terminal-750 rounded-sm overflow-hidden">
        {/* Tabs */}
        <div class="flex border-b border-terminal-750 bg-terminal-900 overflow-x-auto">
          <For each={[
            { id: 'deposit', label: 'DEPOSIT FUNDS', icon: ArrowDownRight },
            { id: 'withdraw', label: 'WITHDRAW FUNDS', icon: ArrowUpRight },
            { id: 'methods', label: 'PAYMENT METHODS', icon: CreditCard },
            { id: 'history', label: 'TRANSACTION HISTORY', icon: Clock },
            { id: 'statements', label: 'STATEMENTS', icon: FileText }
          ]}>
            {(tab) => (
              <button
                onClick={() => setActiveTab(tab.id as TabType)}
                class={`
                  flex items-center gap-2 px-6 py-4 text-xs font-bold font-mono tracking-wider transition-colors border-r border-terminal-750 shrink-0
                  ${activeTab() === tab.id 
                    ? 'bg-terminal-800 text-accent-400 border-b-2 border-b-accent-500' 
                    : 'text-gray-500 hover:text-gray-300 hover:bg-terminal-800/50 border-b-2 border-b-transparent'}
                `}
              >
                <tab.icon size={14} />
                {tab.label}
              </button>
            )}
          </For>
        </div>

        {/* Tab Content */}
        <div class="flex-1 overflow-hidden p-0 relative">
          <Show when={activeTab() === 'deposit'}>
            <div class="h-full overflow-auto p-3 md:p-6">
              <DepositTab 
                paymentMethods={paymentMethods()} 
                limits={limits()} 
                onSuccess={loadData} 
              />
            </div>
          </Show>
          
          <Show when={activeTab() === 'withdraw'}>
            <div class="h-full overflow-auto p-3 md:p-6">
              <WithdrawTab 
                paymentMethods={paymentMethods()} 
                portfolio={portfolio()} 
                onSuccess={loadData} 
              />
            </div>
          </Show>
          
          <Show when={activeTab() === 'methods'}>
            <div class="h-full overflow-auto p-3 md:p-6">
              <PaymentMethodsTab 
                paymentMethods={paymentMethods()} 
                onUpdate={loadData} 
              />
            </div>
          </Show>
          
          <Show when={activeTab() === 'history'}>
            <div class="h-full overflow-hidden">
              <HistoryTab transactions={transactions()} />
            </div>
          </Show>

          <Show when={activeTab() === 'statements'}>
            <div class="h-full overflow-hidden p-3 md:p-6">
              <StatementsTab />
            </div>
          </Show>
        </div>
      </div>
    </div>
  );
}
