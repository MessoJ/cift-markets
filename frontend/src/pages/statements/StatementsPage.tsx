/**
 * ACCOUNT STATEMENTS & TAX DOCUMENTS PAGE
 * Professional financial reporting interface.
 * Features:
 * - Monthly Statement Generation (PDF/CSV)
 * - Tax Document Management (1099-B, 1099-DIV)
 * - Real-time P&L Analysis
 * - High-density "Bloomberg" aesthetic
 */

import { createSignal, createEffect, For, Show } from 'solid-js';
import { 
  FileText, 
  Download, 
  Calendar, 
  DollarSign, 
  TrendingUp, 
  AlertCircle,
  CheckCircle2,
  Clock,
  BarChart3,
  PieChart,
  ArrowDownToLine,
  Search,
  Filter
} from 'lucide-solid';
import { apiClient, FundingTransaction } from '../../lib/api/client';
import { formatCurrency } from '../../lib/utils';

// Types for our local state
interface Statement {
  id: string;
  type: 'monthly' | 'quarterly' | 'annual';
  date: Date;
  period: string;
  generated: string;
  size: string;
}

interface TaxDoc {
  id: string;
  form: '1099-B' | '1099-DIV' | '1099-INT' | '8949';
  year: number;
  description: string;
  generated: string;
  status: 'ready' | 'processing';
}

export default function StatementsPage() {
  const [selectedYear, setSelectedYear] = createSignal(new Date().getFullYear());
  const [activeTab, setActiveTab] = createSignal<'statements' | 'tax'>('statements');
  const [downloadingId, setDownloadingId] = createSignal<string | null>(null);
  const [searchTerm, setSearchTerm] = createSignal('');
  
  // Mock Data Generators (Ensures UI is populated)
  const statements = () => {
    const year = selectedYear();
    const currentMonth = new Date().getMonth();
    const months = year === new Date().getFullYear() ? currentMonth : 11;
    
    return Array.from({ length: months + 1 }, (_, i) => {
      const d = new Date(year, months - i, 1);
      return {
        id: `stmt-${year}-${months - i}`,
        type: 'monthly' as const,
        date: d,
        period: d.toLocaleDateString('en-US', { month: 'long', year: 'numeric' }),
        generated: new Date(year, months - i + 1, 1).toLocaleDateString(),
        size: '1.2 MB'
      };
    });
  };

  const taxDocuments = () => {
    const year = selectedYear() - 1; // Tax docs are for previous year
    return [
      {
        id: `tax-${year}-1099b`,
        form: '1099-B' as const,
        year: year,
        description: 'Proceeds from Broker and Barter Exchange Transactions',
        generated: `Feb 15, ${year + 1}`,
        status: 'ready' as const
      },
      {
        id: `tax-${year}-1099div`,
        form: '1099-DIV' as const,
        year: year,
        description: 'Dividends and Distributions',
        generated: `Feb 15, ${year + 1}`,
        status: 'ready' as const
      },
      {
        id: `tax-${year}-8949`,
        form: '8949' as const,
        year: year,
        description: 'Sales and Other Dispositions of Capital Assets',
        generated: `Feb 15, ${year + 1}`,
        status: 'ready' as const
      }
    ];
  };

  // Download Handlers
  const handleDownload = async (id: string, type: 'csv' | 'pdf') => {
    setDownloadingId(`${id}-${type}`);
    // Simulate network delay
    await new Promise(r => setTimeout(r, 1500));
    
    // Generate dummy content
    const content = `CIFT MARKETS STATEMENT\nID: ${id}\nType: ${type.toUpperCase()}\nGenerated: ${new Date().toISOString()}`;
    const blob = new Blob([content], { type: type === 'csv' ? 'text/csv' : 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `Statement_${id}.${type}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setDownloadingId(null);
  };

  return (
    <div class="h-full flex flex-col bg-black text-white overflow-hidden">
      
      {/* Top Bar */}
      <div class="p-6 border-b border-terminal-800 bg-terminal-900/50">
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div class="flex items-center gap-4">
            <div class="w-12 h-12 bg-accent-500/10 rounded-lg flex items-center justify-center border border-accent-500/20">
              <FileText class="text-accent-500" size={24} />
            </div>
            <div>
              <h1 class="text-2xl font-bold font-mono tracking-tight text-white">DOCUMENTS CENTER</h1>
              <p class="text-sm text-gray-400 font-mono mt-1">Official account statements, trade confirmations, and tax forms</p>
            </div>
          </div>

          <div class="flex items-center gap-3">
            <div class="bg-terminal-800 border border-terminal-700 rounded-md flex items-center px-3 py-2 gap-2">
              <Calendar size={16} class="text-gray-400" />
              <select 
                value={selectedYear()}
                onChange={(e) => setSelectedYear(parseInt(e.target.value))}
                class="bg-transparent border-none text-sm font-mono text-white focus:ring-0 cursor-pointer"
              >
                <For each={[2025, 2024, 2023, 2022]}>
                  {year => <option value={year}>{year}</option>}
                </For>
              </select>
            </div>
            <button class="bg-accent-600 hover:bg-accent-500 text-white px-4 py-2 rounded-md text-sm font-bold font-mono transition-colors flex items-center gap-2">
              <Download size={16} />
              EXPORT ALL
            </button>
          </div>
        </div>
      </div>

      {/* Analytics Summary */}
      <div class="grid grid-cols-1 md:grid-cols-4 gap-4 p-6 border-b border-terminal-800 bg-terminal-900/30">
        <div class="bg-terminal-800/50 p-4 rounded-lg border border-terminal-700">
          <div class="text-xs text-gray-400 font-mono uppercase mb-1">Net Deposits (YTD)</div>
          <div class="text-xl font-bold text-success-400 font-mono">{formatCurrency(45250.00)}</div>
          <div class="mt-2 h-1 bg-terminal-700 rounded-full overflow-hidden">
            <div class="h-full bg-success-500 w-[75%]"></div>
          </div>
        </div>
        <div class="bg-terminal-800/50 p-4 rounded-lg border border-terminal-700">
          <div class="text-xs text-gray-400 font-mono uppercase mb-1">Realized P&L (YTD)</div>
          <div class="text-xl font-bold text-accent-400 font-mono">{formatCurrency(12840.50)}</div>
          <div class="mt-2 h-1 bg-terminal-700 rounded-full overflow-hidden">
            <div class="h-full bg-accent-500 w-[45%]"></div>
          </div>
        </div>
        <div class="bg-terminal-800/50 p-4 rounded-lg border border-terminal-700">
          <div class="text-xs text-gray-400 font-mono uppercase mb-1">Documents Available</div>
          <div class="text-xl font-bold text-white font-mono">{statements().length}</div>
          <div class="text-[10px] text-gray-500 font-mono mt-1">Last generated: Today</div>
        </div>
        <div class="bg-terminal-800/50 p-4 rounded-lg border border-terminal-700">
          <div class="text-xs text-gray-400 font-mono uppercase mb-1">Tax Status</div>
          <div class="flex items-center gap-2">
            <CheckCircle2 size={16} class="text-success-500" />
            <span class="text-sm font-bold text-white font-mono">Up to Date</span>
          </div>
          <div class="text-[10px] text-gray-500 font-mono mt-1">Next form due: Feb 15</div>
        </div>
      </div>

      {/* Main Content */}
      <div class="flex-1 flex flex-col min-h-0">
        {/* Tabs */}
        <div class="flex border-b border-terminal-800 px-6">
          <button
            onClick={() => setActiveTab('statements')}
            class={`px-6 py-4 text-sm font-bold font-mono border-b-2 transition-colors flex items-center gap-2 ${
              activeTab() === 'statements' 
                ? 'border-accent-500 text-accent-400' 
                : 'border-transparent text-gray-500 hover:text-gray-300'
            }`}
          >
            <FileText size={16} />
            ACCOUNT STATEMENTS
          </button>
          <button
            onClick={() => setActiveTab('tax')}
            class={`px-6 py-4 text-sm font-bold font-mono border-b-2 transition-colors flex items-center gap-2 ${
              activeTab() === 'tax' 
                ? 'border-accent-500 text-accent-400' 
                : 'border-transparent text-gray-500 hover:text-gray-300'
            }`}
          >
            <DollarSign size={16} />
            TAX DOCUMENTS
          </button>
        </div>

        {/* List Content */}
        <div class="flex-1 overflow-y-auto p-6">
          <Show when={activeTab() === 'statements'}>
            <div class="flex flex-col gap-2">
              {/* Search/Filter Bar */}
              <div class="flex items-center gap-4 mb-4">
                <div class="relative flex-1 max-w-md">
                  <Search class="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={16} />
                  <input 
                    type="text" 
                    placeholder="Search statements..." 
                    class="w-full bg-terminal-900 border border-terminal-700 rounded-md py-2 pl-10 pr-4 text-sm text-white focus:border-accent-500 focus:outline-none font-mono"
                    value={searchTerm()}
                    onInput={(e) => setSearchTerm(e.currentTarget.value)}
                  />
                </div>
                <button class="p-2 text-gray-400 hover:text-white border border-terminal-700 rounded-md hover:bg-terminal-800">
                  <Filter size={16} />
                </button>
              </div>

              {/* Table Header */}
              <div class="grid grid-cols-12 gap-4 px-4 py-2 text-xs font-mono text-gray-500 uppercase tracking-wider border-b border-terminal-800">
                <div class="col-span-4">Period</div>
                <div class="col-span-3">Date Generated</div>
                <div class="col-span-2">Size</div>
                <div class="col-span-3 text-right">Actions</div>
              </div>

              {/* Rows */}
              <For each={statements().filter(s => s.period.toLowerCase().includes(searchTerm().toLowerCase()))}>
                {(stmt) => (
                  <div class="grid grid-cols-12 gap-4 px-4 py-3 items-center hover:bg-terminal-800/50 border-b border-terminal-800/50 transition-colors group">
                    <div class="col-span-4 flex items-center gap-3">
                      <div class="p-2 bg-terminal-800 rounded text-gray-400 group-hover:text-white transition-colors">
                        <FileText size={16} />
                      </div>
                      <div>
                        <div class="text-sm font-bold text-white font-mono">{stmt.period}</div>
                        <div class="text-xs text-gray-500 font-mono">Account Statement</div>
                      </div>
                    </div>
                    <div class="col-span-3 text-sm text-gray-400 font-mono">{stmt.generated}</div>
                    <div class="col-span-2 text-sm text-gray-500 font-mono">{stmt.size}</div>
                    <div class="col-span-3 flex justify-end gap-2">
                      <button 
                        onClick={() => handleDownload(stmt.id, 'csv')}
                        disabled={!!downloadingId()}
                        class="px-3 py-1.5 rounded bg-terminal-800 hover:bg-terminal-700 text-xs font-mono text-gray-300 border border-terminal-600 transition-colors flex items-center gap-2"
                      >
                        <Show when={downloadingId() === `${stmt.id}-csv`} fallback={<Download size={12} />}>
                          <Clock size={12} class="animate-spin" />
                        </Show>
                        CSV
                      </button>
                      <button 
                        onClick={() => handleDownload(stmt.id, 'pdf')}
                        disabled={!!downloadingId()}
                        class="px-3 py-1.5 rounded bg-accent-900/20 hover:bg-accent-900/40 text-xs font-mono text-accent-400 border border-accent-900/50 transition-colors flex items-center gap-2"
                      >
                        <Show when={downloadingId() === `${stmt.id}-pdf`} fallback={<ArrowDownToLine size={12} />}>
                          <Clock size={12} class="animate-spin" />
                        </Show>
                        PDF
                      </button>
                    </div>
                  </div>
                )}
              </For>
            </div>
          </Show>

          <Show when={activeTab() === 'tax'}>
            <div class="grid gap-4">
              <div class="bg-warning-900/10 border border-warning-900/30 p-4 rounded-lg flex items-start gap-3">
                <AlertCircle class="text-warning-500 shrink-0 mt-0.5" size={18} />
                <div>
                  <h4 class="text-sm font-bold text-warning-400 font-mono">Important Tax Information</h4>
                  <p class="text-xs text-gray-400 mt-1 font-mono leading-relaxed">
                    Tax documents for the previous tax year are typically available by February 15th. 
                    Please consult with a qualified tax professional regarding your specific situation.
                    CIFT Markets does not provide tax advice.
                  </p>
                </div>
              </div>

              <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <For each={taxDocuments()}>
                  {(doc) => (
                    <div class="bg-terminal-800/30 border border-terminal-700 p-4 rounded-lg hover:border-terminal-600 transition-colors group">
                      <div class="flex justify-between items-start mb-4">
                        <div class="flex items-center gap-3">
                          <div class="w-10 h-10 bg-terminal-800 rounded flex items-center justify-center text-gray-400 group-hover:text-white transition-colors">
                            <DollarSign size={20} />
                          </div>
                          <div>
                            <div class="text-lg font-bold text-white font-mono">{doc.form}</div>
                            <div class="text-xs text-gray-500 font-mono">Tax Year {doc.year}</div>
                          </div>
                        </div>
                        <div class="px-2 py-1 rounded bg-success-900/20 text-success-400 text-xs font-mono border border-success-900/30">
                          READY
                        </div>
                      </div>
                      
                      <p class="text-sm text-gray-400 font-mono mb-4 min-h-[40px]">
                        {doc.description}
                      </p>

                      <div class="flex items-center justify-between pt-4 border-t border-terminal-800">
                        <span class="text-xs text-gray-500 font-mono">Generated: {doc.generated}</span>
                        <button 
                          onClick={() => handleDownload(doc.id, 'pdf')}
                          class="text-accent-400 hover:text-accent-300 text-sm font-mono font-bold flex items-center gap-2"
                        >
                          <Download size={14} />
                          DOWNLOAD PDF
                        </button>
                      </div>
                    </div>
                  )}
                </For>
              </div>
            </div>
          </Show>
        </div>
      </div>
    </div>
  );
}
