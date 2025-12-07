import { createSignal, For, Show } from 'solid-js';
import { 
  FileText, 
  Download, 
  Calendar, 
  ChevronRight, 
  Loader2,
  CheckCircle2,
  AlertCircle
} from 'lucide-solid';
import { apiClient, FundingTransaction } from '../../../lib/api/client';
import { formatCurrency } from '../../../lib/utils';

export function StatementsTab() {
  const [downloadingId, setDownloadingId] = createSignal<string | null>(null);
  const [error, setError] = createSignal<string | null>(null);

  // Generate last 12 months
  const statements = Array.from({ length: 12 }, (_, i) => {
    const d = new Date();
    d.setMonth(d.getMonth() - i);
    // Set to first day of month for consistency
    d.setDate(1); 
    return {
      id: `${d.getFullYear()}-${d.getMonth() + 1}`,
      date: d,
      label: d.toLocaleDateString('en-US', { month: 'long', year: 'numeric' }),
      period: `${d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} - ${new Date(d.getFullYear(), d.getMonth() + 1, 0).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}`,
      generated: new Date(d.getFullYear(), d.getMonth() + 1, 1).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
    };
  });

  const handleDownload = async (statement: typeof statements[0], format: 'csv' | 'pdf') => {
    setDownloadingId(`${statement.id}-${format}`);
    setError(null);

    try {
      // Calculate start and end dates for the month
      const startDate = new Date(statement.date.getFullYear(), statement.date.getMonth(), 1).toISOString();
      const endDate = new Date(statement.date.getFullYear(), statement.date.getMonth() + 1, 0).toISOString();

      // Fetch transactions for the period
      const response = await apiClient.getFundingTransactions({
        start_date: startDate,
        end_date: endDate,
        limit: 1000 // Fetch all for the month
      });

      if (format === 'csv') {
        generateCSV(response.transactions, statement.label);
      } else {
        // For PDF, we would ideally call an API. 
        // Since we want to ensure success without a backend PDF generator, 
        // we'll simulate a PDF download or generate a simple text report.
        // For now, let's generate a detailed text report as a .txt file which is safer than a broken PDF.
        generateReport(response.transactions, statement.label);
      }

    } catch (err) {
      console.error('Download failed:', err);
      setError('Failed to download statement. Please try again.');
    } finally {
      setDownloadingId(null);
    }
  };

  const generateCSV = (transactions: FundingTransaction[], period: string) => {
    const headers = ['Date', 'Type', 'Method', 'Amount', 'Fee', 'Status', 'ID'];
    const rows = transactions.map(t => [
      new Date(t.created_at).toLocaleString(),
      t.type,
      t.method,
      t.amount,
      t.fee,
      t.status,
      t.id
    ]);

    const csvContent = [
      headers.join(','),
      ...rows.map(row => row.join(','))
    ].join('\n');

    downloadFile(csvContent, `Statement_${period.replace(' ', '_')}.csv`, 'text/csv');
  };

  const generateReport = (transactions: FundingTransaction[], period: string) => {
    const totalDeposits = transactions
      .filter(t => t.type === 'deposit' && t.status === 'completed')
      .reduce((sum, t) => sum + t.amount, 0);
      
    const totalWithdrawals = transactions
      .filter(t => t.type === 'withdrawal' && t.status === 'completed')
      .reduce((sum, t) => sum + t.amount, 0);

    const content = `
CIFT MARKETS - MONTHLY STATEMENT
Period: ${period}
Generated: ${new Date().toLocaleString()}

SUMMARY
----------------------------------------
Total Deposits:    ${formatCurrency(totalDeposits)}
Total Withdrawals: ${formatCurrency(totalWithdrawals)}
Net Flow:          ${formatCurrency(totalDeposits - totalWithdrawals)}
Transaction Count: ${transactions.length}

TRANSACTION DETAILS
----------------------------------------
${transactions.map(t => `
${new Date(t.created_at).toLocaleDateString()} | ${t.type.toUpperCase()} | ${formatCurrency(t.amount)}
Status: ${t.status} | Method: ${t.method} | ID: ${t.id}
`).join('')}
    `;

    downloadFile(content, `Statement_${period.replace(' ', '_')}.txt`, 'text/plain');
  };

  const downloadFile = (content: string, filename: string, type: string) => {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div class="bg-terminal-900 border border-terminal-750 h-full flex flex-col rounded-xl overflow-hidden">
      
      {/* Header */}
      <div class="p-6 border-b border-terminal-750 bg-terminal-800/30">
        <div class="flex items-center justify-between">
          <div>
            <h3 class="text-lg font-mono font-bold text-white flex items-center gap-2">
              <FileText class="text-accent-500" size={20} />
              Monthly Statements
            </h3>
            <p class="text-sm text-gray-400 mt-1 font-mono">
              View and download your monthly account activity reports.
            </p>
          </div>
          <div class="hidden md:block">
            <div class="bg-terminal-800 border border-terminal-700 rounded-lg p-3 flex items-center gap-3">
              <div class="bg-accent-900/20 p-2 rounded-md">
                <CheckCircle2 class="text-accent-400" size={16} />
              </div>
              <div>
                <div class="text-xs text-gray-400 font-mono uppercase">Account Status</div>
                <div class="text-sm font-bold text-white font-mono">Verified & Active</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Error Message */}
      <Show when={error()}>
        <div class="bg-danger-900/20 border-b border-danger-900/50 p-3 flex items-center gap-2 text-danger-400 text-sm font-mono">
          <AlertCircle size={14} />
          {error()}
        </div>
      </Show>

      {/* Statements List */}
      <div class="flex-1 overflow-y-auto p-4">
        <div class="grid gap-3">
          <For each={statements}>
            {(statement) => (
              <div class="bg-terminal-800/50 border border-terminal-750 rounded-lg p-4 hover:bg-terminal-800 transition-colors group">
                <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
                  
                  {/* Left: Info */}
                  <div class="flex items-start gap-4">
                    <div class="bg-terminal-700/50 p-3 rounded-lg group-hover:bg-terminal-700 transition-colors">
                      <Calendar class="text-gray-400 group-hover:text-white transition-colors" size={20} />
                    </div>
                    <div>
                      <h4 class="text-base font-bold text-white font-mono">{statement.label}</h4>
                      <div class="flex items-center gap-3 mt-1">
                        <span class="text-xs text-gray-500 font-mono">{statement.period}</span>
                        <span class="text-xs text-terminal-600">•</span>
                        <span class="text-xs text-gray-500 font-mono">Generated on {statement.generated}</span>
                      </div>
                    </div>
                  </div>

                  {/* Right: Actions */}
                  <div class="flex items-center gap-2">
                    <button 
                      onClick={() => handleDownload(statement, 'csv')}
                      disabled={!!downloadingId()}
                      class="flex items-center gap-2 px-3 py-2 rounded-md bg-terminal-800 border border-terminal-600 hover:bg-terminal-700 hover:border-terminal-500 text-xs font-mono text-gray-300 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <Show 
                        when={downloadingId() === `${statement.id}-csv`}
                        fallback={<Download size={14} />}
                      >
                        <Loader2 size={14} class="animate-spin" />
                      </Show>
                      CSV
                    </button>
                    
                    <button 
                      onClick={() => handleDownload(statement, 'pdf')}
                      disabled={!!downloadingId()}
                      class="flex items-center gap-2 px-3 py-2 rounded-md bg-terminal-800 border border-terminal-600 hover:bg-terminal-700 hover:border-terminal-500 text-xs font-mono text-gray-300 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <Show 
                        when={downloadingId() === `${statement.id}-pdf`}
                        fallback={<FileText size={14} />}
                      >
                        <Loader2 size={14} class="animate-spin" />
                      </Show>
                      Report
                    </button>
                  </div>

                </div>
              </div>
            )}
          </For>
        </div>
      </div>
    </div>
  );
}
