/**
 * ACCOUNT STATEMENTS & TAX DOCUMENTS PAGE
 * Complete statements history with tax form generation
 */

import { createSignal, createEffect, For, Show } from 'solid-js';
import { FileText, Download, Calendar, DollarSign, TrendingUp, AlertCircle } from 'lucide-solid';
import { apiClient, AccountStatement, TaxDocument } from '../../lib/api/client';

export default function StatementsPage() {
  const [loading, setLoading] = createSignal(false);
  const [statements, setStatements] = createSignal<AccountStatement[]>([]);
  const [taxDocs, setTaxDocs] = createSignal<TaxDocument[]>([]);
  const [selectedYear, setSelectedYear] = createSignal(new Date().getFullYear());
  const [activeTab, setActiveTab] = createSignal<'statements' | 'tax'>('statements');

  const years = Array.from({ length: 5 }, (_, i) => new Date().getFullYear() - i);

  createEffect(() => {
    loadData();
  });

  const loadData = async () => {
    console.log('📄 Loading statements for year:', selectedYear());
    setLoading(true);
    try {
      console.log('🌐 Fetching statements and tax documents...');
      const [statementsData, taxData] = await Promise.all([
        apiClient.getStatements(selectedYear()),
        apiClient.getTaxDocuments(selectedYear()),
      ]);
      console.log('✅ Statements loaded:', statementsData?.length || 0);
      console.log('✅ Tax documents loaded:', taxData?.length || 0);
      setStatements(statementsData || []);
      setTaxDocs(taxData || []);
    } catch (err: any) {
      console.error('❌ Failed to load statements:', err);
      console.error('❌ Error details:', err.message, err.response?.data);
      setStatements([]);
      setTaxDocs([]);
    } finally {
      setLoading(false);
      console.log('✅ Loading complete');
    }
  };

  const handleDownload = async (statementId: string) => {
    try {
      const url = await apiClient.downloadStatement(statementId);
      window.open(url, '_blank');
    } catch (err) {
      console.error('Download failed', err);
    }
  };

  const getStatementIcon = (type: string) => {
    switch (type) {
      case 'monthly': return <Calendar size={20} class="text-primary-500" />;
      case 'quarterly': return <TrendingUp size={20} class="text-success-500" />;
      case 'annual': return <DollarSign size={20} class="text-accent-500" />;
      default: return <FileText size={20} class="text-gray-500" />;
    }
  };

  return (
    <div class="h-full flex flex-col gap-2 sm:gap-3 p-2 sm:p-3">
      {/* Header */}
      <div class="bg-terminal-900 border border-terminal-750 p-3 sm:p-4">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 bg-accent-500/10 rounded flex items-center justify-center">
              <FileText size={20} class="text-accent-500" />
            </div>
            <div>
              <h1 class="text-lg font-bold text-white">Account Statements & Tax Documents</h1>
              <p class="text-xs text-gray-400">View and download your account history</p>
            </div>
          </div>

          <select
            value={selectedYear()}
            onChange={(e) => setSelectedYear(parseInt(e.target.value))}
            class="bg-terminal-850 border border-terminal-750 text-white px-4 py-2 rounded focus:outline-none focus:border-accent-500"
          >
            <For each={years}>
              {(year) => <option value={year}>{year}</option>}
            </For>
          </select>
        </div>
      </div>

      {/* Stats */}
      <div class="grid grid-cols-3 gap-3">
        <div class="bg-terminal-900 border border-terminal-750 p-4">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 bg-primary-500/10 rounded flex items-center justify-center">
              <FileText size={20} class="text-primary-500" />
            </div>
            <div>
              <div class="text-2xl font-bold text-white tabular-nums">{statements()?.length || 0}</div>
              <div class="text-xs text-gray-400">Statements Available</div>
            </div>
          </div>
        </div>

        <div class="bg-terminal-900 border border-terminal-750 p-4">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 bg-success-500/10 rounded flex items-center justify-center">
              <DollarSign size={20} class="text-success-500" />
            </div>
            <div>
              <div class="text-2xl font-bold text-white tabular-nums">{taxDocs()?.length || 0}</div>
              <div class="text-xs text-gray-400">Tax Documents</div>
            </div>
          </div>
        </div>

        <div class="bg-terminal-900 border border-terminal-750 p-4">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 bg-warning-500/10 rounded flex items-center justify-center">
              <Calendar size={20} class="text-warning-500" />
            </div>
            <div>
              <div class="text-2xl font-bold text-white">{selectedYear()}</div>
              <div class="text-xs text-gray-400">Selected Year</div>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div class="flex items-center gap-1 bg-terminal-900 border border-terminal-750 p-1">
        <button
          onClick={() => setActiveTab('statements')}
          class={`flex-1 px-4 py-2 text-sm font-semibold rounded transition-colors ${
            activeTab() === 'statements'
              ? 'bg-primary-500/10 text-primary-500'
              : 'text-gray-400 hover:text-white hover:bg-terminal-800'
          }`}
        >
          Account Statements
        </button>
        <button
          onClick={() => setActiveTab('tax')}
          class={`flex-1 px-4 py-2 text-sm font-semibold rounded transition-colors ${
            activeTab() === 'tax'
              ? 'bg-success-500/10 text-success-500'
              : 'text-gray-400 hover:text-white hover:bg-terminal-800'
          }`}
        >
          Tax Documents
        </button>
      </div>

      {/* Content */}
      <div class="flex-1 overflow-auto">
        <Show when={activeTab() === 'statements'}>
          <div class="bg-terminal-900 border border-terminal-750">
            <div class="p-4 border-b border-terminal-750">
              <h3 class="text-sm font-bold text-white">Account Statements for {selectedYear()}</h3>
            </div>

            <Show when={statements()?.length === 0}>
              <div class="p-8 text-center">
                <FileText size={48} class="text-gray-600 mx-auto mb-4" />
                <div class="text-gray-500 mb-2">No statements available</div>
                <div class="text-xs text-gray-600">
                  Statements are generated monthly and appear here once available
                </div>
              </div>
            </Show>

            <div class="divide-y divide-terminal-750">
              <For each={statements() || []}>
                {(statement) => (
                  <div class="p-4 hover:bg-terminal-850 transition-colors">
                    <div class="flex items-center gap-4">
                      <div class="w-12 h-12 bg-terminal-850 rounded flex items-center justify-center flex-shrink-0">
                        {getStatementIcon(statement.statement_type)}
                      </div>

                      <div class="flex-1">
                        <div class="flex items-center gap-2 mb-1">
                          <h4 class="text-sm font-semibold text-white capitalize">
                            {statement.statement_type} Statement
                          </h4>
                          <span class="px-2 py-0.5 bg-primary-500/10 text-primary-500 text-xs font-semibold rounded">
                            {selectedYear()}
                          </span>
                        </div>
                        <div class="text-xs text-gray-400">
                          {new Date(statement.period_start).toLocaleDateString()} -{' '}
                          {new Date(statement.period_end).toLocaleDateString()}
                        </div>
                        <div class="text-xs text-gray-500 mt-1">
                          Generated: {new Date(statement.generated_at).toLocaleDateString()} •{' '}
                          {(statement.file_size / 1024).toFixed(1)} KB
                        </div>
                      </div>

                      <button
                        onClick={() => handleDownload(statement.id)}
                        class="flex items-center gap-2 px-4 py-2 bg-accent-500 hover:bg-accent-600 text-white text-sm font-semibold rounded transition-colors"
                      >
                        <Download size={16} />
                        <span>Download PDF</span>
                      </button>
                    </div>
                  </div>
                )}
              </For>
            </div>
          </div>
        </Show>

        <Show when={activeTab() === 'tax'}>
          <div class="bg-terminal-900 border border-terminal-750">
            <div class="p-4 border-b border-terminal-750">
              <h3 class="text-sm font-bold text-white">Tax Documents for {selectedYear()}</h3>
            </div>

            <div class="p-4 bg-warning-500/5 border-b border-warning-500/20">
              <div class="flex items-start gap-3">
                <AlertCircle size={20} class="text-warning-500 flex-shrink-0 mt-0.5" />
                <div class="text-xs text-gray-400">
                  <span class="font-semibold text-warning-500">Tax Season Notice:</span> Tax documents
                  for {selectedYear()} will be available by January 31, {selectedYear() + 1}. Download your
                  documents before filing your tax return. Consult your tax advisor for guidance.
                </div>
              </div>
            </div>

            <Show when={taxDocs()?.length === 0}>
              <div class="p-8 text-center">
                <DollarSign size={48} class="text-gray-600 mx-auto mb-4" />
                <div class="text-gray-500 mb-2">No tax documents available</div>
                <div class="text-xs text-gray-600">
                  Tax documents will appear here once generated
                </div>
              </div>
            </Show>

            <div class="divide-y divide-terminal-750">
              <For each={taxDocs() || []}>
                {(doc) => (
                  <div class="p-4 hover:bg-terminal-850 transition-colors">
                    <div class="flex items-center gap-4">
                      <div class="w-12 h-12 bg-terminal-850 rounded flex items-center justify-center flex-shrink-0">
                        <FileText size={20} class="text-success-500" />
                      </div>

                      <div class="flex-1">
                        <div class="flex items-center gap-2 mb-1">
                          <h4 class="text-sm font-semibold text-white">Form {doc.document_type}</h4>
                          <span class={`px-2 py-0.5 text-xs font-semibold rounded ${
                            doc.available
                              ? 'bg-success-500/10 text-success-500'
                              : 'bg-gray-800 text-gray-500'
                          }`}>
                            {doc.available ? 'Available' : 'Pending'}
                          </span>
                        </div>
                        <div class="text-xs text-gray-400">Tax Year: {doc.tax_year}</div>
                        {doc.generated_at && (
                          <div class="text-xs text-gray-500 mt-1">
                            Generated: {new Date(doc.generated_at).toLocaleDateString()}
                          </div>
                        )}
                      </div>

                      <Show when={doc.available}>
                        <button
                          onClick={() => window.open(doc.file_url, '_blank')}
                          class="flex items-center gap-2 px-4 py-2 bg-success-500 hover:bg-success-600 text-white text-sm font-semibold rounded transition-colors"
                        >
                          <Download size={16} />
                          <span>Download</span>
                        </button>
                      </Show>
                    </div>
                  </div>
                )}
              </For>
            </div>
          </div>
        </Show>
      </div>

      {/* Info */}
      <div class="bg-primary-500/5 border border-primary-500/20 p-4 rounded">
        <div class="text-xs text-gray-400">
          <span class="font-semibold text-primary-500">Important Information:</span> All account
          statements and tax documents are available for download at any time. Documents are stored
          securely and are accessible for 7 years. For questions about your statements or tax forms,
          contact our support team.
        </div>
      </div>
    </div>
  );
}
