import { createSignal, createEffect, Show, For } from 'solid-js';
import { apiClient } from '~/lib/api/client';
import { Building, Users, Globe, Activity, FileText, BarChart3 } from 'lucide-solid';
import { formatCurrency, formatNumber } from '~/lib/utils/format';

interface CompanyProfileWidgetProps {
  symbol: string;
}

export default function CompanyProfileWidget(props: CompanyProfileWidgetProps) {
  const [profile, setProfile] = createSignal<any>(null);
  const [financials, setFinancials] = createSignal<any>(null);
  const [reportedFinancials, setReportedFinancials] = createSignal<any>(null);
  const [estimates, setEstimates] = createSignal<any>(null);
  const [loading, setLoading] = createSignal(false);

  createEffect(async () => {
    if (!props.symbol) return;
    
    setLoading(true);
    try {
      const [profData, finData, repFinData, estData] = await Promise.all([
        apiClient.getCompanyProfile(props.symbol),
        apiClient.getFinancials(props.symbol),
        apiClient.getFinancialsReported(props.symbol),
        apiClient.getEstimates(props.symbol)
      ]);
      setProfile(profData);
      setFinancials(finData);
      setReportedFinancials(repFinData);
      setEstimates(estData);
    } catch (e) {
      console.error("Error loading company data", e);
    } finally {
      setLoading(false);
    }
  });

  return (
    <div class="bg-terminal-900 border border-terminal-800 rounded-lg p-4 h-full overflow-y-auto custom-scrollbar">
      <div class="flex items-center gap-2 mb-4 border-b border-terminal-800 pb-2">
        <Building class="w-5 h-5 text-primary-400" />
        <h3 class="text-lg font-bold text-white">Company Profile</h3>
      </div>

      <Show when={!loading()} fallback={<div class="text-center py-8 text-gray-500">Loading data...</div>}>
        <Show when={profile()} fallback={<div class="text-center py-8 text-gray-500">No profile data available</div>}>
          <div class="space-y-6">
            {/* Header Info with Logo */}
            <div class="flex items-start gap-4">
              <Show when={profile().logo}>
                <div class="w-12 h-12 bg-white rounded-lg p-1 flex-shrink-0">
                  <img src={profile().logo} alt={`${profile().name} logo`} class="w-full h-full object-contain" />
                </div>
              </Show>
              <div>
                <h2 class="text-xl font-bold text-white leading-tight">{profile().name}</h2>
                <div class="flex items-center gap-2 text-sm text-gray-400 mt-1">
                  <span class="bg-terminal-800 px-2 py-0.5 rounded text-xs font-mono">{profile().exchange}</span>
                  <span class="font-mono text-primary-400">{profile().ticker}</span>
                </div>
              </div>
            </div>

            {/* Key Details */}
            <div class="grid grid-cols-2 gap-4 bg-terminal-950/50 p-3 rounded-lg border border-terminal-800">
              <div class="space-y-1">
                <span class="text-xs text-gray-500 uppercase">Sector</span>
                <div class="text-sm text-white font-medium truncate" title={profile().finnhubIndustry}>{profile().finnhubIndustry}</div>
              </div>
              <div class="space-y-1">
                <span class="text-xs text-gray-500 uppercase">Country</span>
                <div class="text-sm text-white font-medium">{profile().country}</div>
              </div>
              <div class="space-y-1">
                <span class="text-xs text-gray-500 uppercase">Currency</span>
                <div class="text-sm text-white font-medium">{profile().currency}</div>
              </div>
              <div class="space-y-1">
                <span class="text-xs text-gray-500 uppercase">IPO Date</span>
                <div class="text-sm text-white font-medium">{profile().ipo}</div>
              </div>
            </div>

            {/* Financial Metrics */}
            <Show when={financials() && financials().metric}>
              <div class="border-t border-terminal-800 pt-4">
                <h4 class="text-sm font-bold text-gray-300 mb-3 flex items-center gap-2">
                  <Activity class="w-4 h-4 text-primary-400" /> Key Metrics
                </h4>
                <div class="grid grid-cols-2 gap-y-3 gap-x-4">
                  <div class="flex justify-between items-center">
                    <span class="text-xs text-gray-500">Market Cap</span>
                    <span class="text-xs text-white font-mono">{formatNumber(profile().marketCapitalization)}M</span>
                  </div>
                  <div class="flex justify-between items-center">
                    <span class="text-xs text-gray-500">P/E Ratio</span>
                    <span class="text-xs text-white font-mono">{financials().metric.peTTM?.toFixed(2) || '-'}</span>
                  </div>
                  <div class="flex justify-between items-center">
                    <span class="text-xs text-gray-500">EPS (TTM)</span>
                    <span class="text-xs text-white font-mono">{financials().metric.epsTTM?.toFixed(2) || '-'}</span>
                  </div>
                  <div class="flex justify-between items-center">
                    <span class="text-xs text-gray-500">Div Yield</span>
                    <span class="text-xs text-white font-mono">{financials().metric.dividendYieldIndicatedAnnual?.toFixed(2) || '-'}%</span>
                  </div>
                  <div class="flex justify-between items-center">
                    <span class="text-xs text-gray-500">52W High</span>
                    <span class="text-xs text-success-400 font-mono">{formatCurrency(financials().metric['52WeekHigh'])}</span>
                  </div>
                  <div class="flex justify-between items-center">
                    <span class="text-xs text-gray-500">52W Low</span>
                    <span class="text-xs text-danger-400 font-mono">{formatCurrency(financials().metric['52WeekLow'])}</span>
                  </div>
                </div>
              </div>
            </Show>

            {/* Earnings Estimates */}
            <Show when={estimates() && estimates().data && estimates().data.length > 0}>
              <div class="border-t border-terminal-800 pt-4">
                <h4 class="text-sm font-bold text-gray-300 mb-3 flex items-center gap-2">
                  <BarChart3 class="w-4 h-4 text-primary-400" /> Earnings Estimates
                </h4>
                <div class="space-y-2">
                  <For each={estimates().data.slice(0, 2)}>
                    {(est: any) => (
                      <div class="bg-terminal-950/30 p-2 rounded border border-terminal-800/50">
                        <div class="flex justify-between text-xs mb-1">
                          <span class="text-gray-400">Period: {est.period}</span>
                          <span class="text-primary-400 font-mono">Avg: {est.epsAvg}</span>
                        </div>
                        <div class="flex justify-between text-xs text-gray-500">
                          <span>Low: {est.epsLow}</span>
                          <span>High: {est.epsHigh}</span>
                        </div>
                      </div>
                    )}
                  </For>
                </div>
              </div>
            </Show>

            {/* Reported Financials (Latest) */}
            <Show when={reportedFinancials() && reportedFinancials().data && reportedFinancials().data.length > 0}>
              <div class="border-t border-terminal-800 pt-4">
                <h4 class="text-sm font-bold text-gray-300 mb-3 flex items-center gap-2">
                  <FileText class="w-4 h-4 text-primary-400" /> Latest Report
                </h4>
                <div class="text-xs text-gray-400 mb-2">
                  {reportedFinancials().data[0].year} Q{reportedFinancials().data[0].quarter} ({reportedFinancials().data[0].accessNumber})
                </div>
                <div class="space-y-1">
                   <div class="flex justify-between text-xs">
                      <span class="text-gray-500">Filed</span>
                      <span class="text-white font-mono">{reportedFinancials().data[0].filedDate}</span>
                   </div>
                   <Show when={reportedFinancials().data[0].report?.bs?.length > 0}>
                      <div class="flex justify-between text-xs">
                        <span class="text-gray-500">Balance Sheet Items</span>
                        <span class="text-white font-mono">{reportedFinancials().data[0].report.bs.length}</span>
                      </div>
                   </Show>
                   <Show when={reportedFinancials().data[0].report?.ic?.length > 0}>
                      <div class="flex justify-between text-xs">
                        <span class="text-gray-500">Income Stmt Items</span>
                        <span class="text-white font-mono">{reportedFinancials().data[0].report.ic.length}</span>
                      </div>
                   </Show>
                </div>
              </div>
            </Show>

            {/* Contact Info */}
            <div class="border-t border-terminal-800 pt-4 space-y-2">
              <Show when={profile().weburl}>
                <a href={profile().weburl} target="_blank" rel="noopener noreferrer" class="flex items-center gap-2 text-xs text-primary-400 hover:text-primary-300 transition-colors">
                  <Globe class="w-3 h-3" /> Website
                </a>
              </Show>
              <Show when={profile().phone}>
                <div class="flex items-center gap-2 text-xs text-gray-400">
                  <Users class="w-3 h-3" /> {profile().phone}
                </div>
              </Show>
            </div>
          </div>
        </Show>
      </Show>
    </div>
  );
}
