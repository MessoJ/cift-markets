/**
 * MARKET SCREENER PAGE - Industry Standard Design
 * 
 * Features:
 * - Preset screens (Top Gainers, Most Active, Undervalued, etc.)
 * - Advanced filtering with categorized filters
 * - Sortable columns with sticky headers
 * - Real data from database (no mock data)
 * - Proper scrolling with fixed header
 * - Export to CSV
 * - Heatmap visualization
 * - Pagination
 * - Multiple Views (Overview, Valuation, Performance)
 */

import { createSignal, createEffect, For, Show, onMount } from 'solid-js';
import { useNavigate } from '@solidjs/router';
import { 
  Filter, Save, Trash2, Search, TrendingUp, TrendingDown, Star, 
  RefreshCw, Download, Grid, List, Columns, ChevronDown, ChevronUp,
  Zap, BarChart3, DollarSign, Activity, Building, ArrowUpDown,
  X, Flame, Target, PieChart, ChevronLeft, ChevronRight, Globe
} from 'lucide-solid';
import { formatCurrency, formatPercentage, formatNumber } from '../../lib/utils';
import HeatmapGrid from '~/components/ui/HeatmapGrid';

interface ScreenerResult {
  symbol: string;
  name: string;
  price: number;
  change: number;
  change_pct: number;
  volume: number;
  market_cap: number;
  pe_ratio: number | null;
  forward_pe: number | null;
  dividend_yield: number | null;
  beta: number | null;
  week52_high: number | null;
  week52_low: number | null;
  avg_volume: number | null;
  sector: string;
  industry: string;
  country: string;
}

interface PresetScreen {
  id: string;
  name: string;
  description: string;
  criteria: any;
  sort_by?: string;
  sort_order?: string;
  icon?: any;
}

interface SectorInfo {
  name: string;
  count: number;
}

type ViewTab = 'overview' | 'valuation' | 'performance';

export default function ScreenerPage() {
  const navigate = useNavigate();
  
  // State
  const [loading, setLoading] = createSignal(false);
  const [results, setResults] = createSignal<ScreenerResult[]>([]);
  const [presets, setPresets] = createSignal<PresetScreen[]>([]);
  const [sectors, setSectors] = createSignal<SectorInfo[]>([]);
  const [viewMode, setViewMode] = createSignal<'table' | 'heatmap'>('table');
  const [activePreset, setActivePreset] = createSignal<string | null>(null);
  const [showFilters, setShowFilters] = createSignal(true);
  const [activeTab, setActiveTab] = createSignal<ViewTab>('overview');
  
  // Pagination & Sorting
  const [page, setPage] = createSignal(1);
  const [limit, setLimit] = createSignal(50);
  const [totalCount, setTotalCount] = createSignal(0);
  const [sortColumn, setSortColumn] = createSignal<string>('market_cap');
  const [sortDirection, setSortDirection] = createSignal<'asc' | 'desc'>('desc');

  // Filter states
  const [priceMin, setPriceMin] = createSignal<number | undefined>();
  const [priceMax, setPriceMax] = createSignal<number | undefined>();
  const [volumeMin, setVolumeMin] = createSignal<number | undefined>();
  const [marketCapMin, setMarketCapMin] = createSignal<number | undefined>();
  const [marketCapMax, setMarketCapMax] = createSignal<number | undefined>();
  const [peRatioMin, setPeRatioMin] = createSignal<number | undefined>();
  const [peRatioMax, setPeRatioMax] = createSignal<number | undefined>();
  const [changePctMin, setChangePctMin] = createSignal<number | undefined>();
  const [changePctMax, setChangePctMax] = createSignal<number | undefined>();
  const [dividendMin, setDividendMin] = createSignal<number | undefined>();
  const [betaMin, setBetaMin] = createSignal<number | undefined>();
  const [betaMax, setBetaMax] = createSignal<number | undefined>();
  const [selectedSector, setSelectedSector] = createSignal<string>('');
  const [selectedCountry, setSelectedCountry] = createSignal<string>('');

  // Preset icons mapping
  const presetIcons: Record<string, any> = {
    gainers: TrendingUp,
    losers: TrendingDown,
    most_active: Flame,
    mega_cap: Building,
    large_cap: Building,
    undervalued: Target,
    dividend: DollarSign,
    tech: Zap,
    healthcare: Activity,
  };

  // Format helpers
  const formatMarketCap = (value: number) => {
    if (!value) return '-';
    if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
    if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
    return `$${value.toFixed(0)}`;
  };

  const formatVol = (value: number) => {
    if (!value) return '-';
    if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
    if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
    return value.toString();
  };

  // Load presets and sectors on mount
  onMount(async () => {
    await Promise.all([loadPresets(), loadSectors()]);
    // Auto-run default screen
    await handleScan();
  });

  const loadPresets = async () => {
    try {
      const response = await fetch('/api/v1/screener/presets');
      if (response.ok) {
        const data = await response.json();
        setPresets(data);
      }
    } catch (err) {
      console.error('Failed to load presets:', err);
    }
  };

  const loadSectors = async () => {
    try {
      const response = await fetch('/api/v1/screener/sectors');
      if (response.ok) {
        const data = await response.json();
        setSectors(data);
      }
    } catch (err) {
      console.error('Failed to load sectors:', err);
    }
  };

  const getCriteria = () => ({
    price_min: priceMin(),
    price_max: priceMax(),
    volume_min: volumeMin(),
    // Convert Billions (UI) to Millions (DB)
    market_cap_min: marketCapMin() ? marketCapMin()! * 1000 : undefined,
    market_cap_max: marketCapMax() ? marketCapMax()! * 1000 : undefined,
    pe_ratio_min: peRatioMin(),
    pe_ratio_max: peRatioMax(),
    change_pct_min: changePctMin(),
    change_pct_max: changePctMax(),
    dividend_yield_min: dividendMin(),
    beta_min: betaMin(),
    beta_max: betaMax(),
    sector: selectedSector() || undefined,
    country: selectedCountry() || undefined,
  });

  const handleScan = async (overrideCriteria?: any, newPage = 1) => {
    setLoading(true);
    try {
      const criteria = overrideCriteria || getCriteria();
      const offset = (newPage - 1) * limit();
      
      const queryParams = new URLSearchParams({
        limit: limit().toString(),
        offset: offset.toString(),
        sort_by: sortColumn(),
        sort_order: sortDirection(),
      });

      const response = await fetch(`/api/v1/screener/scan?${queryParams}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(criteria),
      });
      
      if (response.ok) {
        const data = await response.json();
        // Handle both old (array) and new (object) API response formats
        if (Array.isArray(data)) {
          setResults(data);
          setTotalCount(data.length);
        } else {
          setResults(data.results || []);
          setTotalCount(data.total_count || 0);
        }
        setPage(newPage);
      } else {
        console.error('Scan failed:', response.status);
        setResults([]);
        setTotalCount(0);
      }
    } catch (err) {
      console.error('Scan error:', err);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handlePageChange = (newPage: number) => {
    if (newPage < 1 || newPage > Math.ceil(totalCount() / limit())) return;
    handleScan(undefined, newPage);
  };

  const handlePresetClick = async (preset: PresetScreen) => {
    setActivePreset(preset.id);
    
    // Reset all filters first (without running scan)
    resetFilters(false);
    
    // Build criteria directly from preset
    const criteria: any = {};
    const c = preset.criteria;
    
    if (c.price_min) criteria.price_min = c.price_min;
    if (c.price_max) criteria.price_max = c.price_max;
    if (c.volume_min) criteria.volume_min = c.volume_min;
    if (c.market_cap_min) criteria.market_cap_min = c.market_cap_min;
    if (c.pe_ratio_min) criteria.pe_ratio_min = c.pe_ratio_min;
    if (c.pe_ratio_max) criteria.pe_ratio_max = c.pe_ratio_max;
    if (c.change_pct_min) criteria.change_pct_min = c.change_pct_min;
    if (c.change_pct_max) criteria.change_pct_max = c.change_pct_max;
    if (c.dividend_yield_min) criteria.dividend_yield_min = c.dividend_yield_min;
    if (c.sector) criteria.sector = c.sector;
    
    // Update UI state
    if (c.price_min) setPriceMin(c.price_min);
    if (c.price_max) setPriceMax(c.price_max);
    if (c.volume_min) setVolumeMin(c.volume_min);
    // Convert Millions (DB) to Billions (UI)
    if (c.market_cap_min) setMarketCapMin(c.market_cap_min / 1000);
    if (c.pe_ratio_min) setPeRatioMin(c.pe_ratio_min);
    if (c.pe_ratio_max) setPeRatioMax(c.pe_ratio_max);
    if (c.change_pct_min) setChangePctMin(c.change_pct_min);
    if (c.change_pct_max) setChangePctMax(c.change_pct_max);
    if (c.dividend_yield_min) setDividendMin(c.dividend_yield_min);
    if (c.sector) setSelectedSector(c.sector);
    
    if (preset.sort_by) setSortColumn(preset.sort_by);
    if (preset.sort_order) setSortDirection(preset.sort_order as 'asc' | 'desc');
    
    await handleScan(criteria, 1);
  };

  const resetFilters = (runScan = true) => {
    setPriceMin(undefined);
    setPriceMax(undefined);
    setVolumeMin(undefined);
    setMarketCapMin(undefined);
    setMarketCapMax(undefined);
    setPeRatioMin(undefined);
    setPeRatioMax(undefined);
    setChangePctMin(undefined);
    setChangePctMax(undefined);
    setDividendMin(undefined);
    setBetaMin(undefined);
    setBetaMax(undefined);
    setSelectedSector('');
    setSelectedCountry('');
    setActivePreset(null);
    setPage(1);
    if (runScan) {
      handleScan(undefined, 1);
    }
  };

  const handleSort = (column: string) => {
    const newDirection = sortColumn() === column && sortDirection() === 'desc' ? 'asc' : 'desc';
    setSortColumn(column);
    setSortDirection(newDirection);
    handleScan(undefined, 1); // Reset to page 1 on sort
  };

  const exportToCSV = () => {
    const headers = ['Symbol', 'Name', 'Price', 'Change %', 'Volume', 'Market Cap', 'P/E', 'Dividend %', 'Sector', 'Industry', 'Country'];
    const rows = results().map(r => [
      r.symbol,
      `"${r.name}"`,
      r.price?.toFixed(2) || '',
      r.change_pct?.toFixed(2) || '',
      r.volume || '',
      r.market_cap || '',
      r.pe_ratio?.toFixed(2) || '',
      r.dividend_yield?.toFixed(2) || '',
      `"${r.sector}"`,
      `"${r.industry}"`,
      `"${r.country}"`,
    ]);
    
    const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `screener_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
  };

  const heatmapData = () => results().map(r => ({
    id: r.symbol,
    label: r.symbol,
    value: r.market_cap || 0,
    changePercent: r.change_pct || 0,
    change: r.change || 0,
    size: r.market_cap || 1,
  }));

  // Sort header component
  const SortHeader = (props: { column: string; label: string; align?: 'left' | 'right' }) => (
    <th
      class={`px-3 py-3 text-xs font-semibold text-gray-400 cursor-pointer hover:text-white transition-colors sticky top-0 bg-terminal-850 z-10 ${
        props.align === 'right' ? 'text-right' : 'text-left'
      }`}
      onClick={() => handleSort(props.column)}
    >
      <div class={`flex items-center gap-1 ${props.align === 'right' ? 'justify-end' : ''}`}>
        <span>{props.label}</span>
        <Show when={sortColumn() === props.column}>
          {sortDirection() === 'asc' ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        </Show>
        <Show when={sortColumn() !== props.column}>
          <ArrowUpDown size={10} class="opacity-30" />
        </Show>
      </div>
    </th>
  );

  return (
    <div class="h-full flex flex-col bg-terminal-950">
      {/* Header Bar */}
      <div class="flex-shrink-0 bg-terminal-900 border-b border-terminal-750 px-4 py-3">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 bg-gradient-to-br from-primary-500/20 to-accent-500/20 rounded-lg flex items-center justify-center">
              <Filter size={20} class="text-accent-500" />
            </div>
            <div>
              <h1 class="text-lg font-bold text-white">Stock Screener</h1>
              <p class="text-xs text-gray-500">
                {totalCount()} stocks • Last updated {new Date().toLocaleTimeString()}
              </p>
            </div>
          </div>
          
          <div class="flex items-center gap-2">
            {/* View Toggle */}
            <div class="flex bg-terminal-850 border border-terminal-750 rounded overflow-hidden">
              <button
                onClick={() => setViewMode('table')}
                class={`px-3 py-1.5 flex items-center gap-1.5 text-xs transition-all ${
                  viewMode() === 'table' 
                    ? 'bg-primary-600 text-white' 
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                <List size={14} />
                <span class="hidden sm:inline">Table</span>
              </button>
              <button
                onClick={() => setViewMode('heatmap')}
                class={`px-3 py-1.5 flex items-center gap-1.5 text-xs transition-all ${
                  viewMode() === 'heatmap' 
                    ? 'bg-primary-600 text-white' 
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                <Grid size={14} />
                <span class="hidden sm:inline">Heatmap</span>
              </button>
            </div>

            <button
              onClick={() => setShowFilters(!showFilters())}
              class={`px-3 py-1.5 flex items-center gap-1.5 text-xs border rounded transition-all ${
                showFilters()
                  ? 'bg-accent-500/10 border-accent-500/50 text-accent-400'
                  : 'bg-terminal-850 border-terminal-750 text-gray-400 hover:text-white'
              }`}
            >
              <Filter size={14} />
              <span class="hidden sm:inline">Filters</span>
            </button>

            <Show when={results().length > 0}>
              <button
                onClick={exportToCSV}
                class="px-3 py-1.5 flex items-center gap-1.5 text-xs bg-terminal-850 border border-terminal-750 text-gray-400 hover:text-white rounded transition-colors"
              >
                <Download size={14} />
                <span class="hidden sm:inline">Export</span>
              </button>
            </Show>

            <button
              onClick={() => handleScan(undefined, 1)}
              disabled={loading()}
              class="px-4 py-1.5 flex items-center gap-2 text-xs font-semibold bg-accent-500 hover:bg-accent-600 disabled:opacity-50 text-white rounded transition-colors"
            >
              <RefreshCw size={14} class={loading() ? 'animate-spin' : ''} />
              <span>{loading() ? 'Scanning...' : 'Refresh'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Preset Screens - Horizontal Pills */}
      <div class="flex-shrink-0 bg-terminal-900/50 border-b border-terminal-750 px-4 py-2 overflow-x-auto">
        <div class="flex items-center gap-2">
          <span class="text-xs text-gray-500 font-medium mr-2">Quick Screens:</span>
          <For each={presets()}>
            {(preset) => {
              const Icon = presetIcons[preset.id] || PieChart;
              return (
                <button
                  onClick={() => handlePresetClick(preset)}
                  class={`flex-shrink-0 px-3 py-1.5 flex items-center gap-1.5 text-xs font-medium rounded-full transition-all ${
                    activePreset() === preset.id
                      ? 'bg-primary-600 text-white'
                      : 'bg-terminal-850 text-gray-400 hover:text-white border border-terminal-750 hover:border-primary-500/50'
                  }`}
                  title={preset.description}
                >
                  <Icon size={12} />
                  <span>{preset.name}</span>
                </button>
              );
            }}
          </For>
          <Show when={activePreset()}>
            <button
              onClick={() => resetFilters()}
              class="flex-shrink-0 px-2 py-1.5 text-xs text-gray-500 hover:text-white transition-colors"
            >
              <X size={14} />
            </button>
          </Show>
        </div>
      </div>

      <div class="flex-1 flex overflow-hidden relative">
        {/* Filter Panel (Collapsible) */}
        <Show when={showFilters()}>
          <div class="w-64 flex-shrink-0 bg-terminal-900 border-r border-terminal-750 overflow-y-auto absolute inset-y-0 left-0 z-30 md:static shadow-xl md:shadow-none">
            <div class="p-4 space-y-4">
              <div class="flex justify-between items-center md:hidden mb-2">
                <h3 class="font-bold text-white">Filters</h3>
                <button onClick={() => setShowFilters(false)} class="text-gray-400 hover:text-white">
                  <X size={20} />
                </button>
              </div>
              {/* Price */}
              <div>
                <label class="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2 block">
                  Price Range
                </label>
                <div class="grid grid-cols-2 gap-2">
                  <input
                    type="number"
                    value={priceMin() ?? ''}
                    onInput={(e) => setPriceMin(e.target.value ? parseFloat(e.target.value) : undefined)}
                    placeholder="Min"
                    class="w-full bg-terminal-850 border border-terminal-750 text-white text-xs px-3 py-2 rounded focus:outline-none focus:border-accent-500"
                  />
                  <input
                    type="number"
                    value={priceMax() ?? ''}
                    onInput={(e) => setPriceMax(e.target.value ? parseFloat(e.target.value) : undefined)}
                    placeholder="Max"
                    class="w-full bg-terminal-850 border border-terminal-750 text-white text-xs px-3 py-2 rounded focus:outline-none focus:border-accent-500"
                  />
                </div>
              </div>

              {/* Market Cap */}
              <div>
                <label class="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2 block">
                  Market Cap
                </label>
                <div class="grid grid-cols-2 gap-2">
                  <input
                    type="number"
                    value={marketCapMin() ?? ''}
                    onInput={(e) => setMarketCapMin(e.target.value ? parseFloat(e.target.value) : undefined)}
                    placeholder="Min (B)"
                    class="w-full bg-terminal-850 border border-terminal-750 text-white text-xs px-3 py-2 rounded focus:outline-none focus:border-accent-500"
                  />
                  <input
                    type="number"
                    value={marketCapMax() ?? ''}
                    onInput={(e) => setMarketCapMax(e.target.value ? parseFloat(e.target.value) : undefined)}
                    placeholder="Max (B)"
                    class="w-full bg-terminal-850 border border-terminal-750 text-white text-xs px-3 py-2 rounded focus:outline-none focus:border-accent-500"
                  />
                </div>
              </div>

              {/* Volume */}
              <div>
                <label class="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2 block">
                  Min Volume
                </label>
                <input
                  type="number"
                  value={volumeMin() ?? ''}
                  onInput={(e) => setVolumeMin(e.target.value ? parseFloat(e.target.value) : undefined)}
                  placeholder="e.g. 1000000"
                  class="w-full bg-terminal-850 border border-terminal-750 text-white text-xs px-3 py-2 rounded focus:outline-none focus:border-accent-500"
                />
              </div>

              {/* Change % */}
              <div>
                <label class="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2 block">
                  Change %
                </label>
                <div class="grid grid-cols-2 gap-2">
                  <input
                    type="number"
                    value={changePctMin() ?? ''}
                    onInput={(e) => setChangePctMin(e.target.value ? parseFloat(e.target.value) : undefined)}
                    placeholder="Min %"
                    class="w-full bg-terminal-850 border border-terminal-750 text-white text-xs px-3 py-2 rounded focus:outline-none focus:border-accent-500"
                  />
                  <input
                    type="number"
                    value={changePctMax() ?? ''}
                    onInput={(e) => setChangePctMax(e.target.value ? parseFloat(e.target.value) : undefined)}
                    placeholder="Max %"
                    class="w-full bg-terminal-850 border border-terminal-750 text-white text-xs px-3 py-2 rounded focus:outline-none focus:border-accent-500"
                  />
                </div>
              </div>

              {/* P/E Ratio */}
              <div>
                <label class="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2 block">
                  P/E Ratio
                </label>
                <div class="grid grid-cols-2 gap-2">
                  <input
                    type="number"
                    value={peRatioMin() ?? ''}
                    onInput={(e) => setPeRatioMin(e.target.value ? parseFloat(e.target.value) : undefined)}
                    placeholder="Min"
                    class="w-full bg-terminal-850 border border-terminal-750 text-white text-xs px-3 py-2 rounded focus:outline-none focus:border-accent-500"
                  />
                  <input
                    type="number"
                    value={peRatioMax() ?? ''}
                    onInput={(e) => setPeRatioMax(e.target.value ? parseFloat(e.target.value) : undefined)}
                    placeholder="Max"
                    class="w-full bg-terminal-850 border border-terminal-750 text-white text-xs px-3 py-2 rounded focus:outline-none focus:border-accent-500"
                  />
                </div>
              </div>

              {/* Dividend Yield */}
              <div>
                <label class="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2 block">
                  Min Dividend Yield %
                </label>
                <input
                  type="number"
                  value={dividendMin() ?? ''}
                  onInput={(e) => setDividendMin(e.target.value ? parseFloat(e.target.value) : undefined)}
                  placeholder="e.g. 2"
                  class="w-full bg-terminal-850 border border-terminal-750 text-white text-xs px-3 py-2 rounded focus:outline-none focus:border-accent-500"
                />
              </div>

              {/* Beta */}
              <div>
                <label class="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2 block">
                  Beta (Volatility)
                </label>
                <div class="grid grid-cols-2 gap-2">
                  <input
                    type="number"
                    value={betaMin() ?? ''}
                    onInput={(e) => setBetaMin(e.target.value ? parseFloat(e.target.value) : undefined)}
                    placeholder="Min"
                    class="w-full bg-terminal-850 border border-terminal-750 text-white text-xs px-3 py-2 rounded focus:outline-none focus:border-accent-500"
                  />
                  <input
                    type="number"
                    value={betaMax() ?? ''}
                    onInput={(e) => setBetaMax(e.target.value ? parseFloat(e.target.value) : undefined)}
                    placeholder="Max"
                    class="w-full bg-terminal-850 border border-terminal-750 text-white text-xs px-3 py-2 rounded focus:outline-none focus:border-accent-500"
                  />
                </div>
              </div>

              {/* Sector */}
              <div>
                <label class="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2 block">
                  Sector
                </label>
                <select
                  value={selectedSector()}
                  onChange={(e) => setSelectedSector(e.target.value)}
                  class="w-full bg-terminal-850 border border-terminal-750 text-white text-xs px-3 py-2 rounded focus:outline-none focus:border-accent-500"
                >
                  <option value="">All Sectors</option>
                  <For each={sectors()}>
                    {(s) => (
                      <option value={s.name}>
                        {s.name} ({s.count})
                      </option>
                    )}
                  </For>
                </select>
              </div>

              {/* Country */}
              <div>
                <label class="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2 block">
                  Country
                </label>
                <select
                  value={selectedCountry()}
                  onChange={(e) => setSelectedCountry(e.target.value)}
                  class="w-full bg-terminal-850 border border-terminal-750 text-white text-xs px-3 py-2 rounded focus:outline-none focus:border-accent-500"
                >
                  <option value="">All Countries</option>
                  <option value="US">United States</option>
                  <option value="CN">China</option>
                  <option value="UK">United Kingdom</option>
                </select>
              </div>

              {/* Action Buttons */}
              <div class="pt-4 space-y-2">
                <button
                  onClick={() => handleScan(undefined, 1)}
                  disabled={loading()}
                  class="w-full flex items-center justify-center gap-2 px-4 py-3 bg-accent-500 hover:bg-accent-600 disabled:opacity-50 text-white font-semibold rounded transition-colors"
                >
                  <Search size={16} class={loading() ? 'animate-pulse' : ''} />
                  <span>{loading() ? 'Scanning...' : 'Run Screen'}</span>
                </button>

                <button
                  onClick={() => resetFilters()}
                  class="w-full px-4 py-2 bg-terminal-850 hover:bg-terminal-800 border border-terminal-750 text-gray-400 hover:text-white text-sm rounded transition-colors"
                >
                  Reset All Filters
                </button>
              </div>
            </div>
          </div>
        </Show>

        {/* Main Content */}
        <div class="flex-1 flex flex-col overflow-hidden bg-terminal-900">
          {/* View Tabs */}
          <Show when={viewMode() === 'table'}>
            <div class="flex-shrink-0 border-b border-terminal-750 px-4">
              <div class="flex gap-6">
                <button
                  onClick={() => setActiveTab('overview')}
                  class={`py-3 text-xs font-medium border-b-2 transition-colors ${
                    activeTab() === 'overview'
                      ? 'border-accent-500 text-white'
                      : 'border-transparent text-gray-400 hover:text-white'
                  }`}
                >
                  Overview
                </button>
                <button
                  onClick={() => setActiveTab('valuation')}
                  class={`py-3 text-xs font-medium border-b-2 transition-colors ${
                    activeTab() === 'valuation'
                      ? 'border-accent-500 text-white'
                      : 'border-transparent text-gray-400 hover:text-white'
                  }`}
                >
                  Valuation
                </button>
                <button
                  onClick={() => setActiveTab('performance')}
                  class={`py-3 text-xs font-medium border-b-2 transition-colors ${
                    activeTab() === 'performance'
                      ? 'border-accent-500 text-white'
                      : 'border-transparent text-gray-400 hover:text-white'
                  }`}
                >
                  Performance
                </button>
              </div>
            </div>
          </Show>

          {/* Heatmap View */}
          <Show when={viewMode() === 'heatmap'}>
            <div class="flex-1 p-4 overflow-auto">
              <Show when={results().length > 0} fallback={
                <div class="flex items-center justify-center h-full">
                  <div class="text-center">
                    <Grid size={48} class="text-gray-600 mx-auto mb-4" />
                    <p class="text-gray-500">No data for heatmap. Run a screen first.</p>
                  </div>
                </div>
              }>
                <HeatmapGrid
                  data={heatmapData()}
                  valueKey="changePercent"
                  showLabels={true}
                  showValues={true}
                  onCellClick={(cell) => navigate(`/charts?symbol=${cell.id}`)}
                />
              </Show>
            </div>
          </Show>

          {/* Table View */}
          <Show when={viewMode() === 'table'}>
            <div class="flex-1 overflow-auto">
              {/* Loading State */}
              <Show when={loading()}>
                <div class="flex items-center justify-center h-full">
                  <div class="text-center">
                    <RefreshCw size={32} class="text-accent-500 mx-auto mb-4 animate-spin" />
                    <div class="text-gray-400">Scanning stocks...</div>
                  </div>
                </div>
              </Show>

              {/* No Results */}
              <Show when={results().length === 0 && !loading()}>
                <div class="flex items-center justify-center h-full">
                  <div class="text-center">
                    <Filter size={48} class="text-gray-600 mx-auto mb-4" />
                    <div class="text-gray-500 mb-2">No results</div>
                    <div class="text-xs text-gray-600">
                      Adjust your filters or select a preset screen
                    </div>
                  </div>
                </div>
              </Show>

              {/* Results Table */}
              <Show when={results().length > 0 && !loading()}>
                {/* Desktop Table */}
                <div class="hidden md:block">
                  <table class="w-full">
                    <thead class="sticky top-0 z-20">
                      <tr class="bg-terminal-850 border-b border-terminal-750">
                        <SortHeader column="symbol" label="Symbol" />
                        <SortHeader column="name" label="Company" />
                        
                        <Show when={activeTab() === 'overview'}>
                          <SortHeader column="price" label="Price" align="right" />
                          <SortHeader column="change_pct" label="Change" align="right" />
                          <SortHeader column="volume" label="Volume" align="right" />
                          <SortHeader column="market_cap" label="Market Cap" align="right" />
                          <SortHeader column="pe_ratio" label="P/E" align="right" />
                          <SortHeader column="sector" label="Sector" />
                        </Show>

                        <Show when={activeTab() === 'valuation'}>
                          <SortHeader column="market_cap" label="Market Cap" align="right" />
                          <SortHeader column="pe_ratio" label="P/E" align="right" />
                          <SortHeader column="forward_pe" label="Fwd P/E" align="right" />
                          <SortHeader column="dividend_yield" label="Div Yield" align="right" />
                          <SortHeader column="price" label="Price" align="right" />
                        </Show>

                        <Show when={activeTab() === 'performance'}>
                          <SortHeader column="change_pct" label="Change %" align="right" />
                          <SortHeader column="beta" label="Beta" align="right" />
                          <SortHeader column="week52_high" label="52W High" align="right" />
                          <SortHeader column="week52_low" label="52W Low" align="right" />
                          <SortHeader column="avg_volume" label="Avg Vol" align="right" />
                        </Show>
                      </tr>
                    </thead>
                    <tbody class="divide-y divide-terminal-800">
                      <For each={results()}>
                        {(result) => (
                          <tr class="hover:bg-terminal-850/50 transition-colors group">
                            <td class="px-3 py-3">
                              <div class="flex items-center gap-2">
                                <div class={`w-1.5 h-1.5 rounded-full ${
                                  result.change_pct >= 0 ? 'bg-success-500' : 'bg-danger-500'
                                }`} />
                                <button
                                  onClick={() => navigate(`/charts?symbol=${result.symbol}`)}
                                  class="text-sm font-bold text-white hover:text-accent-400 transition-colors cursor-pointer hover:underline"
                                >
                                  {result.symbol}
                                </button>
                              </div>
                            </td>
                            <td class="px-3 py-3">
                              <span class="text-xs text-gray-400 max-w-[180px] truncate block">
                                {result.name}
                              </span>
                            </td>

                            <Show when={activeTab() === 'overview'}>
                              <td class="px-3 py-3 text-right">
                                <span class="text-sm text-white font-mono tabular-nums">
                                  ${result.price?.toFixed(2) || '0.00'}
                                </span>
                              </td>
                              <td class="px-3 py-3 text-right">
                                <span class={`text-xs font-mono font-bold tabular-nums ${
                                  result.change_pct >= 0 ? 'text-success-500' : 'text-danger-500'
                                }`}>
                                  {result.change_pct >= 0 ? '+' : ''}{result.change_pct?.toFixed(2) || '0.00'}%
                                </span>
                              </td>
                              <td class="px-3 py-3 text-right">
                                <span class="text-xs text-gray-400 font-mono tabular-nums">
                                  {formatVol(result.volume)}
                                </span>
                              </td>
                              <td class="px-3 py-3 text-right">
                                <span class="text-xs text-gray-400 font-mono tabular-nums">
                                  {formatMarketCap(result.market_cap * 1000000)}
                                </span>
                              </td>
                              <td class="px-3 py-3 text-right">
                                <span class="text-xs text-gray-400 font-mono tabular-nums">
                                  {result.pe_ratio?.toFixed(1) || '-'}
                                </span>
                              </td>
                              <td class="px-3 py-3">
                                <span class="text-[10px] px-2 py-0.5 bg-terminal-800 text-gray-400 rounded">
                                  {result.sector}
                                </span>
                              </td>
                            </Show>

                            <Show when={activeTab() === 'valuation'}>
                              <td class="px-3 py-3 text-right">
                                <span class="text-xs text-gray-400 font-mono tabular-nums">
                                  {formatMarketCap(result.market_cap * 1000000)}
                                </span>
                              </td>
                              <td class="px-3 py-3 text-right">
                                <span class="text-xs text-gray-400 font-mono tabular-nums">
                                  {result.pe_ratio?.toFixed(2) || '-'}
                                </span>
                              </td>
                              <td class="px-3 py-3 text-right">
                                <span class="text-xs text-gray-400 font-mono tabular-nums">
                                  {result.forward_pe?.toFixed(2) || '-'}
                                </span>
                              </td>
                              <td class="px-3 py-3 text-right">
                                <span class="text-xs text-gray-400 font-mono tabular-nums">
                                  {result.dividend_yield ? `${result.dividend_yield.toFixed(2)}%` : '-'}
                                </span>
                              </td>
                              <td class="px-3 py-3 text-right">
                                <span class="text-sm text-white font-mono tabular-nums">
                                  ${result.price?.toFixed(2) || '0.00'}
                                </span>
                              </td>
                            </Show>

                            <Show when={activeTab() === 'performance'}>
                              <td class="px-3 py-3 text-right">
                                <span class={`text-xs font-mono font-bold tabular-nums ${
                                  result.change_pct >= 0 ? 'text-success-500' : 'text-danger-500'
                                }`}>
                                  {result.change_pct >= 0 ? '+' : ''}{result.change_pct?.toFixed(2) || '0.00'}%
                                </span>
                              </td>
                              <td class="px-3 py-3 text-right">
                                <span class="text-xs text-gray-400 font-mono tabular-nums">
                                  {result.beta?.toFixed(2) || '-'}
                                </span>
                              </td>
                              <td class="px-3 py-3 text-right">
                                <span class="text-xs text-gray-400 font-mono tabular-nums">
                                  {result.week52_high ? `$${result.week52_high.toFixed(2)}` : '-'}
                                </span>
                              </td>
                              <td class="px-3 py-3 text-right">
                                <span class="text-xs text-gray-400 font-mono tabular-nums">
                                  {result.week52_low ? `$${result.week52_low.toFixed(2)}` : '-'}
                                </span>
                              </td>
                              <td class="px-3 py-3 text-right">
                                <span class="text-xs text-gray-400 font-mono tabular-nums">
                                  {formatVol(result.avg_volume || 0)}
                                </span>
                              </td>
                            </Show>
                          </tr>
                        )}
                      </For>
                    </tbody>
                  </table>
                </div>

                {/* Mobile Card List */}
                <div class="md:hidden space-y-2 p-2">
                  <For each={results()}>
                    {(result) => (
                      <div 
                        class="bg-terminal-900 border border-terminal-800 rounded-lg p-3 active:bg-terminal-800 transition-colors"
                        onClick={() => navigate(`/charts?symbol=${result.symbol}`)}
                      >
                        <div class="flex justify-between items-start mb-2">
                          <div>
                            <div class="flex items-center gap-2">
                              <span class="font-bold text-white text-lg">{result.symbol}</span>
                              <div class={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                                result.change_pct >= 0 ? 'bg-success-900/20 text-success-400' : 'bg-danger-900/20 text-danger-400'
                              }`}>
                                {result.change_pct >= 0 ? '+' : ''}{result.change_pct?.toFixed(2) || '0.00'}%
                              </div>
                            </div>
                            <div class="text-xs text-gray-500 truncate max-w-[200px]">{result.name}</div>
                          </div>
                          <div class="text-right">
                            <div class="font-mono font-bold text-white text-lg">${result.price?.toFixed(2) || '0.00'}</div>
                            <div class="text-[10px] text-gray-500 font-mono">Vol: {formatVol(result.volume)}</div>
                          </div>
                        </div>
                        
                        <div class="grid grid-cols-3 gap-2 pt-2 border-t border-terminal-800 text-xs">
                          <div>
                            <div class="text-[10px] text-gray-500 uppercase">Mkt Cap</div>
                            <div class="font-mono text-gray-300">{formatMarketCap(result.market_cap * 1000000)}</div>
                          </div>
                          <div class="text-center">
                            <div class="text-[10px] text-gray-500 uppercase">P/E</div>
                            <div class="font-mono text-gray-300">{result.pe_ratio?.toFixed(2) || '-'}</div>
                          </div>
                          <div class="text-right">
                            <div class="text-[10px] text-gray-500 uppercase">Sector</div>
                            <div class="font-mono text-gray-300 truncate">{result.sector}</div>
                          </div>
                        </div>
                      </div>
                    )}
                  </For>
                </div>
              </Show>
            </div>

            {/* Pagination Footer */}
            <Show when={results().length > 0}>
              <div class="flex-shrink-0 bg-terminal-850 border-t border-terminal-750 px-4 py-2">
                <div class="flex items-center justify-between text-xs text-gray-500">
                  <span>
                    Showing {((page() - 1) * limit()) + 1} to {Math.min(page() * limit(), totalCount())} of {totalCount()} stocks
                  </span>
                  
                  <div class="flex items-center gap-2">
                    <button
                      onClick={() => handlePageChange(page() - 1)}
                      disabled={page() === 1}
                      class="p-1 hover:text-white disabled:opacity-30 disabled:hover:text-gray-500"
                    >
                      <ChevronLeft size={16} />
                    </button>
                    <span class="font-mono">
                      Page {page()} of {Math.ceil(totalCount() / limit())}
                    </span>
                    <button
                      onClick={() => handlePageChange(page() + 1)}
                      disabled={page() >= Math.ceil(totalCount() / limit())}
                      class="p-1 hover:text-white disabled:opacity-30 disabled:hover:text-gray-500"
                    >
                      <ChevronRight size={16} />
                    </button>
                  </div>
                </div>
              </div>
            </Show>
          </Show>
        </div>
      </div>
    </div>
  );
}
