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
  // Valuation metrics
  pe_ratio: number | null;
  forward_pe: number | null;
  peg_ratio: number | null;
  price_to_book: number | null;
  price_to_sales: number | null;
  eps: number | null;
  // Dividend
  dividend_yield: number | null;
  // Profitability
  profit_margin: number | null;
  operating_margin: number | null;
  roe: number | null;
  roa: number | null;
  revenue: number | null;
  net_income: number | null;
  // Analyst data
  analyst_rating: string | null;
  analyst_target: number | null;
  analyst_count: number | null;
  // 52-week data
  week52_high: number | null;
  week52_low: number | null;
  pct_from_52w_high: number | null;
  pct_from_52w_low: number | null;
  avg_volume: number | null;
  // Other
  beta: number | null;
  sector: string;
  industry: string;
  country: string;
  exchange: string;
  asset_type: string;
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

interface SavedScreen {
  id: string;
  name: string;
  criteria: any;
  created_at: string;
  last_run: string | null;
}

interface SectorInfo {
  name: string;
  count: number;
}

// Extended types for new features
interface TechnicalIndicators {
  symbol: string;
  rsi_14: number | null;
  sma_20: number | null;
  sma_50: number | null;
  sma_200: number | null;
  ema_20: number | null;
  current_price: number | null;
  macd?: { macd_line: number | null } | null;
  bollinger_bands?: { upper: number; middle: number; lower: number } | null;
}

interface PerformanceData {
  symbol: string;
  return_1w: number | null;
  return_1m: number | null;
  return_3m: number | null;
  return_6m: number | null;
  return_ytd: number | null;
  return_1y: number | null;
  volatility_30d: number | null;
}

interface ScatterDataPoint {
  symbol: string;
  name: string;
  sector: string;
  x: number | null;
  y: number | null;
  size: number;
}

interface BacktestItem {
  id: string;
  name: string;
  status: string;
  total_return: number | null;
  max_drawdown: number | null;
  created_at: string;
}

interface AlertItem {
  id: string;
  name: string;
  criteria: any;
  is_active: boolean;
  matched_symbols: string[];
  created_at: string;
}

type ViewTab = 'overview' | 'valuation' | 'performance' | 'profitability' | 'technical';

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
  
  // Saved Screens
  const [savedScreens, setSavedScreens] = createSignal<SavedScreen[]>([]);
  const [showSaveModal, setShowSaveModal] = createSignal(false);
  const [saveScreenName, setSaveScreenName] = createSignal('');
  
  // NEW: Technical Indicators & Performance Data
  const [technicalData, setTechnicalData] = createSignal<Record<string, TechnicalIndicators>>({});
  const [performanceData, setPerformanceData] = createSignal<Record<string, PerformanceData>>({});
  const [loadingTechnical, setLoadingTechnical] = createSignal(false);
  const [loadingPerformance, setLoadingPerformance] = createSignal(false);
  
  // NEW: Scatter Plot
  const [showScatterPlot, setShowScatterPlot] = createSignal(false);
  const [scatterData, setScatterData] = createSignal<ScatterDataPoint[]>([]);
  const [scatterXAxis, setScatterXAxis] = createSignal('pe_ratio');
  const [scatterYAxis, setScatterYAxis] = createSignal('roe');
  const [loadingScatter, setLoadingScatter] = createSignal(false);
  
  // NEW: Alerts & Backtests
  const [showAlertsPanel, setShowAlertsPanel] = createSignal(false);
  const [alerts, setAlerts] = createSignal<AlertItem[]>([]);
  const [showBacktestPanel, setShowBacktestPanel] = createSignal(false);
  const [backtests, setBacktests] = createSignal<BacktestItem[]>([]);
  const [newAlertName, setNewAlertName] = createSignal('');
  const [newBacktestName, setNewBacktestName] = createSignal('');
  const [backtestStartDate, setBacktestStartDate] = createSignal('');
  const [backtestEndDate, setBacktestEndDate] = createSignal('');
  
  // Preview count (debounced filter preview)
  const [previewCount, setPreviewCount] = createSignal<number | null>(null);
  const [previewLoading, setPreviewLoading] = createSignal(false);
  let previewTimeout: ReturnType<typeof setTimeout> | null = null;
  const [priceMin, setPriceMin] = createSignal<number | undefined>();
  const [priceMax, setPriceMax] = createSignal<number | undefined>();
  const [volumeMin, setVolumeMin] = createSignal<number | undefined>();
  const [marketCapMin, setMarketCapMin] = createSignal<number | undefined>();
  const [marketCapMax, setMarketCapMax] = createSignal<number | undefined>();
  const [changePctMin, setChangePctMin] = createSignal<number | undefined>();
  const [changePctMax, setChangePctMax] = createSignal<number | undefined>();
  const [selectedSector, setSelectedSector] = createSignal<string>('');
  const [selectedCountry, setSelectedCountry] = createSignal<string>('');
  
  // Filter states - Valuation
  const [peRatioMin, setPeRatioMin] = createSignal<number | undefined>();
  const [peRatioMax, setPeRatioMax] = createSignal<number | undefined>();
  const [forwardPeMin, setForwardPeMin] = createSignal<number | undefined>();
  const [forwardPeMax, setForwardPeMax] = createSignal<number | undefined>();
  const [pegRatioMin, setPegRatioMin] = createSignal<number | undefined>();
  const [pegRatioMax, setPegRatioMax] = createSignal<number | undefined>();
  const [priceToBkMin, setPriceToBkMin] = createSignal<number | undefined>();
  const [priceToBkMax, setPriceToBkMax] = createSignal<number | undefined>();
  const [priceToSalesMin, setPriceToSalesMin] = createSignal<number | undefined>();
  const [priceToSalesMax, setPriceToSalesMax] = createSignal<number | undefined>();
  
  // Filter states - Dividends & Income
  const [dividendMin, setDividendMin] = createSignal<number | undefined>();
  const [epsMin, setEpsMin] = createSignal<number | undefined>();
  
  // Filter states - Profitability
  const [profitMarginMin, setProfitMarginMin] = createSignal<number | undefined>();
  const [roeMin, setRoeMin] = createSignal<number | undefined>();
  const [roaMin, setRoaMin] = createSignal<number | undefined>();
  
  // Filter states - Risk (Beta not in DB yet)
  const [betaMin, setBetaMin] = createSignal<number | undefined>();
  const [betaMax, setBetaMax] = createSignal<number | undefined>();
  
  // Active Filters Tracking
  const getActiveFilters = () => {
    const filters: { key: string; label: string; value: string; clear: () => void }[] = [];
    
    if (priceMin() !== undefined || priceMax() !== undefined) {
      const label = priceMin() && priceMax() ? `$${priceMin()}-$${priceMax()}` :
                    priceMin() ? `>$${priceMin()}` : `<$${priceMax()}`;
      filters.push({ key: 'price', label: `Price: ${label}`, value: label, clear: () => { setPriceMin(undefined); setPriceMax(undefined); } });
    }
    if (marketCapMin() !== undefined || marketCapMax() !== undefined) {
      const label = marketCapMin() && marketCapMax() ? `${marketCapMin()}B-${marketCapMax()}B` :
                    marketCapMin() ? `>${marketCapMin()}B` : `<${marketCapMax()}B`;
      filters.push({ key: 'mcap', label: `Market Cap: ${label}`, value: label, clear: () => { setMarketCapMin(undefined); setMarketCapMax(undefined); } });
    }
    if (peRatioMin() !== undefined || peRatioMax() !== undefined) {
      const label = peRatioMin() && peRatioMax() ? `${peRatioMin()}-${peRatioMax()}` :
                    peRatioMin() ? `>${peRatioMin()}` : `<${peRatioMax()}`;
      filters.push({ key: 'pe', label: `P/E: ${label}`, value: label, clear: () => { setPeRatioMin(undefined); setPeRatioMax(undefined); } });
    }
    if (dividendMin() !== undefined) {
      filters.push({ key: 'div', label: `Dividend: >${dividendMin()}%`, value: `${dividendMin()}`, clear: () => setDividendMin(undefined) });
    }
    if (selectedSector()) {
      filters.push({ key: 'sector', label: `Sector: ${selectedSector()}`, value: selectedSector(), clear: () => setSelectedSector('') });
    }
    if (changePctMin() !== undefined || changePctMax() !== undefined) {
      const label = changePctMin() && changePctMax() ? `${changePctMin()}%-${changePctMax()}%` :
                    changePctMin() ? `>${changePctMin()}%` : `<${changePctMax()}%`;
      filters.push({ key: 'chg', label: `Change: ${label}`, value: label, clear: () => { setChangePctMin(undefined); setChangePctMax(undefined); } });
    }
    if (profitMarginMin() !== undefined) {
      filters.push({ key: 'pm', label: `Margin: >${profitMarginMin()}%`, value: `${profitMarginMin()}`, clear: () => setProfitMarginMin(undefined) });
    }
    if (roeMin() !== undefined) {
      filters.push({ key: 'roe', label: `ROE: >${roeMin()}%`, value: `${roeMin()}`, clear: () => setRoeMin(undefined) });
    }
    if (epsMin() !== undefined) {
      filters.push({ key: 'eps', label: `EPS: >$${epsMin()}`, value: `${epsMin()}`, clear: () => setEpsMin(undefined) });
    }
    
    return filters;
  };

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
    await Promise.all([loadPresets(), loadSectors(), loadSavedScreens()]);
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

  const loadSavedScreens = async () => {
    try {
      const token = localStorage.getItem('auth_token');
      if (!token) return;
      
      const response = await fetch('/api/v1/screener/saved', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setSavedScreens(data.screens || []);
      }
    } catch (err) {
      console.error('Failed to load saved screens:', err);
    }
  };

  // NEW: Load technical indicators for current results
  const loadTechnicalData = async () => {
    const symbols = results().map(r => r.symbol).slice(0, 20); // Limit to 20
    if (symbols.length === 0) return;
    
    setLoadingTechnical(true);
    try {
      const response = await fetch(`/api/v1/screener/technical/bulk?symbols=${symbols.join(',')}&timeframe=1d`);
      if (response.ok) {
        const data = await response.json();
        setTechnicalData(data.data || {});
      }
    } catch (err) {
      console.error('Failed to load technical data:', err);
    } finally {
      setLoadingTechnical(false);
    }
  };

  // NEW: Load performance metrics for current results
  const loadPerformanceData = async () => {
    const symbols = results().map(r => r.symbol).slice(0, 20);
    if (symbols.length === 0) return;
    
    setLoadingPerformance(true);
    try {
      const response = await fetch(`/api/v1/screener/performance/bulk?symbols=${symbols.join(',')}`);
      if (response.ok) {
        const data = await response.json();
        setPerformanceData(data.data || {});
      }
    } catch (err) {
      console.error('Failed to load performance data:', err);
    } finally {
      setLoadingPerformance(false);
    }
  };

  // NEW: Load scatter plot data
  const loadScatterData = async () => {
    setLoadingScatter(true);
    try {
      const params = new URLSearchParams({
        x_axis: scatterXAxis(),
        y_axis: scatterYAxis(),
      });
      if (selectedSector()) {
        params.append('sector', selectedSector());
      }
      const response = await fetch(`/api/v1/screener/scatter-data?${params}`);
      if (response.ok) {
        const data = await response.json();
        setScatterData(data.data || []);
      }
    } catch (err) {
      console.error('Failed to load scatter data:', err);
    } finally {
      setLoadingScatter(false);
    }
  };

  // NEW: Load alerts
  const loadAlerts = async () => {
    const token = localStorage.getItem('auth_token');
    if (!token) return;
    
    try {
      const response = await fetch('/api/v1/screener/alerts', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setAlerts(data.alerts || []);
      }
    } catch (err) {
      console.error('Failed to load alerts:', err);
    }
  };

  // NEW: Create alert from current filters
  const createAlert = async () => {
    const name = newAlertName().trim();
    if (!name) {
      alert('Please enter an alert name');
      return;
    }
    
    const token = localStorage.getItem('auth_token');
    if (!token) {
      alert('Please log in to create alerts');
      return;
    }
    
    try {
      const response = await fetch('/api/v1/screener/alerts', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          name,
          criteria: getCriteria(),
          notify_email: true,
        }),
      });
      
      if (response.ok) {
        await loadAlerts();
        setNewAlertName('');
        alert('Alert created! You\'ll be notified when stocks match your criteria.');
      } else {
        const data = await response.json();
        alert(data.detail || 'Failed to create alert');
      }
    } catch (err) {
      console.error('Failed to create alert:', err);
    }
  };

  // NEW: Delete alert
  const deleteAlert = async (alertId: string) => {
    if (!confirm('Delete this alert?')) return;
    
    const token = localStorage.getItem('auth_token');
    if (!token) return;
    
    try {
      const response = await fetch(`/api/v1/screener/alerts/${alertId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        await loadAlerts();
      }
    } catch (err) {
      console.error('Failed to delete alert:', err);
    }
  };

  // NEW: Load backtests
  const loadBacktests = async () => {
    const token = localStorage.getItem('auth_token');
    if (!token) return;
    
    try {
      const response = await fetch('/api/v1/screener/backtests', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (response.ok) {
        const data = await response.json();
        setBacktests(data.backtests || []);
      }
    } catch (err) {
      console.error('Failed to load backtests:', err);
    }
  };

  // NEW: Create backtest
  const createBacktest = async () => {
    const name = newBacktestName().trim();
    if (!name || !backtestStartDate() || !backtestEndDate()) {
      alert('Please enter backtest name and date range');
      return;
    }
    
    const token = localStorage.getItem('auth_token');
    if (!token) {
      alert('Please log in to create backtests');
      return;
    }
    
    try {
      const response = await fetch('/api/v1/screener/backtests', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          name,
          criteria: getCriteria(),
          start_date: backtestStartDate(),
          end_date: backtestEndDate(),
          initial_capital: 100000,
          rebalance_frequency: 'monthly',
          position_sizing: 'equal_weight',
          max_positions: 20,
        }),
      });
      
      if (response.ok) {
        await loadBacktests();
        setNewBacktestName('');
        setBacktestStartDate('');
        setBacktestEndDate('');
        alert('Backtest started! Results will be available shortly.');
      } else {
        const data = await response.json();
        alert(data.detail || 'Failed to create backtest');
      }
    } catch (err) {
      console.error('Failed to create backtest:', err);
    }
  };

  // NEW: Load technical data when switching to technical tab
  createEffect(() => {
    if (activeTab() === 'technical' && results().length > 0 && Object.keys(technicalData()).length === 0) {
      loadTechnicalData();
    }
  });

  // NEW: Load scatter data when opening scatter plot
  createEffect(() => {
    if (showScatterPlot()) {
      loadScatterData();
    }
  });

  const saveCurrentScreen = async () => {
    const name = saveScreenName().trim();
    if (!name) return;
    
    const token = localStorage.getItem('auth_token');
    if (!token) {
      alert('Please log in to save screens');
      return;
    }
    
    try {
      const response = await fetch('/api/v1/screener/saved', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          name,
          criteria: getCriteria()
        }),
      });
      
      if (response.ok) {
        await loadSavedScreens();
        setShowSaveModal(false);
        setSaveScreenName('');
      } else {
        const data = await response.json();
        alert(data.detail || 'Failed to save screen');
      }
    } catch (err) {
      console.error('Failed to save screen:', err);
      alert('Failed to save screen');
    }
  };

  const deleteSavedScreen = async (screenId: string) => {
    if (!confirm('Delete this saved screen?')) return;
    
    const token = localStorage.getItem('auth_token');
    if (!token) return;
    
    try {
      const response = await fetch(`/api/v1/screener/saved/${screenId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      
      if (response.ok) {
        await loadSavedScreens();
      }
    } catch (err) {
      console.error('Failed to delete screen:', err);
    }
  };

  const applySavedScreen = async (screen: SavedScreen) => {
    resetFilters(false);
    const c = screen.criteria;
    
    // Apply all filters from saved criteria
    if (c.price_min) setPriceMin(c.price_min);
    if (c.price_max) setPriceMax(c.price_max);
    if (c.volume_min) setVolumeMin(c.volume_min);
    if (c.market_cap_min) setMarketCapMin(c.market_cap_min / 1000); // Convert M to B
    if (c.market_cap_max) setMarketCapMax(c.market_cap_max / 1000);
    if (c.pe_ratio_min) setPeRatioMin(c.pe_ratio_min);
    if (c.pe_ratio_max) setPeRatioMax(c.pe_ratio_max);
    if (c.forward_pe_min) setForwardPeMin(c.forward_pe_min);
    if (c.forward_pe_max) setForwardPeMax(c.forward_pe_max);
    if (c.dividend_yield_min) setDividendMin(c.dividend_yield_min);
    if (c.change_pct_min) setChangePctMin(c.change_pct_min);
    if (c.change_pct_max) setChangePctMax(c.change_pct_max);
    if (c.sector) setSelectedSector(c.sector);
    if (c.country) setSelectedCountry(c.country);
    if (c.eps_min) setEpsMin(c.eps_min);
    if (c.profit_margin_min) setProfitMarginMin(c.profit_margin_min);
    if (c.roe_min) setRoeMin(c.roe_min);
    
    await handleScan(c, 1);
  };

  const getCriteria = () => ({
    // Basic filters
    price_min: priceMin(),
    price_max: priceMax(),
    volume_min: volumeMin(),
    // Convert Billions (UI) to Millions (DB)
    market_cap_min: marketCapMin() ? marketCapMin()! * 1000 : undefined,
    market_cap_max: marketCapMax() ? marketCapMax()! * 1000 : undefined,
    change_pct_min: changePctMin(),
    change_pct_max: changePctMax(),
    sector: selectedSector() || undefined,
    country: selectedCountry() || undefined,
    
    // Valuation filters
    pe_ratio_min: peRatioMin(),
    pe_ratio_max: peRatioMax(),
    forward_pe_min: forwardPeMin(),
    forward_pe_max: forwardPeMax(),
    peg_ratio_min: pegRatioMin(),
    peg_ratio_max: pegRatioMax(),
    price_to_book_min: priceToBkMin(),
    price_to_book_max: priceToBkMax(),
    price_to_sales_min: priceToSalesMin(),
    price_to_sales_max: priceToSalesMax(),
    
    // Income filters
    dividend_yield_min: dividendMin() ? dividendMin()! / 100 : undefined, // Convert % to decimal (DB stores as decimal 0.0247)
    eps_min: epsMin(),
    
    // Profitability filters (DB stores as percentage like 26.92, not 0.2692)
    profit_margin_min: profitMarginMin(),
    roe_min: roeMin(),
    roa_min: roaMin(),
    
    // Risk filters (beta not yet in DB)
    beta_min: betaMin(),
    beta_max: betaMax(),
  });

  // Debounced preview count - updates as user changes filters
  const updatePreviewCount = async () => {
    setPreviewLoading(true);
    try {
      const criteria = getCriteria();
      const response = await fetch(`/api/v1/screener/scan?limit=0&offset=0`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(criteria),
      });
      
      if (response.ok) {
        const data = await response.json();
        setPreviewCount(data.total_count || 0);
      }
    } catch (err) {
      console.error('Preview count error:', err);
    } finally {
      setPreviewLoading(false);
    }
  };

  // Trigger preview count on filter changes (debounced)
  createEffect(() => {
    // Track all filter values
    const _ = [
      priceMin(), priceMax(), volumeMin(), marketCapMin(), marketCapMax(),
      changePctMin(), changePctMax(), selectedSector(), selectedCountry(),
      peRatioMin(), peRatioMax(), forwardPeMin(), forwardPeMax(),
      pegRatioMin(), pegRatioMax(), priceToBkMin(), priceToBkMax(),
      priceToSalesMin(), priceToSalesMax(), dividendMin(), epsMin(),
      profitMarginMin(), roeMin(), roaMin(), betaMin(), betaMax()
    ];
    
    // Debounce the preview
    if (previewTimeout) clearTimeout(previewTimeout);
    previewTimeout = setTimeout(updatePreviewCount, 500);
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
    // New filters
    setForwardPeMin(undefined);
    setForwardPeMax(undefined);
    setPegRatioMin(undefined);
    setPegRatioMax(undefined);
    setPriceToBkMin(undefined);
    setPriceToBkMax(undefined);
    setPriceToSalesMin(undefined);
    setPriceToSalesMax(undefined);
    setEpsMin(undefined);
    setProfitMarginMin(undefined);
    setRoeMin(undefined);
    setRoaMin(undefined);
    
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

      {/* Active Filters Badges */}
      <Show when={getActiveFilters().length > 0}>
        <div class="flex-shrink-0 bg-terminal-900/30 border-b border-terminal-750 px-4 py-2 overflow-x-auto">
          <div class="flex items-center gap-2 flex-wrap">
            <span class="text-xs text-gray-500 font-medium">Active Filters:</span>
            <For each={getActiveFilters()}>
              {(filter) => (
                <button
                  onClick={() => { filter.clear(); handleScan(undefined, 1); }}
                  class="flex-shrink-0 px-2 py-1 flex items-center gap-1.5 text-xs font-medium rounded-full bg-accent-500/20 text-accent-400 border border-accent-500/30 hover:bg-accent-500/30 transition-all group"
                >
                  <span>{filter.label}</span>
                  <X size={10} class="opacity-50 group-hover:opacity-100" />
                </button>
              )}
            </For>
            <button
              onClick={() => resetFilters()}
              class="text-xs text-gray-500 hover:text-white transition-colors underline"
            >
              Clear All
            </button>
          </div>
        </div>
      </Show>

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
              
              {/* Quick Filter Presets */}
              <div class="pb-3 border-b border-terminal-750">
                <label class="text-xs font-semibold text-accent-400 uppercase tracking-wide mb-2 block">
                  Quick Filters
                </label>
                <div class="grid grid-cols-2 gap-1.5">
                  <button
                    onClick={() => { setPeRatioMax(15); setDividendMin(2); handleScan(undefined, 1); }}
                    class="px-2 py-1.5 text-[10px] font-medium bg-terminal-850 hover:bg-terminal-800 border border-terminal-750 text-gray-400 hover:text-white rounded transition-all"
                  >
                    💎 Value
                  </button>
                  <button
                    onClick={() => { setPeRatioMin(25); setMarketCapMin(100); handleScan(undefined, 1); }}
                    class="px-2 py-1.5 text-[10px] font-medium bg-terminal-850 hover:bg-terminal-800 border border-terminal-750 text-gray-400 hover:text-white rounded transition-all"
                  >
                    🚀 Growth
                  </button>
                  <button
                    onClick={() => { setDividendMin(3); handleScan(undefined, 1); }}
                    class="px-2 py-1.5 text-[10px] font-medium bg-terminal-850 hover:bg-terminal-800 border border-terminal-750 text-gray-400 hover:text-white rounded transition-all"
                  >
                    💰 Dividend
                  </button>
                  <button
                    onClick={() => { setMarketCapMin(500); handleScan(undefined, 1); }}
                    class="px-2 py-1.5 text-[10px] font-medium bg-terminal-850 hover:bg-terminal-800 border border-terminal-750 text-gray-400 hover:text-white rounded transition-all"
                  >
                    🏢 Mega Cap
                  </button>
                </div>
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

              {/* Advanced Valuation Section */}
              <div class="pt-2 border-t border-terminal-750">
                <label class="text-xs font-semibold text-accent-400 uppercase tracking-wide mb-3 block">
                  Advanced Valuation
                </label>
                
                {/* Price-to-Book */}
                <div class="mb-3">
                  <label class="text-xs text-gray-500 mb-1 block">Price/Book</label>
                  <div class="grid grid-cols-2 gap-2">
                    <input
                      type="number"
                      value={priceToBkMin() ?? ''}
                      onInput={(e) => setPriceToBkMin(e.target.value ? parseFloat(e.target.value) : undefined)}
                      placeholder="Min"
                      class="w-full bg-terminal-850 border border-terminal-750 text-white text-xs px-3 py-2 rounded focus:outline-none focus:border-accent-500"
                    />
                    <input
                      type="number"
                      value={priceToBkMax() ?? ''}
                      onInput={(e) => setPriceToBkMax(e.target.value ? parseFloat(e.target.value) : undefined)}
                      placeholder="Max"
                      class="w-full bg-terminal-850 border border-terminal-750 text-white text-xs px-3 py-2 rounded focus:outline-none focus:border-accent-500"
                    />
                  </div>
                </div>
                
                {/* Price-to-Sales */}
                <div class="mb-3">
                  <label class="text-xs text-gray-500 mb-1 block">Price/Sales</label>
                  <div class="grid grid-cols-2 gap-2">
                    <input
                      type="number"
                      value={priceToSalesMin() ?? ''}
                      onInput={(e) => setPriceToSalesMin(e.target.value ? parseFloat(e.target.value) : undefined)}
                      placeholder="Min"
                      class="w-full bg-terminal-850 border border-terminal-750 text-white text-xs px-3 py-2 rounded focus:outline-none focus:border-accent-500"
                    />
                    <input
                      type="number"
                      value={priceToSalesMax() ?? ''}
                      onInput={(e) => setPriceToSalesMax(e.target.value ? parseFloat(e.target.value) : undefined)}
                      placeholder="Max"
                      class="w-full bg-terminal-850 border border-terminal-750 text-white text-xs px-3 py-2 rounded focus:outline-none focus:border-accent-500"
                    />
                  </div>
                </div>
              </div>

              {/* Profitability Section */}
              <div class="pt-2 border-t border-terminal-750">
                <label class="text-xs font-semibold text-accent-400 uppercase tracking-wide mb-3 block">
                  Profitability
                </label>
                
                {/* Min EPS */}
                <div class="mb-3">
                  <label class="text-xs text-gray-500 mb-1 block">Min EPS ($)</label>
                  <input
                    type="number"
                    value={epsMin() ?? ''}
                    onInput={(e) => setEpsMin(e.target.value ? parseFloat(e.target.value) : undefined)}
                    placeholder="e.g. 2.00"
                    class="w-full bg-terminal-850 border border-terminal-750 text-white text-xs px-3 py-2 rounded focus:outline-none focus:border-accent-500"
                  />
                </div>
                
                {/* Profit Margin */}
                <div class="mb-3">
                  <label class="text-xs text-gray-500 mb-1 block">Min Profit Margin (%)</label>
                  <input
                    type="number"
                    value={profitMarginMin() ?? ''}
                    onInput={(e) => setProfitMarginMin(e.target.value ? parseFloat(e.target.value) : undefined)}
                    placeholder="e.g. 10"
                    class="w-full bg-terminal-850 border border-terminal-750 text-white text-xs px-3 py-2 rounded focus:outline-none focus:border-accent-500"
                  />
                </div>
                
                {/* ROE */}
                <div class="mb-3">
                  <label class="text-xs text-gray-500 mb-1 block">Min ROE (%)</label>
                  <input
                    type="number"
                    value={roeMin() ?? ''}
                    onInput={(e) => setRoeMin(e.target.value ? parseFloat(e.target.value) : undefined)}
                    placeholder="e.g. 15"
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
                {/* Preview Count */}
                <Show when={previewCount() !== null}>
                  <div class="text-center py-2 px-3 bg-terminal-850 border border-terminal-700 rounded">
                    <span class="text-xs text-gray-400">
                      {previewLoading() ? (
                        <span class="animate-pulse">Counting...</span>
                      ) : (
                        <>
                          <span class="text-lg font-bold text-white">{previewCount()}</span>
                          <span class="ml-1">stocks match</span>
                        </>
                      )}
                    </span>
                  </div>
                </Show>
                
                <button
                  onClick={() => handleScan(undefined, 1)}
                  disabled={loading()}
                  class="w-full flex items-center justify-center gap-2 px-4 py-3 bg-accent-500 hover:bg-accent-600 disabled:opacity-50 text-white font-semibold rounded transition-colors"
                >
                  <Search size={16} class={loading() ? 'animate-pulse' : ''} />
                  <span>{loading() ? 'Scanning...' : 'Run Screen'}</span>
                </button>

                <button
                  onClick={() => setShowSaveModal(true)}
                  class="w-full flex items-center justify-center gap-2 px-4 py-2 bg-terminal-850 hover:bg-terminal-800 border border-primary-500/50 text-primary-400 hover:text-primary-300 text-sm rounded transition-colors"
                >
                  <Save size={14} />
                  <span>Save Screen</span>
                </button>

                {/* NEW: Scatter Plot Button */}
                <button
                  onClick={() => { setShowScatterPlot(!showScatterPlot()); if (!showScatterPlot()) loadScatterData(); }}
                  class="w-full flex items-center justify-center gap-2 px-4 py-2 bg-terminal-850 hover:bg-terminal-800 border border-warning-500/50 text-warning-400 hover:text-warning-300 text-sm rounded transition-colors"
                >
                  <PieChart size={14} />
                  <span>Scatter Plot</span>
                </button>

                {/* NEW: Alert Button */}
                <button
                  onClick={() => { setShowAlertsPanel(!showAlertsPanel()); if (!showAlertsPanel()) loadAlerts(); }}
                  class="w-full flex items-center justify-center gap-2 px-4 py-2 bg-terminal-850 hover:bg-terminal-800 border border-danger-500/50 text-danger-400 hover:text-danger-300 text-sm rounded transition-colors"
                >
                  <Zap size={14} />
                  <span>Create Alert</span>
                </button>

                {/* NEW: Backtest Button */}
                <button
                  onClick={() => { setShowBacktestPanel(!showBacktestPanel()); if (!showBacktestPanel()) loadBacktests(); }}
                  class="w-full flex items-center justify-center gap-2 px-4 py-2 bg-terminal-850 hover:bg-terminal-800 border border-success-500/50 text-success-400 hover:text-success-300 text-sm rounded transition-colors"
                >
                  <BarChart3 size={14} />
                  <span>Backtest</span>
                </button>

                <button
                  onClick={() => resetFilters()}
                  class="w-full px-4 py-2 bg-terminal-850 hover:bg-terminal-800 border border-terminal-750 text-gray-400 hover:text-white text-sm rounded transition-colors"
                >
                  Reset All Filters
                </button>
              </div>

              {/* Saved Screens Section */}
              <Show when={savedScreens().length > 0}>
                <div class="pt-4 border-t border-terminal-750 mt-4">
                  <label class="text-xs font-semibold text-primary-400 uppercase tracking-wide mb-2 block">
                    Your Saved Screens
                  </label>
                  <div class="space-y-1.5">
                    <For each={savedScreens()}>
                      {(screen) => (
                        <div class="flex items-center gap-1 group">
                          <button
                            onClick={() => applySavedScreen(screen)}
                            class="flex-1 px-2 py-1.5 text-left text-xs bg-terminal-850 hover:bg-terminal-800 border border-terminal-750 hover:border-primary-500/50 text-gray-300 hover:text-white rounded transition-all truncate"
                            title={`Created: ${new Date(screen.created_at).toLocaleDateString()}`}
                          >
                            <Star size={10} class="inline mr-1.5 text-yellow-500" />
                            {screen.name}
                          </button>
                          <button
                            onClick={() => deleteSavedScreen(screen.id)}
                            class="p-1.5 text-gray-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all"
                            title="Delete"
                          >
                            <Trash2 size={12} />
                          </button>
                        </div>
                      )}
                    </For>
                  </div>
                </div>
              </Show>
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
                <button
                  onClick={() => setActiveTab('profitability')}
                  class={`py-3 text-xs font-medium border-b-2 transition-colors ${
                    activeTab() === 'profitability'
                      ? 'border-accent-500 text-white'
                      : 'border-transparent text-gray-400 hover:text-white'
                  }`}
                >
                  Profitability
                </button>
                <button
                  onClick={() => { setActiveTab('technical'); loadTechnicalData(); }}
                  class={`py-3 text-xs font-medium border-b-2 transition-colors ${
                    activeTab() === 'technical'
                      ? 'border-accent-500 text-white'
                      : 'border-transparent text-gray-400 hover:text-white'
                  }`}
                >
                  Technical
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
                          <SortHeader column="price_to_book" label="P/B" align="right" />
                          <SortHeader column="price_to_sales" label="P/S" align="right" />
                          <SortHeader column="eps" label="EPS" align="right" />
                          <SortHeader column="dividend_yield" label="Div %" align="right" />
                        </Show>

                        <Show when={activeTab() === 'performance'}>
                          <SortHeader column="change_pct" label="Change %" align="right" />
                          <SortHeader column="price" label="Price" align="right" />
                          <th class="px-3 py-3 text-xs font-semibold text-gray-400 text-right sticky top-0 bg-terminal-850 z-10">
                            52W Range
                          </th>
                          <th class="px-3 py-3 text-xs font-semibold text-gray-400 text-right sticky top-0 bg-terminal-850 z-10">
                            From High
                          </th>
                          <SortHeader column="volume" label="Volume" align="right" />
                        </Show>

                        <Show when={activeTab() === 'profitability'}>
                          <SortHeader column="price" label="Price" align="right" />
                          <SortHeader column="roe" label="ROE %" align="right" />
                          <SortHeader column="roa" label="ROA %" align="right" />
                          <SortHeader column="profit_margin" label="Profit Margin" align="right" />
                          <SortHeader column="operating_margin" label="Op Margin" align="right" />
                          <SortHeader column="eps" label="EPS" align="right" />
                        </Show>

                        <Show when={activeTab() === 'technical'}>
                          <SortHeader column="price" label="Price" align="right" />
                          <th class="px-3 py-3 text-xs font-semibold text-gray-400 text-right sticky top-0 bg-terminal-850 z-10">
                            RSI (14)
                          </th>
                          <th class="px-3 py-3 text-xs font-semibold text-gray-400 text-right sticky top-0 bg-terminal-850 z-10">
                            SMA 20
                          </th>
                          <th class="px-3 py-3 text-xs font-semibold text-gray-400 text-right sticky top-0 bg-terminal-850 z-10">
                            SMA 50
                          </th>
                          <th class="px-3 py-3 text-xs font-semibold text-gray-400 text-center sticky top-0 bg-terminal-850 z-10">
                            Above SMA20
                          </th>
                          <th class="px-3 py-3 text-xs font-semibold text-gray-400 text-center sticky top-0 bg-terminal-850 z-10">
                            Above SMA50
                          </th>
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
                                  {formatMarketCap(result.market_cap)}
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
                                  {formatMarketCap(result.market_cap)}
                                </span>
                              </td>
                              <td class="px-3 py-3 text-right">
                                <span class={`text-xs font-mono tabular-nums ${
                                  result.pe_ratio && result.pe_ratio < 15 ? 'text-success-400' :
                                  result.pe_ratio && result.pe_ratio > 30 ? 'text-warning-400' : 'text-gray-400'
                                }`}>
                                  {result.pe_ratio?.toFixed(1) || '-'}
                                </span>
                              </td>
                              <td class="px-3 py-3 text-right">
                                <span class={`text-xs font-mono tabular-nums ${
                                  result.price_to_book && result.price_to_book < 2 ? 'text-success-400' : 'text-gray-400'
                                }`}>
                                  {result.price_to_book?.toFixed(2) || '-'}
                                </span>
                              </td>
                              <td class="px-3 py-3 text-right">
                                <span class={`text-xs font-mono tabular-nums ${
                                  result.price_to_sales && result.price_to_sales < 3 ? 'text-success-400' : 'text-gray-400'
                                }`}>
                                  {result.price_to_sales?.toFixed(2) || '-'}
                                </span>
                              </td>
                              <td class="px-3 py-3 text-right">
                                <span class={`text-xs font-mono tabular-nums ${
                                  result.eps && result.eps > 0 ? 'text-success-400' : 
                                  result.eps && result.eps < 0 ? 'text-danger-400' : 'text-gray-400'
                                }`}>
                                  {result.eps ? `$${result.eps.toFixed(2)}` : '-'}
                                </span>
                              </td>
                              <td class="px-3 py-3 text-right">
                                <span class={`text-xs font-mono tabular-nums ${
                                  result.dividend_yield && result.dividend_yield > 0.02 ? 'text-success-400' : 'text-gray-400'
                                }`}>
                                  {result.dividend_yield ? `${(result.dividend_yield * 100).toFixed(2)}%` : '-'}
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
                              {/* Price */}
                              <td class="px-3 py-3 text-right">
                                <span class="text-sm text-white font-mono tabular-nums">
                                  ${result.price?.toFixed(2) || '0.00'}
                                </span>
                              </td>
                              {/* 52-Week Range Visual Bar */}
                              <td class="px-3 py-3">
                                <Show when={result.week52_low && result.week52_high} fallback={
                                  <span class="text-xs text-gray-600">-</span>
                                }>
                                  <div class="flex items-center gap-2 min-w-[120px]">
                                    <span class="text-[10px] text-gray-500 font-mono w-12 text-right">
                                      ${result.week52_low?.toFixed(0)}
                                    </span>
                                    <div class="flex-1 h-1.5 bg-terminal-800 rounded-full relative">
                                      {(() => {
                                        const low = result.week52_low || 0;
                                        const high = result.week52_high || 1;
                                        const range = high - low;
                                        const position = range > 0 ? ((result.price - low) / range) * 100 : 50;
                                        return (
                                          <div 
                                            class="absolute top-1/2 -translate-y-1/2 w-2 h-2 bg-accent-500 rounded-full shadow-lg"
                                            style={{ left: `${Math.min(100, Math.max(0, position))}%`, transform: 'translate(-50%, -50%)' }}
                                          />
                                        );
                                      })()}
                                    </div>
                                    <span class="text-[10px] text-gray-500 font-mono w-12">
                                      ${result.week52_high?.toFixed(0)}
                                    </span>
                                  </div>
                                </Show>
                              </td>
                              {/* % From 52W High */}
                              <td class="px-3 py-3 text-right">
                                <span class={`text-xs font-mono tabular-nums ${
                                  result.pct_from_52w_high !== null && result.pct_from_52w_high > -10 
                                    ? 'text-success-400' 
                                    : result.pct_from_52w_high !== null && result.pct_from_52w_high < -30 
                                    ? 'text-danger-400' 
                                    : 'text-gray-400'
                                }`}>
                                  {result.pct_from_52w_high !== null ? `${result.pct_from_52w_high.toFixed(1)}%` : '-'}
                                </span>
                              </td>
                              {/* Volume */}
                              <td class="px-3 py-3 text-right">
                                <span class="text-xs text-gray-400 font-mono tabular-nums">
                                  {formatVol(result.volume || 0)}
                                </span>
                              </td>
                            </Show>

                            <Show when={activeTab() === 'profitability'}>
                              {/* Price */}
                              <td class="px-3 py-3 text-right">
                                <span class="text-sm text-white font-mono tabular-nums">
                                  ${result.price?.toFixed(2) || '0.00'}
                                </span>
                              </td>
                              {/* ROE */}
                              <td class="px-3 py-3 text-right">
                                <span class={`text-xs font-mono tabular-nums ${
                                  result.roe && result.roe > 20 ? 'text-success-400' :
                                  result.roe && result.roe < 10 ? 'text-warning-400' : 'text-gray-400'
                                }`}>
                                  {result.roe ? `${result.roe.toFixed(1)}%` : '-'}
                                </span>
                              </td>
                              {/* ROA */}
                              <td class="px-3 py-3 text-right">
                                <span class={`text-xs font-mono tabular-nums ${
                                  result.roa && result.roa > 10 ? 'text-success-400' : 'text-gray-400'
                                }`}>
                                  {result.roa ? `${result.roa.toFixed(1)}%` : '-'}
                                </span>
                              </td>
                              {/* Profit Margin */}
                              <td class="px-3 py-3 text-right">
                                <span class={`text-xs font-mono tabular-nums ${
                                  result.profit_margin && result.profit_margin > 20 ? 'text-success-400' :
                                  result.profit_margin && result.profit_margin < 5 ? 'text-danger-400' : 'text-gray-400'
                                }`}>
                                  {result.profit_margin ? `${result.profit_margin.toFixed(1)}%` : '-'}
                                </span>
                              </td>
                              {/* Operating Margin */}
                              <td class="px-3 py-3 text-right">
                                <span class={`text-xs font-mono tabular-nums ${
                                  result.operating_margin && result.operating_margin > 20 ? 'text-success-400' : 'text-gray-400'
                                }`}>
                                  {result.operating_margin ? `${result.operating_margin.toFixed(1)}%` : '-'}
                                </span>
                              </td>
                              {/* EPS */}
                              <td class="px-3 py-3 text-right">
                                <span class={`text-xs font-mono tabular-nums ${
                                  result.eps && result.eps > 0 ? 'text-success-400' : 
                                  result.eps && result.eps < 0 ? 'text-danger-400' : 'text-gray-400'
                                }`}>
                                  {result.eps ? `$${result.eps.toFixed(2)}` : '-'}
                                </span>
                              </td>
                            </Show>

                            <Show when={activeTab() === 'technical'}>
                              {/* Price */}
                              <td class="px-3 py-3 text-right">
                                <span class="text-sm text-white font-mono tabular-nums">
                                  ${result.price?.toFixed(2) || '0.00'}
                                </span>
                              </td>
                              {/* RSI */}
                              <td class="px-3 py-3 text-right">
                                {(() => {
                                  const tech = technicalData()[result.symbol];
                                  const rsi = tech?.rsi_14;
                                  return (
                                    <span class={`text-xs font-mono tabular-nums ${
                                      rsi && rsi > 70 ? 'text-danger-400' :
                                      rsi && rsi < 30 ? 'text-success-400' : 'text-gray-400'
                                    }`}>
                                      {rsi ? rsi.toFixed(1) : loadingTechnical() ? '...' : '-'}
                                    </span>
                                  );
                                })()}
                              </td>
                              {/* SMA 20 */}
                              <td class="px-3 py-3 text-right">
                                {(() => {
                                  const tech = technicalData()[result.symbol];
                                  return (
                                    <span class="text-xs font-mono tabular-nums text-gray-400">
                                      {tech?.sma_20 ? `$${tech.sma_20.toFixed(2)}` : loadingTechnical() ? '...' : '-'}
                                    </span>
                                  );
                                })()}
                              </td>
                              {/* SMA 50 */}
                              <td class="px-3 py-3 text-right">
                                {(() => {
                                  const tech = technicalData()[result.symbol];
                                  return (
                                    <span class="text-xs font-mono tabular-nums text-gray-400">
                                      {tech?.sma_50 ? `$${tech.sma_50.toFixed(2)}` : loadingTechnical() ? '...' : '-'}
                                    </span>
                                  );
                                })()}
                              </td>
                              {/* Above SMA20 */}
                              <td class="px-3 py-3 text-center">
                                {(() => {
                                  const tech = technicalData()[result.symbol];
                                  const above = tech?.current_price && tech?.sma_20 && tech.current_price > tech.sma_20;
                                  return loadingTechnical() ? (
                                    <span class="text-xs text-gray-500">...</span>
                                  ) : tech ? (
                                    <span class={`text-xs font-bold ${above ? 'text-success-400' : 'text-danger-400'}`}>
                                      {above ? '✓' : '✗'}
                                    </span>
                                  ) : <span class="text-xs text-gray-600">-</span>;
                                })()}
                              </td>
                              {/* Above SMA50 */}
                              <td class="px-3 py-3 text-center">
                                {(() => {
                                  const tech = technicalData()[result.symbol];
                                  const above = tech?.current_price && tech?.sma_50 && tech.current_price > tech.sma_50;
                                  return loadingTechnical() ? (
                                    <span class="text-xs text-gray-500">...</span>
                                  ) : tech ? (
                                    <span class={`text-xs font-bold ${above ? 'text-success-400' : 'text-danger-400'}`}>
                                      {above ? '✓' : '✗'}
                                    </span>
                                  ) : <span class="text-xs text-gray-600">-</span>;
                                })()}
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

      {/* Save Screen Modal */}
      <Show when={showSaveModal()}>
        <div class="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={() => setShowSaveModal(false)}>
          <div class="bg-terminal-900 border border-terminal-750 rounded-lg p-6 w-full max-w-md shadow-2xl" onClick={(e) => e.stopPropagation()}>
            <h3 class="text-lg font-bold text-white mb-4 flex items-center gap-2">
              <Save size={20} class="text-primary-400" />
              Save Screen
            </h3>
            <p class="text-sm text-gray-400 mb-4">
              Save your current filter settings for quick access later.
            </p>
            <input
              type="text"
              value={saveScreenName()}
              onInput={(e) => setSaveScreenName(e.target.value)}
              placeholder="Enter screen name..."
              class="w-full bg-terminal-850 border border-terminal-750 text-white px-4 py-3 rounded-lg focus:outline-none focus:border-primary-500 mb-4"
              onKeyPress={(e) => e.key === 'Enter' && saveCurrentScreen()}
              autofocus
            />
            <div class="flex gap-3">
              <button
                onClick={() => setShowSaveModal(false)}
                class="flex-1 px-4 py-2 bg-terminal-850 hover:bg-terminal-800 border border-terminal-750 text-gray-400 hover:text-white rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={saveCurrentScreen}
                disabled={!saveScreenName().trim()}
                class="flex-1 px-4 py-2 bg-primary-600 hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors"
              >
                Save Screen
              </button>
            </div>
          </div>
        </div>
      </Show>

      {/* Scatter Plot Modal */}
      <Show when={showScatterPlot()}>
        <div class="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={() => setShowScatterPlot(false)}>
          <div class="bg-terminal-900 border border-terminal-750 rounded-lg p-6 w-full max-w-4xl max-h-[90vh] overflow-auto shadow-2xl" onClick={(e) => e.stopPropagation()}>
            <div class="flex items-center justify-between mb-4">
              <h3 class="text-lg font-bold text-white flex items-center gap-2">
                <PieChart size={20} class="text-warning-400" />
                Scatter Plot Analysis
              </h3>
              <button onClick={() => setShowScatterPlot(false)} class="text-gray-400 hover:text-white">
                <X size={20} />
              </button>
            </div>
            
            {/* Axis Selection */}
            <div class="flex gap-4 mb-4">
              <div class="flex-1">
                <label class="text-xs text-gray-400 mb-1 block">X-Axis</label>
                <select
                  value={scatterXAxis()}
                  onChange={(e) => { setScatterXAxis(e.target.value); loadScatterData(); }}
                  class="w-full bg-terminal-850 border border-terminal-750 text-white px-3 py-2 rounded text-sm"
                >
                  <option value="pe_ratio">P/E Ratio</option>
                  <option value="market_cap">Market Cap</option>
                  <option value="price_to_book">P/B Ratio</option>
                  <option value="price_to_sales">P/S Ratio</option>
                  <option value="dividend_yield">Dividend Yield</option>
                  <option value="roe">ROE %</option>
                  <option value="roa">ROA %</option>
                  <option value="profit_margin">Profit Margin</option>
                  <option value="eps">EPS</option>
                </select>
              </div>
              <div class="flex-1">
                <label class="text-xs text-gray-400 mb-1 block">Y-Axis</label>
                <select
                  value={scatterYAxis()}
                  onChange={(e) => { setScatterYAxis(e.target.value); loadScatterData(); }}
                  class="w-full bg-terminal-850 border border-terminal-750 text-white px-3 py-2 rounded text-sm"
                >
                  <option value="roe">ROE %</option>
                  <option value="market_cap">Market Cap</option>
                  <option value="pe_ratio">P/E Ratio</option>
                  <option value="profit_margin">Profit Margin</option>
                  <option value="price_to_book">P/B Ratio</option>
                  <option value="dividend_yield">Dividend Yield</option>
                  <option value="eps">EPS</option>
                  <option value="change_pct">Day Change %</option>
                </select>
              </div>
            </div>

            {/* Scatter Plot Area */}
            <div class="bg-terminal-850 rounded-lg p-4 min-h-[400px]">
              <Show when={loadingScatter()}>
                <div class="flex items-center justify-center h-[400px]">
                  <RefreshCw size={32} class="text-accent-500 animate-spin" />
                </div>
              </Show>
              <Show when={!loadingScatter() && scatterData().length > 0}>
                <div class="relative h-[400px]">
                  {/* Simple scatter visualization - Grid Background */}
                  <div class="absolute inset-0 grid grid-cols-10 grid-rows-10 opacity-20">
                    <For each={Array(100).fill(0)}>
                      {() => <div class="border border-terminal-700" />}
                    </For>
                  </div>
                  
                  {/* Data Points */}
                  <For each={scatterData()}>
                    {(point) => {
                      // Normalize positions (0-100%)
                      const allX = scatterData().map(p => p.x).filter((v): v is number => v !== null);
                      const allY = scatterData().map(p => p.y).filter((v): v is number => v !== null);
                      const minX = Math.min(...allX);
                      const maxX = Math.max(...allX);
                      const minY = Math.min(...allY);
                      const maxY = Math.max(...allY);
                      
                      const x = point.x !== null ? ((point.x - minX) / (maxX - minX || 1)) * 90 + 5 : 50;
                      const y = point.y !== null ? 95 - ((point.y - minY) / (maxY - minY || 1)) * 90 : 50;
                      
                      return (
                        <div
                          class="absolute w-3 h-3 rounded-full bg-accent-500 hover:bg-accent-400 cursor-pointer transform -translate-x-1/2 -translate-y-1/2 transition-all hover:scale-150 z-10"
                          style={{ left: `${x}%`, top: `${y}%` }}
                          title={`${point.symbol}: ${scatterXAxis()}=${point.x?.toFixed(2)}, ${scatterYAxis()}=${point.y?.toFixed(2)}`}
                          onClick={() => navigate(`/charts?symbol=${point.symbol}`)}
                        >
                          <div class="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 opacity-0 hover:opacity-100 bg-terminal-900 px-2 py-1 rounded text-xs whitespace-nowrap">
                            {point.symbol}
                          </div>
                        </div>
                      );
                    }}
                  </For>
                  
                  {/* Axis Labels */}
                  <div class="absolute bottom-0 left-1/2 -translate-x-1/2 text-xs text-gray-400">
                    {scatterXAxis().replace(/_/g, ' ').toUpperCase()}
                  </div>
                  <div class="absolute left-0 top-1/2 -translate-y-1/2 -rotate-90 text-xs text-gray-400">
                    {scatterYAxis().replace(/_/g, ' ').toUpperCase()}
                  </div>
                </div>
                <div class="mt-2 text-xs text-gray-500 text-center">
                  {scatterData().length} data points • Click on a point to view chart
                </div>
              </Show>
              <Show when={!loadingScatter() && scatterData().length === 0}>
                <div class="flex items-center justify-center h-[400px] text-gray-500">
                  No data available for selected axes
                </div>
              </Show>
            </div>
          </div>
        </div>
      </Show>

      {/* Alerts Panel */}
      <Show when={showAlertsPanel()}>
        <div class="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={() => setShowAlertsPanel(false)}>
          <div class="bg-terminal-900 border border-terminal-750 rounded-lg p-6 w-full max-w-lg shadow-2xl" onClick={(e) => e.stopPropagation()}>
            <div class="flex items-center justify-between mb-4">
              <h3 class="text-lg font-bold text-white flex items-center gap-2">
                <Zap size={20} class="text-danger-400" />
                Screener Alerts
              </h3>
              <button onClick={() => setShowAlertsPanel(false)} class="text-gray-400 hover:text-white">
                <X size={20} />
              </button>
            </div>

            {/* Create New Alert */}
            <div class="bg-terminal-850 rounded-lg p-4 mb-4">
              <h4 class="text-sm font-medium text-white mb-3">Create Alert from Current Filters</h4>
              <input
                type="text"
                value={newAlertName()}
                onInput={(e) => setNewAlertName(e.target.value)}
                placeholder="Alert name..."
                class="w-full bg-terminal-900 border border-terminal-750 text-white px-3 py-2 rounded text-sm mb-3"
              />
              <button
                onClick={createAlert}
                disabled={!newAlertName().trim()}
                class="w-full px-4 py-2 bg-danger-600 hover:bg-danger-700 disabled:opacity-50 text-white font-medium rounded transition-colors text-sm"
              >
                Create Alert
              </button>
              <p class="text-xs text-gray-500 mt-2">
                You'll receive notifications when stocks match your current filter criteria.
              </p>
            </div>

            {/* Existing Alerts */}
            <div class="space-y-2 max-h-[300px] overflow-auto">
              <h4 class="text-sm font-medium text-gray-400 mb-2">Your Alerts</h4>
              <Show when={alerts().length === 0}>
                <p class="text-xs text-gray-500 text-center py-4">No alerts created yet</p>
              </Show>
              <For each={alerts()}>
                {(alert) => (
                  <div class="bg-terminal-850 rounded-lg p-3 flex items-center justify-between">
                    <div>
                      <div class="text-sm font-medium text-white">{alert.name}</div>
                      <div class="text-xs text-gray-500">
                        {alert.matched_symbols?.length || 0} matches • Created {new Date(alert.created_at).toLocaleDateString()}
                      </div>
                    </div>
                    <button
                      onClick={() => deleteAlert(alert.id)}
                      class="text-gray-500 hover:text-danger-400 transition-colors"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                )}
              </For>
            </div>
          </div>
        </div>
      </Show>

      {/* Backtest Panel */}
      <Show when={showBacktestPanel()}>
        <div class="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={() => setShowBacktestPanel(false)}>
          <div class="bg-terminal-900 border border-terminal-750 rounded-lg p-6 w-full max-w-lg shadow-2xl" onClick={(e) => e.stopPropagation()}>
            <div class="flex items-center justify-between mb-4">
              <h3 class="text-lg font-bold text-white flex items-center gap-2">
                <BarChart3 size={20} class="text-success-400" />
                Backtest Screener
              </h3>
              <button onClick={() => setShowBacktestPanel(false)} class="text-gray-400 hover:text-white">
                <X size={20} />
              </button>
            </div>

            {/* Create New Backtest */}
            <div class="bg-terminal-850 rounded-lg p-4 mb-4">
              <h4 class="text-sm font-medium text-white mb-3">Run Backtest with Current Filters</h4>
              <input
                type="text"
                value={newBacktestName()}
                onInput={(e) => setNewBacktestName(e.target.value)}
                placeholder="Backtest name..."
                class="w-full bg-terminal-900 border border-terminal-750 text-white px-3 py-2 rounded text-sm mb-3"
              />
              <div class="grid grid-cols-2 gap-3 mb-3">
                <div>
                  <label class="text-xs text-gray-400 mb-1 block">Start Date</label>
                  <input
                    type="date"
                    value={backtestStartDate()}
                    onInput={(e) => setBacktestStartDate(e.target.value)}
                    class="w-full bg-terminal-900 border border-terminal-750 text-white px-3 py-2 rounded text-sm"
                  />
                </div>
                <div>
                  <label class="text-xs text-gray-400 mb-1 block">End Date</label>
                  <input
                    type="date"
                    value={backtestEndDate()}
                    onInput={(e) => setBacktestEndDate(e.target.value)}
                    class="w-full bg-terminal-900 border border-terminal-750 text-white px-3 py-2 rounded text-sm"
                  />
                </div>
              </div>
              <button
                onClick={createBacktest}
                disabled={!newBacktestName().trim() || !backtestStartDate() || !backtestEndDate()}
                class="w-full px-4 py-2 bg-success-600 hover:bg-success-700 disabled:opacity-50 text-white font-medium rounded transition-colors text-sm"
              >
                Run Backtest
              </button>
              <p class="text-xs text-gray-500 mt-2">
                Test how your screening criteria would have performed historically.
              </p>
            </div>

            {/* Existing Backtests */}
            <div class="space-y-2 max-h-[250px] overflow-auto">
              <h4 class="text-sm font-medium text-gray-400 mb-2">Your Backtests</h4>
              <Show when={backtests().length === 0}>
                <p class="text-xs text-gray-500 text-center py-4">No backtests run yet</p>
              </Show>
              <For each={backtests()}>
                {(bt) => (
                  <div class="bg-terminal-850 rounded-lg p-3">
                    <div class="flex items-center justify-between">
                      <div class="text-sm font-medium text-white">{bt.name}</div>
                      <span class={`text-xs px-2 py-0.5 rounded ${
                        bt.status === 'completed' ? 'bg-success-900/30 text-success-400' :
                        bt.status === 'running' ? 'bg-warning-900/30 text-warning-400' :
                        'bg-danger-900/30 text-danger-400'
                      }`}>
                        {bt.status}
                      </span>
                    </div>
                    <Show when={bt.status === 'completed' && bt.total_return !== null}>
                      <div class="mt-2 flex items-center gap-4 text-xs">
                        <span class={`font-bold ${bt.total_return! >= 0 ? 'text-success-400' : 'text-danger-400'}`}>
                          Return: {bt.total_return! >= 0 ? '+' : ''}{bt.total_return?.toFixed(2)}%
                        </span>
                        <Show when={bt.max_drawdown !== null}>
                          <span class="text-danger-400">Max DD: {bt.max_drawdown?.toFixed(2)}%</span>
                        </Show>
                      </div>
                    </Show>
                    <div class="text-xs text-gray-500 mt-1">
                      Created {new Date(bt.created_at).toLocaleDateString()}
                    </div>
                  </div>
                )}
              </For>
            </div>
          </div>
        </div>
      </Show>
    </div>
  );
}
