/**
 * UI Components Barrel Export
 * 
 * Central export for all reusable UI components.
 * Import like: import { Button, Card, Sparkline } from '@/components/ui';
 */

// Core UI Components
export { Button } from './Button';
export { Card } from './Card';
export { Input } from './Input';
export { Modal } from './Modal';
export { Table } from './Table';

// Data Visualization Components
export { Sparkline, SparklineWithValue } from './Sparkline';
export { DonutChart, MiniDonutChart } from './DonutChart';
export { Treemap } from './Treemap';
export { EquityCurve, MiniEquityCurve } from './EquityCurve';
export { HeatmapGrid, SectorHeatmap, CorrelationMatrix } from './HeatmapGrid';

// Trading Components
export { OrderBook, CompactOrderBook } from './OrderBook';
export { TimeSales, CompactTimeSales } from './TimeSales';
export { MarketTicker, StaticMarketBar, MarketMovers } from './MarketTicker';
export { SymbolSearch } from './SymbolSearch';

// Utility Components
export { 
  DateRangePicker, 
  CompactDateRangePicker,
  defaultPresets,
  type DateRange 
} from './DateRangePicker';

export {
  EmptyState,
  NoTradesState,
  NoPositionsState,
  NoAlertsState,
  NoSearchResults,
  NoWatchlistState,
  ErrorState,
  LoadingState,
  InlineEmptyState,
  emptyStateIcons
} from './EmptyState';

export {
  KeyboardShortcutsProvider,
  registerShortcut,
  registerShortcuts,
  ShortcutHint,
  isHelpOpen,
  setIsHelpOpen,
  isCommandPaletteOpen,
  setIsCommandPaletteOpen,
  type Shortcut
} from './KeyboardShortcuts';

// Re-export types
export type { DonutSegment } from './DonutChart';
export type { TreemapItem } from './Treemap';
export type { HeatmapCell } from './HeatmapGrid';
export type { SearchResult } from './SymbolSearch';
