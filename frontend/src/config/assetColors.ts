/**
 * Asset Category Colors & Event States
 * 
 * Bloomberg-inspired color scheme for different asset classes
 * and event-driven status indicators for real-time market visualization
 */

export enum AssetCategory {
  EQUITY = 'equity',
  COMMODITY = 'commodity',
  FOREX = 'forex',
  CRYPTO = 'crypto',
  BOND = 'bond',
  ETF = 'etf',
  INDEX = 'index',
  OPTION = 'option',
  FUTURE = 'future',
  EXCHANGE = 'exchange',
  CAPITAL = 'capital',
  UNKNOWN = 'unknown',
}

export enum EventStatus {
  NORMAL = 'normal',
  WARNING = 'warning',      // Minor news/volatility
  CRITICAL = 'critical',    // Major market-moving event
  HALTED = 'halted',        // Trading halted
  POSITIVE = 'positive',    // Significant gains
  NEGATIVE = 'negative',    // Significant losses
}

export const ASSET_COLORS: Record<AssetCategory, string> = {
  [AssetCategory.EQUITY]: '#3B82F6',      // Blue - stocks
  [AssetCategory.COMMODITY]: '#F59E0B',   // Orange - commodities
  [AssetCategory.FOREX]: '#10B981',       // Green - currencies
  [AssetCategory.CRYPTO]: '#8B5CF6',      // Purple - crypto
  [AssetCategory.BOND]: '#6366F1',        // Indigo - bonds
  [AssetCategory.ETF]: '#EC4899',         // Pink - ETFs
  [AssetCategory.INDEX]: '#14B8A6',       // Teal - indices
  [AssetCategory.OPTION]: '#F97316',      // Deep orange - options
  [AssetCategory.FUTURE]: '#EAB308',      // Yellow - futures
  [AssetCategory.EXCHANGE]: '#06B6D4',    // Cyan - exchanges
  [AssetCategory.CAPITAL]: '#A855F7',     // Purple variant - capitals
  [AssetCategory.UNKNOWN]: '#6B7280',     // Gray - unknown
};

export const EVENT_STATUS_COLORS: Record<EventStatus, string> = {
  [EventStatus.NORMAL]: '#10B981',        // Green - normal
  [EventStatus.WARNING]: '#F59E0B',       // Orange - caution
  [EventStatus.CRITICAL]: '#EF4444',      // Red - critical
  [EventStatus.HALTED]: '#DC2626',        // Dark red - halted
  [EventStatus.POSITIVE]: '#22C55E',      // Bright green - gains
  [EventStatus.NEGATIVE]: '#F43F5E',      // Rose - losses
};

export interface AssetMarkerData {
  id: string;
  name: string;
  symbol?: string;
  category: AssetCategory;
  lat: number;
  lng: number;
  value?: number;           // Market cap, volume, or asset value
  change?: number;          // Price/value change
  change_pct?: number;      // Percentage change
  eventStatus: EventStatus;
  news?: NewsEvent[];
  lastUpdate: Date;
}

export interface NewsEvent {
  id: string;
  headline: string;
  summary: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  importance: 'low' | 'medium' | 'high' | 'critical';
  timestamp: Date;
  source: string;
  url?: string;
  relatedAssets: string[];
  impactScore?: number;     // 0-100 severity/impact
}

export interface MarketMovingEvent {
  assetId: string;
  eventType: 'earnings' | 'merger' | 'regulatory' | 'geopolitical' | 'economic' | 'disaster' | 'other';
  severity: EventStatus;
  description: string;
  triggeredAt: Date;
  relatedNews: NewsEvent[];
}

/**
 * Determine event status based on price movement and news sentiment
 */
export function calculateEventStatus(
  change_pct?: number,
  news?: NewsEvent[],
  isHalted?: boolean
): EventStatus {
  if (isHalted) return EventStatus.HALTED;
  
  if (!change_pct && (!news || news.length === 0)) {
    return EventStatus.NORMAL;
  }

  // Check for critical news first
  const hasCriticalNews = news?.some(n => n.importance === 'critical' && Math.abs(n.impactScore || 0) > 75);
  if (hasCriticalNews) return EventStatus.CRITICAL;

  // Check price movement
  if (change_pct !== undefined) {
    if (change_pct > 10) return EventStatus.POSITIVE;
    if (change_pct < -10) return EventStatus.NEGATIVE;
    if (Math.abs(change_pct) > 5) return EventStatus.WARNING;
  }

  // Check high-impact news
  const hasHighImpactNews = news?.some(n => n.importance === 'high' && Math.abs(n.impactScore || 0) > 50);
  if (hasHighImpactNews) return EventStatus.WARNING;

  return EventStatus.NORMAL;
}

/**
 * Get marker size multiplier based on asset value
 * Scales from 0.5x to 3x base size
 */
export function getMarkerSizeMultiplier(value?: number): number {
  if (!value) return 1.0;
  
  // Logarithmic scale for better distribution
  const logValue = Math.log10(value);
  
  // Typical ranges:
  // $1M = 6, $100M = 8, $10B = 10, $1T = 12
  if (logValue < 6) return 0.5;      // < $1M
  if (logValue < 8) return 0.8;      // $1M - $100M
  if (logValue < 10) return 1.2;     // $100M - $10B
  if (logValue < 11) return 1.8;     // $10B - $100B
  if (logValue < 12) return 2.5;     // $100B - $1T
  return 3.0;                         // > $1T
}

/**
 * Format large numbers for display (1.2B, 45.3M, etc.)
 */
export function formatLargeNumber(num: number): string {
  if (num >= 1e12) return `$${(num / 1e12).toFixed(1)}T`;
  if (num >= 1e9) return `$${(num / 1e9).toFixed(1)}B`;
  if (num >= 1e6) return `$${(num / 1e6).toFixed(1)}M`;
  if (num >= 1e3) return `$${(num / 1e3).toFixed(1)}K`;
  return `$${num.toFixed(0)}`;
}

/**
 * Get pulsing animation speed based on event status
 */
export function getPulseSpeed(status: EventStatus): number {
  switch (status) {
    case EventStatus.CRITICAL:
    case EventStatus.HALTED:
      return 0.5; // Fast pulse
    case EventStatus.WARNING:
    case EventStatus.POSITIVE:
    case EventStatus.NEGATIVE:
      return 1.0; // Medium pulse
    default:
      return 0; // No pulse
  }
}
