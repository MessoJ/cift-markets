/**
 * Formatting Utilities
 * 
 * Functions for formatting numbers, currency, and dates.
 */

// ============================================================================
// CURRENCY FORMATTING
// ============================================================================

export function formatCurrency(
  value: number,
  options?: {
    minimumFractionDigits?: number;
    maximumFractionDigits?: number;
    showSign?: boolean;
  }
): string {
  const { minimumFractionDigits = 2, maximumFractionDigits = 2, showSign = false } = options || {};

  const formatted = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits,
    maximumFractionDigits,
  }).format(Math.abs(value));

  if (showSign && value > 0) {
    return `+${formatted}`;
  } else if (value < 0) {
    return `-${formatted}`;
  }

  return formatted;
}

// ============================================================================
// PERCENT FORMATTING
// ============================================================================

export function formatPercent(
  value: number,
  options?: {
    minimumFractionDigits?: number;
    maximumFractionDigits?: number;
    showSign?: boolean;
  }
): string {
  const { minimumFractionDigits = 2, maximumFractionDigits = 2, showSign = false } = options || {};

  // Handle invalid values
  if (!Number.isFinite(value)) {
    console.warn('[Fix Applied] formatPercent received non-finite value:', value);
    return '0.00%';
  }

  const formatted = new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits,
    maximumFractionDigits,
  }).format(Math.abs(value) / 100);

  if (showSign && value > 0) {
    return `+${formatted}`;
  } else if (value < 0) {
    return `-${formatted}`;
  }

  return formatted;

}

// ============================================================================
// NUMBER FORMATTING
// ============================================================================

export function formatNumber(
  value: number,
  options?: {
    minimumFractionDigits?: number;
    maximumFractionDigits?: number;
    notation?: 'standard' | 'compact' | 'scientific' | 'engineering';
  }
): string {
  const { minimumFractionDigits = 0, maximumFractionDigits = 2, notation = 'standard' } = options || {};

  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits,
    maximumFractionDigits,
    notation,
  }).format(value);
}

// ============================================================================
// DATE FORMATTING
// ============================================================================

export function formatDate(
  date: string | Date,
  options?: {
    dateStyle?: 'full' | 'long' | 'medium' | 'short';
    timeStyle?: 'full' | 'long' | 'medium' | 'short';
  }
): string {
  const { dateStyle = 'medium', timeStyle } = options || {};

  return new Intl.DateTimeFormat('en-US', {
    dateStyle,
    timeStyle,
  }).format(typeof date === 'string' ? new Date(date) : date);
}

export function formatRelativeTime(date: string | Date): string {
  const now = new Date();
  const target = typeof date === 'string' ? new Date(date) : date;
  const diffMs = now.getTime() - target.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHour = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHour / 24);

  if (diffSec < 60) return 'just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  if (diffHour < 24) return `${diffHour}h ago`;
  if (diffDay < 7) return `${diffDay}d ago`;

  return formatDate(target, { dateStyle: 'short' });
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

export function formatLargeNumber(value: number): string {
  if (value >= 1_000_000_000_000) {
    return `${(value / 1_000_000_000_000).toFixed(2)}T`;
  }
  if (value >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(1)}B`;
  }
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1)}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}K`;
  }
  return value.toFixed(0);
}

export function getPnLColorClass(value: number): string {
  return value >= 0 ? 'text-success-500' : 'text-danger-500';
}

export function getPnLBgClass(value: number): string {
  return value >= 0 ? 'bg-success-900/20 border-success-700/50' : 'bg-danger-900/20 border-danger-700/50';
}
