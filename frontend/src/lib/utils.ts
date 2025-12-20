/**
 * Utility Functions - Central Export
 * 
 * Re-exports all formatting utilities for easier imports.
 * This file allows importing from both:
 * - import { formatCurrency } from '~/lib/utils'
 * - import { formatCurrency } from '../../lib/utils'
 */

// Re-export all formatting utilities
export {
  formatCurrency,
  formatPercent,
  formatNumber,
  formatDate,
  formatRelativeTime,
  formatLargeNumber,
  getPnLColorClass,
  getPnLBgClass,
} from './utils/format';

// Add alias for formatPercentage (used in some pages)
export { formatPercent as formatPercentage } from './utils/format';
