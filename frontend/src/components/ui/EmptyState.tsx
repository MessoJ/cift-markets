/**
 * EmptyState Component
 * 
 * Professional empty/zero states for lists, tables, and dashboards.
 * Provides clear guidance and actions instead of blank space.
 * 
 * Design System: Helpful and action-oriented
 */

import { JSX, Show, For } from 'solid-js';
import { 
  BarChart2, 
  TrendingUp, 
  AlertTriangle,
  Search,
  Plus,
  Bell,
  Briefcase,
  Eye,
  Activity,
  FileText,
  Zap,
  RefreshCw
} from 'lucide-solid';

interface EmptyStateAction {
  label: string;
  onClick: () => void;
  variant?: 'primary' | 'secondary';
  icon?: JSX.Element;
}

interface EmptyStateProps {
  icon?: JSX.Element;
  title: string;
  description: string;
  actions?: EmptyStateAction[];
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

// Pre-built icons for common scenarios
export const emptyStateIcons = {
  noData: <BarChart2 class="w-8 h-8" />,
  noResults: <Search class="w-8 h-8" />,
  noTrades: <TrendingUp class="w-8 h-8" />,
  noAlerts: <Bell class="w-8 h-8" />,
  noPositions: <Briefcase class="w-8 h-8" />,
  noWatchlist: <Eye class="w-8 h-8" />,
  noActivity: <Activity class="w-8 h-8" />,
  noReports: <FileText class="w-8 h-8" />,
  error: <AlertTriangle class="w-8 h-8" />,
  loading: <RefreshCw class="w-8 h-8 animate-spin" />,
};

export function EmptyState(props: EmptyStateProps) {
  const size = () => props.size || 'md';
  
  const sizeClasses = {
    sm: { icon: 'w-10 h-10', title: 'text-sm', desc: 'text-xs', padding: 'py-6' },
    md: { icon: 'w-16 h-16', title: 'text-base', desc: 'text-sm', padding: 'py-10' },
    lg: { icon: 'w-20 h-20', title: 'text-lg', desc: 'text-base', padding: 'py-16' },
  };

  return (
    <div class={`flex flex-col items-center justify-center text-center ${sizeClasses[size()].padding} ${props.className || ''}`}>
      {/* Icon */}
      <div class={`${sizeClasses[size()].icon} flex items-center justify-center text-gray-600 mb-4`}>
        {props.icon || emptyStateIcons.noData}
      </div>
      
      {/* Title */}
      <h3 class={`${sizeClasses[size()].title} font-semibold text-white mb-2`}>
        {props.title}
      </h3>
      
      {/* Description */}
      <p class={`${sizeClasses[size()].desc} text-gray-500 max-w-sm mb-6`}>
        {props.description}
      </p>
      
      {/* Actions */}
      <Show when={props.actions && props.actions.length > 0}>
        <div class="flex items-center gap-3">
          <For each={props.actions}>
            {(action) => (
              <button
                class={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded transition-colors
                  ${action.variant === 'secondary'
                    ? 'bg-terminal-800 text-gray-300 hover:bg-terminal-700'
                    : 'bg-accent-500 text-white hover:bg-accent-600'
                  }`}
                onClick={action.onClick}
              >
                {action.icon}
                {action.label}
              </button>
            )}
          </For>
        </div>
      </Show>
    </div>
  );
}

/**
 * Common pre-configured empty states
 */

export function NoTradesState(props: { onCreateTrade?: () => void }) {
  return (
    <EmptyState
      icon={emptyStateIcons.noTrades}
      title="No trades yet"
      description="Start trading to see your performance history, P&L breakdown, and analytics."
      actions={props.onCreateTrade ? [
        { label: 'Place First Trade', onClick: props.onCreateTrade, icon: <Plus class="w-4 h-4" /> }
      ] : undefined}
    />
  );
}

export function NoPositionsState(props: { onTrade?: () => void }) {
  return (
    <EmptyState
      icon={emptyStateIcons.noPositions}
      title="No open positions"
      description="Your portfolio is currently empty. Open a position to start building your portfolio."
      actions={props.onTrade ? [
        { label: 'Start Trading', onClick: props.onTrade, icon: <TrendingUp class="w-4 h-4" /> }
      ] : undefined}
    />
  );
}

export function NoAlertsState(props: { onCreateAlert?: () => void }) {
  return (
    <EmptyState
      icon={emptyStateIcons.noAlerts}
      title="No alerts configured"
      description="Set up price alerts to get notified when markets move. Never miss an opportunity."
      actions={props.onCreateAlert ? [
        { label: 'Create Alert', onClick: props.onCreateAlert, icon: <Zap class="w-4 h-4" /> }
      ] : undefined}
    />
  );
}

export function NoSearchResults(props: { query?: string; onClear?: () => void }) {
  return (
    <EmptyState
      icon={emptyStateIcons.noResults}
      title="No results found"
      description={props.query 
        ? `No matches for "${props.query}". Try adjusting your search or filters.`
        : 'Try adjusting your search or filters.'}
      actions={props.onClear ? [
        { label: 'Clear Search', onClick: props.onClear, variant: 'secondary' }
      ] : undefined}
      size="sm"
    />
  );
}

export function NoWatchlistState(props: { onAddSymbol?: () => void }) {
  return (
    <EmptyState
      icon={emptyStateIcons.noWatchlist}
      title="Watchlist is empty"
      description="Add symbols to track their prices, changes, and volume in real-time."
      actions={props.onAddSymbol ? [
        { label: 'Add Symbol', onClick: props.onAddSymbol, icon: <Plus class="w-4 h-4" /> }
      ] : undefined}
    />
  );
}

export function ErrorState(props: { 
  message?: string; 
  onRetry?: () => void;
}) {
  return (
    <EmptyState
      icon={emptyStateIcons.error}
      title="Something went wrong"
      description={props.message || 'An unexpected error occurred. Please try again.'}
      actions={props.onRetry ? [
        { label: 'Try Again', onClick: props.onRetry, icon: <RefreshCw class="w-4 h-4" /> }
      ] : undefined}
    />
  );
}

export function LoadingState(props: { message?: string }) {
  return (
    <div class="flex flex-col items-center justify-center py-10">
      <RefreshCw class="w-8 h-8 text-accent-500 animate-spin mb-4" />
      <p class="text-sm text-gray-500">{props.message || 'Loading...'}</p>
    </div>
  );
}

/**
 * InlineEmptyState - For smaller inline contexts like table rows
 */
export function InlineEmptyState(props: { message: string; icon?: JSX.Element }) {
  return (
    <div class="flex items-center gap-2 text-gray-500 text-sm py-4 px-3">
      {props.icon || <Search class="w-4 h-4" />}
      <span>{props.message}</span>
    </div>
  );
}

export default EmptyState;
