/**
 * Professional Trading Table Component
 * 
 * Dense, information-rich table designed for financial data display.
 * Inspired by Bloomberg Terminal and institutional trading platforms.
 * 
 * Features:
 * - High information density
 * - Monospaced numbers for alignment
 * - Compact row spacing
 * - Color-coded values (green/red)
 * - Professional dark theme
 */

import { JSX, For, Show, createSignal } from 'solid-js';
import { twMerge } from 'tailwind-merge';
import { ChevronUp, ChevronDown } from 'lucide-solid';

// ============================================================================
// TYPES
// ============================================================================

export interface Column<T> {
  key: string;
  label: string;
  sortable?: boolean;
  render?: (item: T) => JSX.Element;
  headerClass?: string;
  cellClass?: string;
  align?: 'left' | 'center' | 'right';
  width?: string;
}

interface TableProps<T> {
  data: T[];
  columns: Column<T>[];
  loading?: boolean;
  emptyMessage?: JSX.Element;
  onRowClick?: (item: T) => void;
  striped?: boolean;
  compact?: boolean;
  hoverable?: boolean;
  class?: string;
  rowClass?: (item: T) => string;
}

// ============================================================================
// COMPONENT
// ============================================================================

export function Table<T extends Record<string, any>>(props: TableProps<T>) {
  const [sortKey, setSortKey] = createSignal<string | null>(null);
  const [sortDirection, setSortDirection] = createSignal<'asc' | 'desc'>('asc');

  const handleSort = (column: Column<T>) => {
    if (!column.sortable) return;

    if (sortKey() === column.key) {
      setSortDirection(sortDirection() === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(column.key);
      setSortDirection('asc');
    }
  };

  const sortedData = () => {
    if (!sortKey()) return props.data;

    return [...props.data].sort((a, b) => {
      const aVal = a[sortKey()!];
      const bVal = b[sortKey()!];

      if (aVal === bVal) return 0;

      const comparison = aVal > bVal ? 1 : -1;
      return sortDirection() === 'asc' ? comparison : -comparison;
    });
  };

  const compact = () => props.compact !== false;
  const hoverable = () => props.hoverable !== false;
  const striped = () => props.striped === true;

  return (
    <div class={twMerge('overflow-x-auto border border-terminal-750', props.class)}>
      <table class="w-full border-collapse">
        {/* Header */}
        <thead>
          <tr class="border-b border-terminal-750 bg-terminal-900">
            <For each={props.columns}>
              {(column) => (
                <th
                  class={twMerge(
                    'text-left font-medium text-gray-400 uppercase tracking-wider',
                    compact() ? 'px-3 py-2 text-[11px]' : 'px-4 py-3 text-xs',
                    column.align === 'center' && 'text-center',
                    column.align === 'right' && 'text-right',
                    column.sortable && 'cursor-pointer select-none hover:bg-terminal-850 hover:text-gray-300 transition-colors',
                    column.headerClass
                  )}
                  style={column.width ? { width: column.width } : {}}
                  onClick={() => handleSort(column)}
                >
                  <div class={twMerge(
                    'flex items-center gap-1.5',
                    column.align === 'center' && 'justify-center',
                    column.align === 'right' && 'justify-end'
                  )}>
                    <span>{column.label}</span>
                    <Show when={column.sortable}>
                      <div class="flex flex-col">
                        <ChevronUp
                          class={twMerge(
                            'w-3 h-3 -mb-1',
                            sortKey() === column.key && sortDirection() === 'asc'
                              ? 'text-accent-500'
                              : 'text-gray-600'
                          )}
                        />
                        <ChevronDown
                          class={twMerge(
                            'w-3 h-3',
                            sortKey() === column.key && sortDirection() === 'desc'
                              ? 'text-accent-500'
                              : 'text-gray-600'
                          )}
                        />
                      </div>
                    </Show>
                  </div>
                </th>
              )}
            </For>
          </tr>
        </thead>
        
        {/* Body */}
        <tbody>
          <Show
            when={!props.loading && sortedData().length > 0}
            fallback={
              <tr>
                <td 
                  colspan={props.columns.length} 
                  class={twMerge(
                    'text-center text-gray-500',
                    compact() ? 'py-8' : 'py-12'
                  )}
                >
                  <Show
                    when={!props.loading}
                    fallback={
                      <div class="flex items-center justify-center gap-3">
                        <div class="spinner w-4 h-4 border-2 border-accent-500 border-t-transparent rounded-full animate-spin" />
                        <span class="text-sm">Loading data...</span>
                      </div>
                    }
                  >
                    {typeof props.emptyMessage === 'string' ? (
                      <p class="text-sm">{props.emptyMessage}</p>
                    ) : (
                      props.emptyMessage || <p class="text-sm">No data available</p>
                    )}
                  </Show>
                </td>
              </tr>
            }
          >
            <For each={sortedData()}>
              {(item, index) => (
                <tr
                  class={twMerge(
                    'border-b border-terminal-800',
                    hoverable() && 'hover:bg-terminal-850 transition-colors',
                    striped() && index() % 2 === 1 && 'bg-terminal-900/30',
                    props.onRowClick && 'cursor-pointer',
                    props.rowClass?.(item)
                  )}
                  onClick={() => props.onRowClick?.(item)}
                >
                  <For each={props.columns}>
                    {(column) => (
                      <td 
                        class={twMerge(
                          'text-gray-300 font-mono tabular-nums',
                          compact() ? 'px-3 py-1.5 text-xs' : 'px-4 py-2 text-sm',
                          column.align === 'center' && 'text-center',
                          column.align === 'right' && 'text-right',
                          column.cellClass
                        )}
                      >
                        {column.render ? column.render(item) : item[column.key]}
                      </td>
                    )}
                  </For>
                </tr>
              )}
            </For>
          </Show>
        </tbody>
      </table>
    </div>
  );
}
