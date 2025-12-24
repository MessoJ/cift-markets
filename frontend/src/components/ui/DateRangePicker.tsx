/**
 * DateRangePicker Component
 * 
 * Date range selection for analytics and reports.
 * Common presets + custom range selection.
 * 
 * Design System: Professional financial UI
 */

import { createSignal, createEffect, For, Show } from 'solid-js';
import { Calendar, ChevronDown } from 'lucide-solid';

export interface DateRange {
  start: Date;
  end: Date;
  label?: string;
}

interface Preset {
  label: string;
  getValue: () => DateRange;
}

interface DateRangePickerProps {
  value: DateRange;
  onChange: (range: DateRange) => void;
  presets?: Preset[];
  showPresets?: boolean;
  className?: string;
}

// Common date range presets
export const defaultPresets = [
  {
    label: 'Today',
    getValue: () => {
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      const end = new Date();
      return { start: today, end, label: 'Today' };
    },
  },
  {
    label: '1 Week',
    getValue: () => {
      const end = new Date();
      const start = new Date();
      start.setDate(start.getDate() - 7);
      return { start, end, label: '1 Week' };
    },
  },
  {
    label: '1 Month',
    getValue: () => {
      const end = new Date();
      const start = new Date();
      start.setMonth(start.getMonth() - 1);
      return { start, end, label: '1 Month' };
    },
  },
  {
    label: '3 Months',
    getValue: () => {
      const end = new Date();
      const start = new Date();
      start.setMonth(start.getMonth() - 3);
      return { start, end, label: '3 Months' };
    },
  },
  {
    label: '6 Months',
    getValue: () => {
      const end = new Date();
      const start = new Date();
      start.setMonth(start.getMonth() - 6);
      return { start, end, label: '6 Months' };
    },
  },
  {
    label: 'YTD',
    getValue: () => {
      const end = new Date();
      const start = new Date(end.getFullYear(), 0, 1);
      return { start, end, label: 'YTD' };
    },
  },
  {
    label: '1 Year',
    getValue: () => {
      const end = new Date();
      const start = new Date();
      start.setFullYear(start.getFullYear() - 1);
      return { start, end, label: '1 Year' };
    },
  },
  {
    label: 'All Time',
    getValue: () => {
      const end = new Date();
      const start = new Date(2020, 0, 1); // Platform start date
      return { start, end, label: 'All Time' };
    },
  },
];

export function DateRangePicker(props: DateRangePickerProps) {
  const [isOpen, setIsOpen] = createSignal(false);
  const [activeTab, setActiveTab] = createSignal<'presets' | 'custom'>('presets');
  const [customStart, setCustomStart] = createSignal('');
  const [customEnd, setCustomEnd] = createSignal('');
  
  const presets = () => props.presets || defaultPresets;
  
  const formatDate = (date: Date) => {
    if (!date) return '';
    try {
      return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      });
    } catch (e) {
      return '';
    }
  };
  
  const formatDateInput = (date: Date) => {
    if (!date) return '';
    try {
      if (typeof date.toISOString !== 'function') return '';
      return date.toISOString().split('T')[0];
    } catch (e) {
      return '';
    }
  };
  
  createEffect(() => {
    if (props.value) {
      setCustomStart(formatDateInput(props.value.start));
      setCustomEnd(formatDateInput(props.value.end));
    }
  });
  
  const handlePresetSelect = (preset: Preset) => {
    const range = preset.getValue();
    props.onChange(range);
    setIsOpen(false);
  };
  
  const handleCustomApply = () => {
    if (!customStart() || !customEnd()) return;
    
    const range: DateRange = {
      start: new Date(customStart()),
      end: new Date(customEnd()),
      label: 'Custom',
    };
    
    props.onChange(range);
    setIsOpen(false);
  };
  
  const displayLabel = () => {
    if (props.value.label) return props.value.label;
    return `${formatDate(props.value.start)} - ${formatDate(props.value.end)}`;
  };

  return (
    <div class={`relative ${props.className || ''}`}>
      {/* Trigger Button */}
      <button
        class="flex items-center gap-2 px-3 py-2 bg-terminal-850 border border-terminal-750 
               hover:bg-terminal-800 transition-colors text-sm"
        onClick={() => setIsOpen(!isOpen())}
      >
        <Calendar class="w-4 h-4 text-gray-500" />
        <span class="text-white font-mono">{displayLabel()}</span>
        <ChevronDown class={`w-4 h-4 text-gray-500 transition-transform ${isOpen() ? 'rotate-180' : ''}`} />
      </button>
      
      {/* Dropdown */}
      <Show when={isOpen()}>
        <div class="absolute z-50 mt-1 right-0 bg-terminal-900 border border-terminal-750 shadow-xl rounded-lg overflow-hidden min-w-[280px]">
          {/* Tabs */}
          <div class="flex border-b border-terminal-750">
            <button
              class={`flex-1 px-4 py-2 text-xs font-semibold transition-colors ${
                activeTab() === 'presets'
                  ? 'text-accent-400 bg-accent-500/10 border-b-2 border-accent-500'
                  : 'text-gray-500 hover:text-gray-300'
              }`}
              onClick={() => setActiveTab('presets')}
            >
              Quick Select
            </button>
            <button
              class={`flex-1 px-4 py-2 text-xs font-semibold transition-colors ${
                activeTab() === 'custom'
                  ? 'text-accent-400 bg-accent-500/10 border-b-2 border-accent-500'
                  : 'text-gray-500 hover:text-gray-300'
              }`}
              onClick={() => setActiveTab('custom')}
            >
              Custom Range
            </button>
          </div>
          
          {/* Presets */}
          <Show when={activeTab() === 'presets'}>
            <div class="grid grid-cols-2 gap-1 p-2">
              <For each={presets()}>
                {(preset) => (
                  <button
                    class={`px-3 py-2 text-xs font-medium rounded transition-colors text-left
                      ${props.value.label === preset.label
                        ? 'bg-accent-500/20 text-accent-400'
                        : 'text-gray-300 hover:bg-terminal-850'
                      }`}
                    onClick={() => handlePresetSelect(preset)}
                  >
                    {preset.label}
                  </button>
                )}
              </For>
            </div>
          </Show>
          
          {/* Custom Range */}
          <Show when={activeTab() === 'custom'}>
            <div class="p-4 space-y-4">
              <div>
                <label class="text-[10px] text-gray-500 uppercase mb-1 block">Start Date</label>
                <input
                  type="date"
                  value={customStart()}
                  onInput={(e) => setCustomStart(e.currentTarget.value)}
                  class="w-full bg-terminal-850 border border-terminal-750 text-white text-sm px-3 py-2 
                         focus:outline-none focus:border-accent-500"
                />
              </div>
              <div>
                <label class="text-[10px] text-gray-500 uppercase mb-1 block">End Date</label>
                <input
                  type="date"
                  value={customEnd()}
                  onInput={(e) => setCustomEnd(e.currentTarget.value)}
                  class="w-full bg-terminal-850 border border-terminal-750 text-white text-sm px-3 py-2 
                         focus:outline-none focus:border-accent-500"
                />
              </div>
              <button
                class="w-full px-4 py-2 bg-accent-500 hover:bg-accent-600 text-white text-sm font-semibold 
                       rounded transition-colors disabled:opacity-50"
                disabled={!customStart() || !customEnd()}
                onClick={handleCustomApply}
              >
                Apply Range
              </button>
            </div>
          </Show>
        </div>
      </Show>
      
      {/* Click outside to close */}
      <Show when={isOpen()}>
        <div 
          class="fixed inset-0 z-40" 
          onClick={() => setIsOpen(false)}
        />
      </Show>
    </div>
  );
}

/**
 * CompactDateRangePicker - Inline button group
 */
interface CompactDateRangePickerProps {
  value: string;
  options: Array<{ value: string; label: string }>;
  onChange: (value: string) => void;
  className?: string;
}

export function CompactDateRangePicker(props: CompactDateRangePickerProps) {
  return (
    <div class={`flex items-center gap-1 bg-terminal-850 rounded p-1 ${props.className || ''}`}>
      <For each={props.options}>
        {(option) => (
          <button
            class={`px-2 py-1 text-xs font-medium rounded transition-colors
              ${props.value === option.value
                ? 'bg-accent-500 text-white'
                : 'text-gray-400 hover:text-white hover:bg-terminal-800'
              }`}
            onClick={() => props.onChange(option.value)}
          >
            {option.label}
          </button>
        )}
      </For>
    </div>
  );
}

export default DateRangePicker;
