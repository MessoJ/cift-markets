/**
 * Drawing Toolbar Component
 * 
 * Toolbar for selecting chart drawing tools (trendline, Fibonacci, shapes, etc.)
 */

import { createSignal, For, Show } from 'solid-js';
import { 
  TrendingUp, 
  Minus, 
  Square, 
  Type, 
  ArrowRight,
  Trash2,
  Lock,
  Unlock,
  Eye,
  EyeOff,
  Ruler,
} from 'lucide-solid';
import type { DrawingType } from '~/types/drawing.types';

export interface DrawingToolbarProps {
  activeTool: DrawingType | null;
  selectedDrawingId?: string | null;
  onToolSelect: (tool: DrawingType | null) => void;
  onDeleteSelected?: () => void;
  onClearAll?: () => void;
  drawingCount: number;
}

interface DrawingTool {
  id: DrawingType;
  name: string;
  icon: any;
  shortcut?: string;
  description: string;
}

const DRAWING_TOOLS: DrawingTool[] = [
  {
    id: 'trendline',
    name: 'Trendline',
    icon: TrendingUp,
    shortcut: 'T',
    description: 'Draw trendline between two points',
  },
  {
    id: 'horizontal_line',
    name: 'Horizontal Line',
    icon: Minus,
    shortcut: 'H',
    description: 'Draw horizontal support/resistance line',
  },
  {
    id: 'fibonacci',
    name: 'Fibonacci',
    icon: TrendingUp,
    shortcut: 'F',
    description: 'Fibonacci retracement levels',
  },
  {
    id: 'rectangle',
    name: 'Rectangle',
    icon: Square,
    shortcut: 'R',
    description: 'Draw rectangle area',
  },
  {
    id: 'text',
    name: 'Text',
    icon: Type,
    shortcut: 'A',
    description: 'Add text annotation',
  },
  {
    id: 'arrow',
    name: 'Arrow',
    icon: ArrowRight,
    shortcut: 'W',
    description: 'Draw directional arrow',
  },
  {
    id: 'ruler',
    name: 'Ruler',
    icon: Ruler,
    shortcut: 'M',
    description: 'Measure price and time distance',
  },
];

export default function DrawingToolbar(props: DrawingToolbarProps) {
  const [expanded, setExpanded] = createSignal(false);

  const handleToolClick = (toolId: DrawingType) => {
    if (props.activeTool === toolId) {
      // Deselect tool
      props.onToolSelect(null);
    } else {
      // Select tool
      props.onToolSelect(toolId);
    }
  };

  return (
    <div class="absolute left-4 top-20 z-20">
      <div class="bg-terminal-900 border border-terminal-750 rounded-lg shadow-xl">
        {/* Header */}
        <div class="flex items-center justify-between p-2 border-b border-terminal-750">
          <div class="flex items-center gap-2">
            <TrendingUp size={16} class="text-accent-500" />
            <span class="text-xs font-semibold text-white">Drawing Tools</span>
            <Show when={props.drawingCount > 0}>
              <span class="px-1.5 py-0.5 bg-accent-500/20 text-accent-500 text-xs rounded">
                {props.drawingCount}
              </span>
            </Show>
          </div>
          <button
            onClick={() => setExpanded(!expanded())}
            class="text-gray-400 hover:text-white transition-colors"
            title={expanded() ? 'Collapse' : 'Expand'}
          >
            {expanded() ? '−' : '+'}
          </button>
        </div>

        {/* Tool Buttons */}
        <Show when={expanded()}>
          <div class="p-2 space-y-1">
            <For each={DRAWING_TOOLS}>
              {(tool) => {
                const Icon = tool.icon;
                const isActive = () => props.activeTool === tool.id;
                
                return (
                  <button
                    onClick={() => handleToolClick(tool.id)}
                    class="w-full flex items-center gap-2 px-3 py-2 rounded transition-colors group"
                    classList={{
                      'bg-accent-600 text-white': isActive(),
                      'text-gray-400 hover:bg-terminal-800 hover:text-white': !isActive(),
                    }}
                    title={`${tool.description} (${tool.shortcut})`}
                  >
                    <Icon size={16} />
                    <span class="text-xs font-medium flex-1 text-left">{tool.name}</span>
                    <Show when={tool.shortcut}>
                      <span class="text-xs opacity-60">{tool.shortcut}</span>
                    </Show>
                  </button>
                );
              }}
            </For>

            {/* Divider */}
            <div class="h-px bg-terminal-750 my-2" />

            {/* Actions */}
            <Show when={props.selectedDrawingId}>
              <button
                onClick={() => props.onDeleteSelected?.()}
                class="w-full flex items-center gap-2 px-3 py-2 rounded text-white bg-red-600 hover:bg-red-700 transition-colors"
                title="Delete selected drawing (Delete key)"
              >
                <Trash2 size={16} />
                <span class="text-xs font-medium flex-1 text-left">Delete Selected</span>
                <span class="text-xs opacity-80">Del</span>
              </button>
            </Show>
            
            <button
              onClick={() => props.onClearAll?.()}
              class="w-full flex items-center gap-2 px-3 py-2 rounded text-gray-400 hover:bg-terminal-800 hover:text-red-500 transition-colors"
              title="Clear all drawings"
              disabled={props.drawingCount === 0}
            >
              <Trash2 size={16} />
              <span class="text-xs font-medium flex-1 text-left">Clear All</span>
            </button>
          </div>
        </Show>

        {/* Compact View (when collapsed) */}
        <Show when={!expanded()}>
          <div class="p-2 space-y-1">
            <For each={DRAWING_TOOLS.slice(0, 3)}>
              {(tool) => {
                const Icon = tool.icon;
                const isActive = () => props.activeTool === tool.id;
                
                return (
                  <button
                    onClick={() => handleToolClick(tool.id)}
                    class="w-full p-2 rounded transition-colors flex items-center justify-center"
                    classList={{
                      'bg-accent-600 text-white': isActive(),
                      'text-gray-400 hover:bg-terminal-800 hover:text-white': !isActive(),
                    }}
                    title={tool.name}
                  >
                    <Icon size={16} />
                  </button>
                );
              }}
            </For>
          </div>
        </Show>
      </div>

      {/* Active Tool Indicator */}
      <Show when={props.activeTool}>
        <div class="mt-2 bg-terminal-900 border border-accent-600 rounded-lg px-3 py-2">
          <div class="flex items-center gap-2">
            <div class="w-2 h-2 bg-accent-500 rounded-full animate-pulse" />
            <span class="text-xs text-white font-medium">
              {DRAWING_TOOLS.find(t => t.id === props.activeTool)?.name} mode
            </span>
          </div>
          <p class="text-xs text-gray-400 mt-1">
            Click on chart to start drawing
          </p>
        </div>
      </Show>
    </div>
  );
}
