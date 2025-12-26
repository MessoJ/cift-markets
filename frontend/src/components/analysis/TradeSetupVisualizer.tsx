import { Show } from 'solid-js';
import { formatCurrency } from '~/lib/utils/format';

interface TradeSetupVisualizerProps {
  currentPrice: number;
  entryLow: number;
  entryHigh: number;
  stopLoss: number;
  target1: number;
  target2?: number;
  className?: string;
}

export function TradeSetupVisualizer(props: TradeSetupVisualizerProps) {
  // Calculate range for the visualization
  // Add some padding (5%) to the min/max values
  const values = [
    props.currentPrice,
    props.entryLow,
    props.entryHigh,
    props.stopLoss,
    props.target1,
    props.target2 || props.target1
  ];
  
  const minPrice = Math.min(...values);
  const maxPrice = Math.max(...values);
  const range = maxPrice - minPrice;
  const padding = range * 0.1;
  
  const displayMin = minPrice - padding;
  const displayMax = maxPrice + padding;
  const displayRange = displayMax - displayMin;

  // Helper to convert price to percentage position
  const getPos = (price: number) => {
    return ((price - displayMin) / displayRange) * 100;
  };

  const stopLossPos = getPos(props.stopLoss);
  const entryLowPos = getPos(props.entryLow);
  const entryHighPos = getPos(props.entryHigh);
  const target1Pos = getPos(props.target1);
  const target2Pos = props.target2 ? getPos(props.target2) : null;
  const currentPos = getPos(props.currentPrice);

  return (
    <div class={`w-full ${props.className}`}>
      {/* Price Bar Container */}
      <div class="relative h-12 w-full mt-6 mb-2">
        {/* Base Line */}
        <div class="absolute top-1/2 left-0 right-0 h-1 bg-slate-700 rounded-full transform -translate-y-1/2"></div>

        {/* Stop Loss Zone (Red) */}
        <div 
          class="absolute top-1/2 h-1 bg-red-500/50 transform -translate-y-1/2"
          style={{ 
            left: `${Math.min(stopLossPos, entryLowPos)}%`, 
            width: `${Math.abs(entryLowPos - stopLossPos)}%` 
          }}
        ></div>

        {/* Profit Zone (Green) */}
        <div 
          class="absolute top-1/2 h-1 bg-emerald-500/50 transform -translate-y-1/2"
          style={{ 
            left: `${entryHighPos}%`, 
            width: `${(target2Pos || target1Pos) - entryHighPos}%` 
          }}
        ></div>

        {/* Stop Loss Marker */}
        <div 
          class="absolute top-1/2 w-0.5 h-4 bg-red-500 transform -translate-y-1/2"
          style={{ left: `${stopLossPos}%` }}
        >
          <div class="absolute -top-6 left-1/2 transform -translate-x-1/2 text-[10px] font-bold text-red-400 whitespace-nowrap">
            SL: {formatCurrency(props.stopLoss)}
          </div>
        </div>

        {/* Entry Zone (Blue Box) */}
        <div 
          class="absolute top-1/2 h-3 bg-blue-500/30 border border-blue-500/50 rounded-sm transform -translate-y-1/2"
          style={{ 
            left: `${entryLowPos}%`, 
            width: `${entryHighPos - entryLowPos}%` 
          }}
        >
          <div class="absolute -bottom-6 left-1/2 transform -translate-x-1/2 text-[10px] font-medium text-blue-400 whitespace-nowrap">
            Entry Zone
          </div>
        </div>

        {/* Target 1 Marker */}
        <div 
          class="absolute top-1/2 w-0.5 h-4 bg-emerald-500 transform -translate-y-1/2"
          style={{ left: `${target1Pos}%` }}
        >
          <div class="absolute -top-6 left-1/2 transform -translate-x-1/2 text-[10px] font-bold text-emerald-400 whitespace-nowrap">
            TP1: {formatCurrency(props.target1)}
          </div>
        </div>

        {/* Target 2 Marker */}
        <Show when={props.target2}>
          <div 
            class="absolute top-1/2 w-0.5 h-4 bg-emerald-400 transform -translate-y-1/2"
            style={{ left: `${target2Pos}%` }}
          >
            <div class="absolute -top-6 left-1/2 transform -translate-x-1/2 text-[10px] font-bold text-emerald-300 whitespace-nowrap">
              TP2: {formatCurrency(props.target2!)}
            </div>
          </div>
        </Show>

        {/* Current Price Indicator (Pulsing Dot) */}
        <div 
          class="absolute top-1/2 transform -translate-y-1/2 z-10 transition-all duration-500 ease-out"
          style={{ left: `${currentPos}%` }}
        >
          <div class="relative">
            <div class="w-4 h-4 bg-white rounded-full shadow-lg border-2 border-slate-900 flex items-center justify-center">
              <div class="w-1.5 h-1.5 bg-blue-600 rounded-full"></div>
            </div>
            <div class="absolute -bottom-8 left-1/2 transform -translate-x-1/2 bg-slate-800 text-white text-[10px] px-1.5 py-0.5 rounded border border-slate-700 whitespace-nowrap shadow-xl">
              {formatCurrency(props.currentPrice)}
            </div>
          </div>
        </div>
      </div>
      
      {/* Legend/Status */}
      <div class="flex justify-between items-center mt-8 text-xs text-slate-400 px-1">
        <div class="flex items-center gap-2">
          <span class="w-2 h-2 rounded-full bg-red-500"></span>
          <span>Risk: {((props.currentPrice - props.stopLoss) / props.currentPrice * 100).toFixed(2)}%</span>
        </div>
        <div class="flex items-center gap-2">
          <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
          <span>Reward: {((props.target1 - props.currentPrice) / props.currentPrice * 100).toFixed(2)}%</span>
        </div>
      </div>
    </div>
  );
}
