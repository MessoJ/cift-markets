/**
 * Connection Status Indicator
 * 
 * Shows WebSocket connection status with visual feedback.
 */

import { Show } from 'solid-js';
import { Wifi, WifiOff, RefreshCw } from 'lucide-solid';
import type { ConnectionStatus } from '~/hooks/useMarketDataWebSocket';

export interface ConnectionStatusProps {
  status: ConnectionStatus;
  subscribedSymbols: string[];
  onReconnect?: () => void;
}

export default function ConnectionStatusIndicator(props: ConnectionStatusProps) {
  return (
    <div class="flex items-center gap-2 px-3 py-1.5 rounded text-xs">
      {/* Status Icon */}
      <Show when={props.status === 'connected'}>
        <div class="flex items-center gap-1.5 text-green-500">
          <Wifi size={14} class="animate-pulse" />
          <span class="font-medium">Live</span>
        </div>
      </Show>

      <Show when={props.status === 'connecting'}>
        <div class="flex items-center gap-1.5 text-yellow-500">
          <RefreshCw size={14} class="animate-spin" />
          <span class="font-medium">Connecting...</span>
        </div>
      </Show>

      <Show when={props.status === 'disconnected' || props.status === 'error'}>
        <div class="flex items-center gap-1.5 text-red-500">
          <WifiOff size={14} />
          <span class="font-medium">Offline</span>
          <Show when={props.onReconnect}>
            <button
              onClick={props.onReconnect}
              class="ml-1 px-2 py-0.5 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 rounded transition-colors"
            >
              Reconnect
            </button>
          </Show>
        </div>
      </Show>

      {/* Subscribed Symbols Count */}
      <Show when={props.status === 'connected' && props.subscribedSymbols.length > 0}>
        <div class="ml-2 text-gray-500">
          <span class="text-gray-600">•</span> {props.subscribedSymbols.length} symbol{props.subscribedSymbols.length !== 1 ? 's' : ''}
        </div>
      </Show>
    </div>
  );
}
