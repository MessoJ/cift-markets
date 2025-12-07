/**
 * Professional Order Detail Page
 * 
 * Deep dive into a single order with:
 * - Order details and status
 * - Fill history (partial fills)
 * - Execution timeline
 * - Quick actions (Cancel, Modify, Duplicate)
 * 
 * ALL DATA FROM BACKEND - NO MOCK DATA
 */

import { createSignal, createEffect, Show, For } from 'solid-js';
import { useParams, useNavigate } from '@solidjs/router';
import { apiClient } from '~/lib/api/client';
import { formatCurrency } from '~/lib/utils/format';
import { X, Copy } from 'lucide-solid';

export default function OrderDetailPage() {
  const params = useParams();
  const navigate = useNavigate();
  
  const [order, setOrder] = createSignal<any>(null);
  const [fills, setFills] = createSignal<any[]>([]);
  const [loading, setLoading] = createSignal(true);

  const fetchOrder = async () => {
    try {
      setLoading(true);
      const data = await apiClient.getOrder(params.id);
      setOrder(data);
    } catch (err) {
      console.error('Failed to load order:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchFills = async () => {
    try {
      const data = await apiClient.getOrderFills(params.id);
      setFills(data);
    } catch (err) {
      console.error('Failed to load fills:', err);
    }
  };

  createEffect(() => {
    fetchOrder();
    fetchFills();
  });

  const cancelOrder = async () => {
    if (!confirm('Cancel this order?')) return;
    try {
      await apiClient.cancelOrder(params.id);
      await fetchOrder();
    } catch (err: any) {
      alert(`Failed to cancel order: ${err.message}`);
    }
  };

  const duplicateOrder = () => {
    if (!order()) return;
    navigate('/trading', { 
      state: { 
        symbol: order()!.symbol,
        side: order()!.side,
        quantity: order()!.quantity,
        orderType: order()!.order_type,
        limitPrice: order()!.limit_price
      }
    });
  };

  return (
    <div class="h-full flex flex-col gap-2">
      {/* Top Bar */}
      <div class="bg-terminal-900 border border-terminal-750 p-2 sm:p-3">
        <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 sm:gap-0">
          <div class="flex flex-wrap items-center gap-2 sm:gap-4">
            <button
              onClick={() => navigate('/orders')}
              class="text-gray-500 hover:text-white text-xs font-mono"
            >
              ← BACK
            </button>
            <h2 class="text-base sm:text-lg font-mono font-bold text-white">
              Order #{params.id.slice(0, 8)}
            </h2>
            <Show when={order()}>
              <span class={`text-xs font-mono font-bold px-2 py-1 ${
                order()!.status === 'filled' ? 'bg-success-900/30 text-success-400 border border-success-700' :
                order()!.status === 'open' ? 'bg-accent-900/30 text-accent-400 border border-accent-700' :
                'bg-gray-800 text-gray-500 border border-gray-700'
              }`}>
                {order()!.status.toUpperCase()}
              </span>
            </Show>
          </div>

          {/* Actions */}
          <div class="flex items-center gap-2 w-full sm:w-auto">
            <Show when={order()?.status === 'open'}>
              <button
                onClick={cancelOrder}
                class="px-3 py-1.5 bg-danger-500 hover:bg-danger-600 text-black text-xs font-bold font-mono transition-colors flex items-center gap-1.5"
              >
                <X class="w-3.5 h-3.5" />
                CANCEL
              </button>
            </Show>
            <button
              onClick={duplicateOrder}
              class="px-3 py-1.5 bg-terminal-850 hover:bg-terminal-800 border border-terminal-750 text-gray-400 hover:text-white text-xs font-mono transition-colors flex items-center gap-1.5"
            >
              <Copy class="w-3.5 h-3.5" />
              DUPLICATE
            </button>
          </div>
        </div>
      </div>

      <Show when={!loading() && order()} fallback={
        <div class="flex-1 flex items-center justify-center">
          <span class="text-xs font-mono text-gray-600">Loading order...</span>
        </div>
      }>
        <div class="flex-1 overflow-y-auto p-2 sm:p-3 space-y-2 sm:space-y-3">
          {/* Order Details */}
          <div class="bg-terminal-900 border border-terminal-750 p-3 sm:p-4">
            <h3 class="text-xs font-mono font-bold text-gray-400 uppercase mb-3">Order Details</h3>
            
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4">
              <div>
                <div class="text-[10px] font-mono text-gray-500 uppercase mb-1">Symbol</div>
                <div class="text-lg font-mono font-bold text-white">{order()!.symbol}</div>
              </div>
              <div>
                <div class="text-[10px] font-mono text-gray-500 uppercase mb-1">Side</div>
                <div class={`text-lg font-mono font-bold ${
                  order()!.side === 'buy' ? 'text-success-400' : 'text-danger-400'
                }`}>
                  {order()!.side.toUpperCase()}
                </div>
              </div>
              <div>
                <div class="text-[10px] font-mono text-gray-500 uppercase mb-1">Type</div>
                <div class="text-lg font-mono font-bold text-white">
                  {order()!.order_type.toUpperCase()}
                </div>
              </div>
              <div>
                <div class="text-[10px] font-mono text-gray-500 uppercase mb-1">Time</div>
                <div class="text-sm font-mono text-gray-400">
                  {new Date(order()!.created_at).toLocaleString()}
                </div>
              </div>
            </div>

            <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4 mt-3 sm:mt-4">
              <div>
                <div class="text-[10px] font-mono text-gray-500 uppercase mb-1">Quantity</div>
                <div class="text-2xl font-mono font-bold text-white tabular-nums">
                  {order()!.quantity}
                </div>
              </div>
              <div>
                <div class="text-[10px] font-mono text-gray-500 uppercase mb-1">Filled</div>
                <div class="text-2xl font-mono font-bold text-accent-400 tabular-nums">
                  {order()!.filled_quantity || 0}
                </div>
              </div>
              <div>
                <div class="text-[10px] font-mono text-gray-500 uppercase mb-1">Limit Price</div>
                <div class="text-2xl font-mono font-bold text-white tabular-nums">
                  {order()!.limit_price ? formatCurrency(order()!.limit_price) : 'MARKET'}
                </div>
              </div>
              <div>
                <div class="text-[10px] font-mono text-gray-500 uppercase mb-1">Avg Fill Price</div>
                <div class="text-2xl font-mono font-bold text-white tabular-nums">
                  {order()!.avg_fill_price ? formatCurrency(order()!.avg_fill_price) : '-'}
                </div>
              </div>
            </div>

            <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4 mt-3 sm:mt-4">
              <div>
                <div class="text-[10px] font-mono text-gray-500 uppercase mb-1">Total Value</div>
                <div class="text-2xl font-mono font-bold text-white tabular-nums">
                  {order()!.avg_fill_price 
                    ? formatCurrency(order()!.avg_fill_price * order()!.filled_quantity)
                    : '-'
                  }
                </div>
              </div>
              <div>
                <div class="text-[10px] font-mono text-gray-500 uppercase mb-1">Fees</div>
                <div class="text-2xl font-mono font-bold text-danger-400 tabular-nums">
                  {order()!.fees ? formatCurrency(order()!.fees) : '$0.00'}
                </div>
              </div>
            </div>
          </div>

          {/* Fill History */}
          <div class="bg-terminal-900 border border-terminal-750 p-3 sm:p-4">
            <h3 class="text-xs font-mono font-bold text-gray-400 uppercase mb-3">
              Fill History ({fills().length})
            </h3>
            
            <Show when={fills().length > 0} fallback={
              <div class="text-center py-8 text-xs font-mono text-gray-600">
                No fills yet
              </div>
            }>
              <div class="space-y-2">
                <For each={fills()}>
                  {(fill) => (
                    <div class="bg-terminal-850 border border-terminal-750 p-2 sm:p-3 flex items-center justify-between">
                      <div class="flex-1">
                        <div class="flex flex-wrap items-center gap-2 sm:gap-4 text-xs font-mono">
                          <div>
                            <span class="text-gray-500">TIME:</span>
                            <span class="ml-2 text-white">
                              {new Date(fill.timestamp).toLocaleTimeString()}
                            </span>
                          </div>
                          <div>
                            <span class="text-gray-500">QTY:</span>
                            <span class="ml-2 text-white font-bold">{fill.quantity}</span>
                          </div>
                          <div>
                            <span class="text-gray-500">PRICE:</span>
                            <span class="ml-2 text-white font-bold">
                              {formatCurrency(fill.price)}
                            </span>
                          </div>
                          <div>
                            <span class="text-gray-500">VALUE:</span>
                            <span class="ml-2 text-white font-bold">
                              {formatCurrency(fill.price * fill.quantity)}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </For>
              </div>
            </Show>
          </div>

          {/* Execution Timeline */}
          <div class="bg-terminal-900 border border-terminal-750 p-3 sm:p-4">
            <h3 class="text-xs font-mono font-bold text-gray-400 uppercase mb-3">Timeline</h3>
            
            <div class="space-y-2">
              <div class="flex items-start gap-3">
                <div class="w-2 h-2 bg-success-500 rounded-full mt-2" />
                <div class="flex-1">
                  <div class="text-xs font-mono text-white">Order Created</div>
                  <div class="text-xs font-mono text-gray-500">
                    {new Date(order()!.created_at).toLocaleString()}
                  </div>
                </div>
              </div>
              
              <Show when={order()!.filled_quantity > 0}>
                <div class="flex items-start gap-3">
                  <div class="w-2 h-2 bg-accent-500 rounded-full mt-2" />
                  <div class="flex-1">
                    <div class="text-xs font-mono text-white">
                      Partially Filled ({order()!.filled_quantity} / {order()!.quantity})
                    </div>
                    <div class="text-xs font-mono text-gray-500">
                      {fills().length > 0 && new Date(fills()[0].timestamp).toLocaleString()}
                    </div>
                  </div>
                </div>
              </Show>

              <Show when={order()!.status === 'filled'}>
                <div class="flex items-start gap-3">
                  <div class="w-2 h-2 bg-success-500 rounded-full mt-2" />
                  <div class="flex-1">
                    <div class="text-xs font-mono text-white">Order Filled</div>
                    <div class="text-xs font-mono text-gray-500">
                      {order()!.updated_at && new Date(order()!.updated_at).toLocaleString()}
                    </div>
                  </div>
                </div>
              </Show>

              <Show when={order()!.status === 'cancelled'}>
                <div class="flex items-start gap-3">
                  <div class="w-2 h-2 bg-gray-500 rounded-full mt-2" />
                  <div class="flex-1">
                    <div class="text-xs font-mono text-white">Order Cancelled</div>
                    <div class="text-xs font-mono text-gray-500">
                      {order()!.updated_at && new Date(order()!.updated_at).toLocaleString()}
                    </div>
                  </div>
                </div>
              </Show>
            </div>
          </div>
        </div>
      </Show>
    </div>
  );
}
