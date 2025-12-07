/**
 * Price Alert Manager
 * 
 * UI for creating, viewing, and managing price alerts.
 * Integrates with backend API and WebSocket for real-time checking.
 */

import { createSignal, createEffect, For, Show, onMount } from 'solid-js';
import { Bell, BellOff, Plus, Trash2, TrendingUp, TrendingDown } from 'lucide-solid';

export interface PriceAlert {
  id: string;
  symbol: string;
  alert_type: 'above' | 'below' | 'crosses_above' | 'crosses_below';
  price: number;
  message?: string;
  triggered: boolean;
  triggered_at?: string;
  triggered_price?: number;
  enabled: boolean;
  created_at: string;
}

export interface AlertManagerProps {
  symbol: string;
  currentPrice?: number;
  onAlertTriggered?: (alert: PriceAlert) => void;
}

export default function AlertManager(props: AlertManagerProps) {
  const [alerts, setAlerts] = createSignal<PriceAlert[]>([]);
  const [showCreateDialog, setShowCreateDialog] = createSignal(false);
  const [loading, setLoading] = createSignal(false);
  
  // Create dialog fields
  const [alertType, setAlertType] = createSignal<'above' | 'below'>('above');
  const [alertPrice, setAlertPrice] = createSignal('');
  const [alertMessage, setAlertMessage] = createSignal('');

  /**
   * Load alerts from API
   */
  const loadAlerts = async () => {
    setLoading(true);
    try {
      const response = await fetch(`/api/v1/price-alerts?symbol=${props.symbol}&active_only=false`, {
        credentials: 'include',
      });
      
      if (response.ok) {
        const data = await response.json();
        setAlerts(data);
        console.log(`🔔 Loaded ${data.length} alerts for ${props.symbol}`);
      }
    } catch (error) {
      console.error('Failed to load alerts:', error);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Create new alert
   */
  const createAlert = async () => {
    const price = parseFloat(alertPrice());
    if (isNaN(price) || price <= 0) {
      alert('Please enter a valid price');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/v1/price-alerts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          symbol: props.symbol,
          alert_type: alertType(),
          price: price,
          message: alertMessage() || null,
        }),
      });

      if (response.ok) {
        const newAlert = await response.json();
        setAlerts([newAlert, ...alerts()]);
        setShowCreateDialog(false);
        setAlertPrice('');
        setAlertMessage('');
        console.log(`✅ Created alert: ${props.symbol} ${alertType()} $${price}`);
      } else {
        const error = await response.json();
        alert(`Failed to create alert: ${error.detail}`);
      }
    } catch (error) {
      console.error('Failed to create alert:', error);
      alert('Failed to create alert');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Delete an alert
   */
  const deleteAlert = async (id: string) => {
    if (!confirm('Delete this alert?')) return;

    setLoading(true);
    try {
      const response = await fetch(`/api/v1/price-alerts/${id}`, {
        method: 'DELETE',
        credentials: 'include',
      });

      if (response.ok) {
        setAlerts(alerts().filter(a => a.id !== id));
        console.log(`🗑️ Deleted alert: ${id}`);
      }
    } catch (error) {
      console.error('Failed to delete alert:', error);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Toggle alert enabled state
   */
  const toggleAlert = async (id: string, currentState: boolean) => {
    try {
      const response = await fetch(`/api/v1/price-alerts/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ enabled: !currentState }),
      });

      if (response.ok) {
        const updated = await response.json();
        setAlerts(alerts().map(a => a.id === id ? updated : a));
        console.log(`🔔 Toggled alert: ${id} → ${!currentState}`);
      }
    } catch (error) {
      console.error('Failed to toggle alert:', error);
    }
  };

  /**
   * Check alerts against current price
   */
  const checkAlerts = async () => {
    if (!props.currentPrice) return;

    try {
      const response = await fetch(
        `/api/v1/price-alerts/check/${props.symbol}?current_price=${props.currentPrice}`,
        { credentials: 'include' }
      );

      if (response.ok) {
        const result = await response.json();
        if (result.triggered_count > 0) {
          console.log(`🚨 ${result.triggered_count} alerts triggered!`);
          loadAlerts(); // Reload to get updated triggered status
          
          // Notify parent component
          result.triggered_ids.forEach((id: string) => {
            const alert = alerts().find(a => a.id === id);
            if (alert) {
              props.onAlertTriggered?.(alert);
            }
          });
        }
      }
    } catch (error) {
      console.error('Failed to check alerts:', error);
    }
  };

  // Load alerts on mount
  onMount(() => {
    loadAlerts();
  });

  // Reload alerts when symbol changes
  createEffect(() => {
    if (props.symbol) {
      loadAlerts();
    }
  });

  // Check alerts when price changes
  createEffect(() => {
    if (props.currentPrice) {
      checkAlerts();
    }
  });

  const activeAlerts = () => alerts().filter(a => !a.triggered && a.enabled);
  const triggeredAlerts = () => alerts().filter(a => a.triggered);

  return (
    <div class="space-y-3">
      {/* Header */}
      <div class="flex items-center justify-between">
        <h3 class="text-sm font-semibold text-gray-300 flex items-center gap-2">
          <Bell size={16} />
          Price Alerts
        </h3>
        <button
          class="px-2 py-1 bg-primary-600 hover:bg-primary-700 text-white text-xs rounded flex items-center gap-1 transition-colors"
          onClick={() => setShowCreateDialog(true)}
        >
          <Plus size={14} />
          New Alert
        </button>
      </div>

      {/* Active Alerts */}
      <div class="space-y-2">
        <Show when={activeAlerts().length === 0}>
          <div class="text-center py-4 text-xs text-gray-500">
            No active alerts. Create one to get notified!
          </div>
        </Show>

        <For each={activeAlerts()}>
          {(alert) => (
            <div class="p-2 bg-terminal-800 border border-terminal-750 rounded text-xs">
              <div class="flex items-start justify-between gap-2">
                <div class="flex-1">
                  <div class="flex items-center gap-2 text-white font-medium">
                    {alert.alert_type === 'above' ? (
                      <TrendingUp size={14} class="text-green-500" />
                    ) : (
                      <TrendingDown size={14} class="text-red-500" />
                    )}
                    <span>${alert.price.toFixed(2)}</span>
                    <span class="text-gray-500">({alert.alert_type})</span>
                  </div>
                  <Show when={alert.message}>
                    <p class="text-gray-400 mt-1">{alert.message}</p>
                  </Show>
                  <Show when={props.currentPrice}>
                    <div class="mt-1 text-gray-500">
                      Current: ${props.currentPrice?.toFixed(2)} 
                      <span class="ml-2">
                        {alert.alert_type === 'above' 
                          ? `(${((alert.price - (props.currentPrice || 0)) / (props.currentPrice || 1) * 100).toFixed(2)}% away)`
                          : `(${(((props.currentPrice || 0) - alert.price) / (props.currentPrice || 1) * 100).toFixed(2)}% away)`
                        }
                      </span>
                    </div>
                  </Show>
                </div>
                <div class="flex gap-1">
                  <button
                    class="p-1 hover:bg-terminal-750 rounded transition-colors"
                    onClick={() => toggleAlert(alert.id, alert.enabled)}
                    title={alert.enabled ? 'Disable alert' : 'Enable alert'}
                  >
                    {alert.enabled ? (
                      <Bell size={14} class="text-primary-500" />
                    ) : (
                      <BellOff size={14} class="text-gray-600" />
                    )}
                  </button>
                  <button
                    class="p-1 text-red-500 hover:bg-red-500/10 rounded transition-colors"
                    onClick={() => deleteAlert(alert.id)}
                    title="Delete alert"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>
            </div>
          )}
        </For>
      </div>

      {/* Triggered Alerts */}
      <Show when={triggeredAlerts().length > 0}>
        <div class="pt-2 border-t border-terminal-750">
          <h4 class="text-xs font-semibold text-gray-500 mb-2">Triggered</h4>
          <div class="space-y-1">
            <For each={triggeredAlerts()}>
              {(alert) => (
                <div class="p-2 bg-green-500/10 border border-green-500/30 rounded text-xs">
                  <div class="flex items-center justify-between">
                    <div class="text-green-400">
                      ✓ ${alert.price.toFixed(2)} reached at ${alert.triggered_price?.toFixed(2)}
                    </div>
                    <button
                      class="p-1 text-red-400 hover:bg-red-500/10 rounded transition-colors"
                      onClick={() => deleteAlert(alert.id)}
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                </div>
              )}
            </For>
          </div>
        </div>
      </Show>

      {/* Create Dialog */}
      <Show when={showCreateDialog()}>
        <div class="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center" onClick={() => setShowCreateDialog(false)}>
          <div class="bg-terminal-900 border border-terminal-750 rounded-lg p-6 w-full max-w-md" onClick={(e) => e.stopPropagation()}>
            <h3 class="text-lg font-semibold text-white mb-4">Create Price Alert</h3>
            
            <div class="space-y-4">
              <div>
                <label class="block text-sm text-gray-400 mb-1">Symbol</label>
                <input
                  type="text"
                  class="w-full px-3 py-2 bg-terminal-800 border border-terminal-750 rounded text-white"
                  value={props.symbol}
                  disabled
                />
              </div>

              <div>
                <label class="block text-sm text-gray-400 mb-1">Alert Type</label>
                <select
                  class="w-full px-3 py-2 bg-terminal-800 border border-terminal-750 rounded text-white"
                  value={alertType()}
                  onChange={(e) => setAlertType(e.currentTarget.value as 'above' | 'below')}
                >
                  <option value="above">Price Above</option>
                  <option value="below">Price Below</option>
                </select>
              </div>

              <div>
                <label class="block text-sm text-gray-400 mb-1">Price *</label>
                <input
                  type="number"
                  step="0.01"
                  class="w-full px-3 py-2 bg-terminal-800 border border-terminal-750 rounded text-white"
                  placeholder="175.00"
                  value={alertPrice()}
                  onInput={(e) => setAlertPrice(e.currentTarget.value)}
                />
                <Show when={props.currentPrice}>
                  <p class="text-xs text-gray-500 mt-1">
                    Current price: ${props.currentPrice?.toFixed(2)}
                  </p>
                </Show>
              </div>

              <div>
                <label class="block text-sm text-gray-400 mb-1">Message (Optional)</label>
                <textarea
                  class="w-full px-3 py-2 bg-terminal-800 border border-terminal-750 rounded text-white resize-none"
                  rows="2"
                  placeholder="Alert message..."
                  value={alertMessage()}
                  onInput={(e) => setAlertMessage(e.currentTarget.value)}
                />
              </div>

              <div class="flex gap-2 pt-2">
                <button
                  class="flex-1 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded transition-colors"
                  onClick={createAlert}
                  disabled={loading()}
                >
                  {loading() ? 'Creating...' : 'Create Alert'}
                </button>
                <button
                  class="px-4 py-2 bg-terminal-800 hover:bg-terminal-750 text-gray-300 rounded transition-colors"
                  onClick={() => setShowCreateDialog(false)}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      </Show>
    </div>
  );
}
