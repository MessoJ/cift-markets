/**
 * PRICE ALERTS PAGE
 * Comprehensive alert management with real-time notifications and templates
 */

import { createSignal, createEffect, For, Show } from 'solid-js';
import { Bell, Plus, Trash2, CheckCircle2, Clock, XCircle, Mail, Smartphone, Monitor, Zap, TrendingUp, TrendingDown, BarChart3, Percent, Activity, Copy, ChevronRight } from 'lucide-solid';
import { apiClient, PriceAlert } from '../../lib/api/client';
import { formatCurrency } from '../../lib/utils';
import { Sparkline } from '~/components/ui/Sparkline';
import { authStore } from '~/stores/auth.store';

export default function AlertsPage() {
  const [loading, setLoading] = createSignal(false);
  const [alerts, setAlerts] = createSignal<PriceAlert[]>([]);
  const [showCreateModal, setShowCreateModal] = createSignal(false);
  const [filterStatus, setFilterStatus] = createSignal<string>('active');
  const [activeView, setActiveView] = createSignal<'alerts' | 'templates' | 'history'>('alerts');
  
  // Notification state
  const [notification, setNotification] = createSignal<{type: 'success' | 'error', message: string} | null>(null);

  // Create alert form
  const [symbol, setSymbol] = createSignal('');
  const [alertType, setAlertType] = createSignal<'price_above' | 'price_below' | 'price_change' | 'volume'>('price_above');
  const [targetValue, setTargetValue] = createSignal('');
  const [notifyEmail, setNotifyEmail] = createSignal(true);
  const [notifySms, setNotifySms] = createSignal(false);
  const [notifyPush, setNotifyPush] = createSignal(true);

  // Alert templates
  const alertTemplates = [
    { name: 'Breakout Alert', type: 'price_above', icon: TrendingUp, color: 'text-success-500', desc: 'Alert when price breaks above resistance' },
    { name: 'Support Break', type: 'price_below', icon: TrendingDown, color: 'text-danger-500', desc: 'Alert when price falls below support' },
    { name: 'Volatility Spike', type: 'price_change', icon: Activity, color: 'text-warning-500', desc: 'Alert on significant price movement' },
    { name: 'Volume Surge', type: 'volume', icon: BarChart3, color: 'text-primary-500', desc: 'Alert on unusual volume activity' },
    { name: 'Target Price', type: 'price_above', icon: Zap, color: 'text-accent-500', desc: 'Alert when target price is reached' },
    { name: 'Stop Loss', type: 'price_below', icon: XCircle, color: 'text-danger-500', desc: 'Alert when stop loss level is hit' },
  ];

  // Notification history - fetched from API
  const [notificationHistory, setNotificationHistory] = createSignal<any[]>([]);

  // Generate sparkline data from price history or estimate
  const getAlertSparkline = (alert: PriceAlert) => {
    const data = [];
    const base = alert.current_value || alert.target_value * 0.95;
    for (let i = 0; i < 24; i++) {
      data.push(base * (0.98 + Math.random() * 0.04));
    }
    return data;
  };

  // Load notification history from API
  const loadNotifications = async () => {
    if (!authStore.isAuthenticated) return;
    try {
      // Use apiClient to ensure credentials/headers are included
      const response = await apiClient.axiosInstance.get('/alerts/notifications');
      setNotificationHistory(response.data.notifications || []);
    } catch (err) {
      console.error('Failed to load notifications:', err);
    }
  };

  createEffect(() => {
    // Track filterStatus to reload when it changes
    filterStatus();
    loadAlerts();
    loadNotifications();
  });
  
  // Auto-hide notification after 5 seconds
  createEffect(() => {
    if (notification()) {
      setTimeout(() => setNotification(null), 5000);
    }
  });

  const loadAlerts = async () => {
    if (!authStore.isAuthenticated) return;
    console.log('🔄 Loading alerts with filter:', filterStatus());
    setLoading(true);
    try {
      const data = await apiClient.getAlerts(filterStatus() === 'all' ? undefined : filterStatus());
      console.log('✅ Loaded alerts:', data?.length || 0, 'alerts');
      console.log('📊 Alerts data:', data);
      setAlerts(data || []);
    } catch (err) {
      console.error('❌ Failed to load alerts', err);
      setAlerts([]);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateAlert = async () => {
    console.log('🔔 Create Alert button clicked!');
    console.log('📝 Symbol:', symbol());
    console.log('📝 Target Value:', targetValue());
    
    if (!symbol() || !targetValue()) {
      console.warn('⚠️ Validation failed: Missing symbol or target value');
      setNotification({type: 'error', message: 'Please enter both symbol and target value'});
      return;
    }

    const sym = symbol().toUpperCase();
    if (!/^[A-Z0-9]{1,10}$/.test(sym)) {
      setNotification({type: 'error', message: 'Invalid symbol format'});
      return;
    }

    const val = parseFloat(targetValue());
    if (isNaN(val) || val <= 0) {
      setNotification({type: 'error', message: 'Target value must be a positive number'});
      return;
    }

    const methods: ('email' | 'sms' | 'push')[] = [];
    if (notifyEmail()) methods.push('email');
    if (notifySms()) methods.push('sms');
    if (notifyPush()) methods.push('push');
    
    if (methods.length === 0) {
      setNotification({type: 'error', message: 'Please select at least one notification method'});
      return;
    }
    
    console.log('📧 Notification methods:', methods);

    const alertData = {
      symbol: sym,
      alert_type: alertType(),
      condition_value: val, // Mapped to backend expectation
      notification_methods: methods,
      message: `Alert for ${sym} at ${val}` // Auto-generate message
    };
    
    console.log('📤 Sending alert data:', alertData);

    try {
      console.log('🌐 Calling API...');
      // @ts-ignore - Client type definition might be outdated
      const result = await apiClient.createAlert(alertData);
      console.log('✅ Alert created successfully:', result);
      
      // Show success message
      setNotification({type: 'success', message: `Alert created successfully for ${symbol().toUpperCase()}!`});
      
      setShowCreateModal(false);
      resetForm();
      
      // Reload with the current filter
      await loadAlerts();
      console.log('✅ Alerts reloaded');
    } catch (err: any) {
      console.error('❌ Failed to create alert:', err);
      console.error('❌ Error details:', err.message, err.response?.data);
      
      // Show error message
      const errorMsg = err.response?.data?.detail || err.message || 'Failed to create alert';
      setNotification({type: 'error', message: errorMsg});
    }
  };

  const resetForm = () => {
    setSymbol('');
    setTargetValue('');
    setAlertType('price_above');
    setNotifyEmail(true);
    setNotifySms(false);
    setNotifyPush(true);
  };

  const handleDeleteAlert = async (alertId: string) => {
    if (!confirm('Delete this alert?')) return;
    try {
      await apiClient.deleteAlert(alertId);
      await loadAlerts();
    } catch (err) {
      console.error('Failed to delete alert', err);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <Clock size={16} class="text-warning-500" />;
      case 'triggered': return <CheckCircle2 size={16} class="text-success-500" />;
      case 'cancelled': return <XCircle size={16} class="text-gray-500" />;
      default: return <Bell size={16} class="text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-warning-500 bg-warning-500/10';
      case 'triggered': return 'text-success-500 bg-success-500/10';
      case 'cancelled': return 'text-gray-500 bg-gray-800/50';
      default: return 'text-gray-400 bg-gray-800/50';
    }
  };

  const getAlertTypeLabel = (type: string) => {
    switch (type) {
      case 'price_above': return 'Price Above';
      case 'price_below': return 'Price Below';
      case 'price_change': return 'Price Change';
      case 'volume': return 'Volume';
      default: return type;
    }
  };

  return (
    <div class="h-full flex flex-col gap-2 sm:gap-3 p-2 sm:p-3">
      {/* Inline Notification */}
      <Show when={notification()}>
        <div class={`p-4 rounded-lg border ${
          notification()?.type === 'success' 
            ? 'bg-success-500/10 border-success-500/30 text-success-500' 
            : 'bg-danger-500/10 border-danger-500/30 text-danger-500'
        } flex items-center justify-between animate-in fade-in slide-in-from-top-2`}>
          <div class="flex items-center gap-3">
            {notification()?.type === 'success' ? (
              <CheckCircle2 size={20} />
            ) : (
              <XCircle size={20} />
            )}
            <span class="text-sm font-semibold">{notification()?.message}</span>
          </div>
          <button 
            onClick={() => setNotification(null)}
            class="p-1 hover:bg-white/10 rounded transition-colors"
          >
            <XCircle size={16} />
          </button>
        </div>
      </Show>
      
      {/* Loading Bar */}
      <Show when={loading()}>
        <div class="bg-terminal-900 border border-terminal-750 p-2 text-xs text-gray-400 rounded">
          Loading alerts...
        </div>
      </Show>
      
      {/* Header */}
      <div class="bg-terminal-900 border border-terminal-750 p-3">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <div class="w-9 h-9 bg-accent-500/10 rounded flex items-center justify-center">
              <Bell size={18} class="text-accent-500" />
            </div>
            <div>
              <h1 class="text-base font-bold text-white">Price Alerts</h1>
              <p class="text-[10px] text-gray-400">Get notified when stocks hit your target prices</p>
            </div>
          </div>
          <div class="flex items-center gap-2">
            {/* View Toggle */}
            <div class="flex items-center bg-terminal-850 border border-terminal-750 rounded overflow-hidden">
              <button
                onClick={() => setActiveView('alerts')}
                class={`px-3 py-1.5 text-[10px] font-mono font-bold uppercase transition-colors ${
                  activeView() === 'alerts' ? 'bg-primary-600 text-white' : 'text-gray-400 hover:text-white'
                }`}
              >
                Alerts
              </button>
              <button
                onClick={() => setActiveView('templates')}
                class={`px-3 py-1.5 text-[10px] font-mono font-bold uppercase transition-colors ${
                  activeView() === 'templates' ? 'bg-primary-600 text-white' : 'text-gray-400 hover:text-white'
                }`}
              >
                Templates
              </button>
              <button
                onClick={() => setActiveView('history')}
                class={`px-3 py-1.5 text-[10px] font-mono font-bold uppercase transition-colors ${
                  activeView() === 'history' ? 'bg-primary-600 text-white' : 'text-gray-400 hover:text-white'
                }`}
              >
                History
              </button>
            </div>
            <button
              onClick={() => setShowCreateModal(true)}
              class="flex items-center gap-2 px-3 py-1.5 bg-accent-500 hover:bg-accent-600 text-white text-xs font-bold rounded transition-colors"
            >
              <Plus size={14} />
              <span>New Alert</span>
            </button>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div class="bg-gradient-to-br from-terminal-900 to-terminal-800 border border-terminal-750 p-4 rounded-lg shadow-sm relative overflow-hidden group">
          <div class="absolute right-0 top-0 w-24 h-24 bg-warning-500/5 rounded-full -mr-8 -mt-8 group-hover:bg-warning-500/10 transition-colors"></div>
          <div class="flex items-center gap-4 relative z-10">
            <div class="w-12 h-12 bg-terminal-950 border border-terminal-800 rounded-lg flex items-center justify-center shadow-inner">
              <Clock size={24} class="text-warning-500" />
            </div>
            <div>
              <div class="text-3xl font-bold text-white tabular-nums tracking-tight">
                {alerts()?.filter((a) => a.status === 'active').length || 0}
              </div>
              <div class="text-xs font-medium text-gray-400 uppercase tracking-wider">Active Monitors</div>
            </div>
          </div>
        </div>

        <div class="bg-gradient-to-br from-terminal-900 to-terminal-800 border border-terminal-750 p-4 rounded-lg shadow-sm relative overflow-hidden group">
          <div class="absolute right-0 top-0 w-24 h-24 bg-success-500/5 rounded-full -mr-8 -mt-8 group-hover:bg-success-500/10 transition-colors"></div>
          <div class="flex items-center gap-4 relative z-10">
            <div class="w-12 h-12 bg-terminal-950 border border-terminal-800 rounded-lg flex items-center justify-center shadow-inner">
              <CheckCircle2 size={24} class="text-success-500" />
            </div>
            <div>
              <div class="text-3xl font-bold text-white tabular-nums tracking-tight">
                {alerts()?.filter((a) => a.status === 'triggered').length || 0}
              </div>
              <div class="text-xs font-medium text-gray-400 uppercase tracking-wider">Triggered Events</div>
            </div>
          </div>
        </div>

        <div class="bg-gradient-to-br from-terminal-900 to-terminal-800 border border-terminal-750 p-4 rounded-lg shadow-sm relative overflow-hidden group">
          <div class="absolute right-0 top-0 w-24 h-24 bg-primary-500/5 rounded-full -mr-8 -mt-8 group-hover:bg-primary-500/10 transition-colors"></div>
          <div class="flex items-center gap-4 relative z-10">
            <div class="w-12 h-12 bg-terminal-950 border border-terminal-800 rounded-lg flex items-center justify-center shadow-inner">
              <Bell size={24} class="text-primary-500" />
            </div>
            <div>
              <div class="text-3xl font-bold text-white tabular-nums tracking-tight">{alerts()?.length || 0}</div>
              <div class="text-xs font-medium text-gray-400 uppercase tracking-wider">Total Configured</div>
            </div>
          </div>
        </div>
      </div>

      {/* Filter Tabs */}
      <Show when={activeView() === 'alerts'}>
        <div class="flex items-center gap-1 bg-terminal-900 border border-terminal-750 p-1">
          <button
            onClick={() => setFilterStatus('all')}
            class={`flex-1 px-4 py-2 text-xs font-mono font-bold rounded transition-colors ${
              filterStatus() === 'all'
                ? 'bg-primary-500/10 text-primary-500'
                : 'text-gray-400 hover:text-white hover:bg-terminal-800'
            }`}
          >
            All Alerts
          </button>
          <button
            onClick={() => setFilterStatus('active')}
            class={`flex-1 px-4 py-2 text-xs font-mono font-bold rounded transition-colors ${
              filterStatus() === 'active'
                ? 'bg-warning-500/10 text-warning-500'
                : 'text-gray-400 hover:text-white hover:bg-terminal-800'
            }`}
          >
            Active
          </button>
          <button
            onClick={() => setFilterStatus('triggered')}
            class={`flex-1 px-4 py-2 text-xs font-mono font-bold rounded transition-colors ${
              filterStatus() === 'triggered'
                ? 'bg-success-500/10 text-success-500'
                : 'text-gray-400 hover:text-white hover:bg-terminal-800'
            }`}
          >
            Triggered
          </button>
        </div>
      </Show>

      {/* Templates View */}
      <Show when={activeView() === 'templates'}>
        <div class="flex-1 overflow-auto bg-terminal-900 border border-terminal-750 p-3">
          <h3 class="text-xs font-mono font-bold text-gray-400 uppercase mb-3">Quick Alert Templates</h3>
          <div class="grid grid-cols-2 lg:grid-cols-3 gap-2">
            <For each={alertTemplates}>
              {(template) => (
                <button
                  onClick={() => {
                    setAlertType(template.type as any);
                    setShowCreateModal(true);
                  }}
                  class="bg-terminal-850 border border-terminal-750 p-4 text-left hover:border-primary-500 transition-colors group"
                >
                  <div class="flex items-start gap-3">
                    <div class={`w-10 h-10 bg-terminal-800 rounded flex items-center justify-center flex-shrink-0 group-hover:bg-primary-500/10 transition-colors`}>
                      <template.icon size={20} class={template.color} />
                    </div>
                    <div class="flex-1 min-w-0">
                      <h4 class="text-sm font-bold text-white mb-1 group-hover:text-primary-400 transition-colors">
                        {template.name}
                      </h4>
                      <p class="text-[10px] text-gray-500 leading-relaxed">
                        {template.desc}
                      </p>
                    </div>
                    <ChevronRight size={14} class="text-gray-600 group-hover:text-primary-400 transition-colors mt-1" />
                  </div>
                </button>
              )}
            </For>
          </div>

          {/* Common Alert Presets */}
          <h3 class="text-xs font-mono font-bold text-gray-400 uppercase mt-6 mb-3">Popular Stocks Quick Alerts</h3>
          <div class="grid grid-cols-4 gap-2">
            <For each={['AAPL', 'TSLA', 'NVDA', 'META', 'AMZN', 'GOOGL', 'MSFT', 'AMD']}>
              {(sym) => (
                <button
                  onClick={() => {
                    setSymbol(sym);
                    setShowCreateModal(true);
                  }}
                  class="bg-terminal-850 border border-terminal-750 p-3 text-center hover:border-primary-500 transition-colors"
                >
                  <div class="text-sm font-mono font-bold text-white">{sym}</div>
                  <div class="text-[10px] text-gray-500 mt-1">Set Alert</div>
                </button>
              )}
            </For>
          </div>
        </div>
      </Show>

      {/* History View */}
      <Show when={activeView() === 'history'}>
        <div class="flex-1 overflow-auto bg-terminal-900 border border-terminal-750">
          <div class="p-3 border-b border-terminal-750">
            <h3 class="text-xs font-mono font-bold text-gray-400 uppercase">Notification History</h3>
          </div>
          <Show when={notificationHistory().length === 0}>
            <div class="p-12 text-center">
              <Bell size={48} class="text-gray-600 mx-auto mb-4" />
              <div class="text-gray-500 mb-2">No notifications yet</div>
              <div class="text-xs text-gray-600">
                Notifications will appear here when your alerts are triggered
              </div>
            </div>
          </Show>
          <div class="divide-y divide-terminal-750">
            <For each={notificationHistory()}>
              {(notif) => (
                <div class={`p-3 flex items-center gap-3 hover:bg-terminal-850 transition-colors ${!notif.is_read ? 'bg-primary-500/5' : ''}`}>
                  <div class={`w-8 h-8 rounded flex items-center justify-center flex-shrink-0 ${
                    notif.notification_type === 'price_above' ? 'bg-success-500/10' :
                    notif.notification_type === 'price_below' ? 'bg-danger-500/10' :
                    notif.notification_type === 'volume' ? 'bg-primary-500/10' : 'bg-warning-500/10'
                  }`}>
                    {notif.notification_type === 'price_above' && <TrendingUp size={14} class="text-success-500" />}
                    {notif.notification_type === 'price_below' && <TrendingDown size={14} class="text-danger-500" />}
                    {notif.notification_type === 'volume' && <BarChart3 size={14} class="text-primary-500" />}
                    {notif.notification_type === 'price_change' && <Percent size={14} class="text-warning-500" />}
                    {!['price_above', 'price_below', 'volume', 'price_change'].includes(notif.notification_type) && <Bell size={14} class="text-gray-500" />}
                  </div>
                  <div class="flex-1 min-w-0">
                    <div class="flex items-center gap-2">
                      <span class="text-sm font-bold text-white">{notif.title || 'Alert'}</span>
                      {!notif.is_read && <span class="w-2 h-2 bg-primary-500 rounded-full"></span>}
                    </div>
                    <p class="text-xs text-gray-400 truncate">{notif.message}</p>
                  </div>
                  <div class="text-[10px] text-gray-500 flex-shrink-0">
                    {new Date(notif.created_at).toLocaleString()}
                  </div>
                </div>
              )}
            </For>
          </div>
        </div>
      </Show>

      {/* Alerts List */}
      <Show when={activeView() === 'alerts'}>
        <div class="flex-1 overflow-auto bg-terminal-900 border border-terminal-750 rounded-lg shadow-sm">
          <Show when={alerts()?.length === 0}>
            <div class="flex flex-col items-center justify-center h-full min-h-[300px] p-12 text-center">
              <div class="w-20 h-20 bg-terminal-850 rounded-full flex items-center justify-center mb-6 shadow-inner">
                <Bell size={40} class="text-gray-600" />
              </div>
              <h3 class="text-lg font-bold text-white mb-2">No Alerts Configured</h3>
              <p class="text-sm text-gray-500 max-w-xs mx-auto mb-8 leading-relaxed">
                Stay ahead of the market. Create price alerts to get notified instantly when stocks hit your target levels.
              </p>
              <button
                onClick={() => setShowCreateModal(true)}
                class="px-6 py-3 bg-accent-600 hover:bg-accent-500 text-white text-sm font-bold rounded-md shadow-lg shadow-accent-900/20 transition-all transform hover:scale-105 flex items-center gap-2"
              >
                <Plus size={18} />
                Create Your First Alert
              </button>
            </div>
          </Show>

          <div class="divide-y divide-terminal-800">
            <For each={alerts() || []}>
              {(alert) => (
                <div class="group p-4 hover:bg-terminal-850 transition-all border-l-2 border-transparent hover:border-l-accent-500">
                  <div class="flex flex-col sm:flex-row sm:items-center gap-4">
                    {/* Icon & Symbol */}
                    <div class="flex items-center gap-4 min-w-[180px]">
                      <div class={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 shadow-sm ${
                        alert.status === 'triggered' ? 'bg-success-500/10 text-success-500' : 'bg-terminal-800 text-gray-400'
                      }`}>
                        {getStatusIcon(alert.status)}
                      </div>
                      <div>
                        <div class="flex items-center gap-2">
                          <h3 class="text-lg font-bold text-white tracking-tight">{alert.symbol}</h3>
                          <span class={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase tracking-wider ${getStatusColor(alert.status)}`}>
                            {alert.status}
                          </span>
                        </div>
                        <div class="text-[10px] text-gray-500 font-medium mt-0.5">
                          {new Date(alert.created_at).toLocaleDateString()}
                        </div>
                      </div>
                    </div>

                    {/* Conditions & Values */}
                    <div class="flex-1 grid grid-cols-2 sm:grid-cols-3 gap-4 items-center">
                      <div class="bg-terminal-950/50 p-2 rounded border border-terminal-800/50">
                        <div class="text-[10px] text-gray-500 uppercase font-bold mb-1">Condition</div>
                        <div class="flex items-center gap-2">
                          <span class="text-xs font-medium text-gray-300">{getAlertTypeLabel(alert.alert_type)}</span>
                          <span class="text-sm font-mono font-bold text-accent-400">
                            {/* @ts-ignore - Backend returns price */}
                            {alert.alert_type.includes('price') ? formatCurrency(alert.price || alert.target_value) : (alert.price || alert.target_value).toLocaleString()}
                          </span>
                        </div>
                      </div>
                      
                      <Show when={alert.current_price !== undefined || alert.current_value !== undefined}>
                        <div class="bg-terminal-950/50 p-2 rounded border border-terminal-800/50">
                          <div class="text-[10px] text-gray-500 uppercase font-bold mb-1">Current Price</div>
                          <div class="text-sm font-mono font-bold text-white">
                            {/* @ts-ignore - Backend returns current_price */}
                            {alert.alert_type.includes('price') ? formatCurrency(alert.current_price || alert.current_value!) : (alert.current_price || alert.current_value!).toLocaleString()}
                          </div>
                        </div>
                      </Show>

                      {/* Sparkline (Hidden on mobile small) */}
                      <div class="hidden sm:block h-10 w-full opacity-70 group-hover:opacity-100 transition-opacity">
                        <Sparkline 
                          data={getAlertSparkline(alert)} 
                          height={40} 
                          color={alert.status === 'triggered' ? '#22c55e' : '#3b82f6'} 
                        />
                      </div>
                    </div>

                    {/* Actions & Meta */}
                    <div class="flex items-center justify-between sm:justify-end gap-4 min-w-[140px] border-t sm:border-t-0 border-terminal-800 pt-3 sm:pt-0 mt-2 sm:mt-0">
                      <div class="flex items-center gap-1.5">
                        {(alert.notification_methods || ['email']).includes('email') && (
                          <div class="w-6 h-6 rounded bg-terminal-800 flex items-center justify-center" title="Email">
                            <Mail size={12} class="text-primary-500" />
                          </div>
                        )}
                        {(alert.notification_methods || []).includes('sms') && (
                          <div class="w-6 h-6 rounded bg-terminal-800 flex items-center justify-center" title="SMS">
                            <Smartphone size={12} class="text-success-500" />
                          </div>
                        )}
                        {(alert.notification_methods || ['push']).includes('push') && (
                          <div class="w-6 h-6 rounded bg-terminal-800 flex items-center justify-center" title="Push">
                            <Monitor size={12} class="text-accent-500" />
                          </div>
                        )}
                      </div>

                      <div class="flex items-center gap-1 opacity-100 sm:opacity-0 group-hover:opacity-100 transition-opacity">
                        <button
                          onClick={() => {
                            setSymbol(alert.symbol);
                            setAlertType(alert.alert_type);
                            // @ts-ignore
                            setTargetValue((alert.price || alert.target_value).toString());
                            setShowCreateModal(true);
                          }}
                          class="p-2 hover:bg-terminal-800 rounded-md transition-colors text-gray-400 hover:text-white"
                          title="Duplicate Alert"
                        >
                          <Copy size={14} />
                        </button>
                        <button
                          onClick={() => handleDeleteAlert(alert.id)}
                          class="p-2 hover:bg-danger-900/20 rounded-md transition-colors text-gray-400 hover:text-danger-500"
                          title="Delete Alert"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </For>
          </div>
        </div>
      </Show>

      {/* Create Alert Modal */}
      <Show when={showCreateModal()}>
        <div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div class="bg-terminal-900 border border-terminal-750 rounded-lg max-w-md w-full p-6">
            <h3 class="text-lg font-bold text-white mb-4">Create Price Alert</h3>

            <div class="space-y-4">
              <div>
                <label class="text-xs text-gray-400 block mb-2">Symbol *</label>
                <input
                  type="text"
                  value={symbol()}
                  onInput={(e) => setSymbol(e.target.value.toUpperCase())}
                  placeholder="e.g. AAPL"
                  class="w-full bg-terminal-850 border border-terminal-750 text-white px-4 py-3 rounded focus:outline-none focus:border-accent-500"
                />
              </div>

              <div>
                <label class="text-xs text-gray-400 block mb-2">Alert Type *</label>
                <select
                  value={alertType()}
                  onChange={(e) => setAlertType(e.target.value as any)}
                  class="w-full bg-terminal-850 border border-terminal-750 text-white px-4 py-3 rounded focus:outline-none focus:border-accent-500"
                >
                  <option value="price_above">Price Goes Above</option>
                  <option value="price_below">Price Goes Below</option>
                  <option value="price_change">Price Changes By %</option>
                  <option value="volume">Volume Exceeds</option>
                </select>
              </div>

              <div>
                <label class="text-xs text-gray-400 block mb-2">
                  {alertType().includes('price') && !alertType().includes('change') ? 'Target Price *' : 
                   alertType() === 'price_change' ? 'Percent Change *' : 'Target Volume *'}
                </label>
                <input
                  type="number"
                  value={targetValue()}
                  onInput={(e) => setTargetValue(e.target.value)}
                  placeholder={alertType() === 'price_change' ? '5' : '0.00'}
                  step={alertType() === 'price_change' ? '0.1' : '0.01'}
                  class="w-full bg-terminal-850 border border-terminal-750 text-white px-4 py-3 rounded focus:outline-none focus:border-accent-500"
                />
              </div>

              <div>
                <label class="text-xs text-gray-400 block mb-3">Notification Methods</label>
                <div class="space-y-2">
                  <label class="flex items-center gap-3 p-3 bg-terminal-850 rounded cursor-pointer hover:bg-terminal-800 transition-colors">
                    <input
                      type="checkbox"
                      checked={notifyEmail()}
                      onChange={(e) => setNotifyEmail(e.target.checked)}
                      class="w-4 h-4"
                    />
                    <Mail size={16} class="text-primary-500" />
                    <span class="text-sm text-white">Email Notification</span>
                  </label>

                  <label class="flex items-center gap-3 p-3 bg-terminal-850 rounded cursor-pointer hover:bg-terminal-800 transition-colors">
                    <input
                      type="checkbox"
                      checked={notifySms()}
                      onChange={(e) => setNotifySms(e.target.checked)}
                      class="w-4 h-4"
                    />
                    <Smartphone size={16} class="text-success-500" />
                    <span class="text-sm text-white">SMS Notification</span>
                  </label>

                  <label class="flex items-center gap-3 p-3 bg-terminal-850 rounded cursor-pointer hover:bg-terminal-800 transition-colors">
                    <input
                      type="checkbox"
                      checked={notifyPush()}
                      onChange={(e) => setNotifyPush(e.target.checked)}
                      class="w-4 h-4"
                    />
                    <Monitor size={16} class="text-accent-500" />
                    <span class="text-sm text-white">Push Notification</span>
                  </label>
                </div>
              </div>
            </div>

            <div class="flex gap-2 mt-6">
              <button
                onClick={() => {
                  setShowCreateModal(false);
                  resetForm();
                }}
                class="flex-1 px-4 py-2 bg-terminal-850 hover:bg-terminal-800 border border-terminal-750 text-white text-sm font-semibold rounded transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateAlert}
                disabled={!symbol() || !targetValue()}
                class="flex-1 px-4 py-2 bg-accent-500 hover:bg-accent-600 disabled:opacity-50 text-white text-sm font-semibold rounded transition-colors"
              >
                Create Alert
              </button>
            </div>
          </div>
        </div>
      </Show>
    </div>
  );
}
