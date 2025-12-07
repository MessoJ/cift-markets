/**
 * Professional Settings Page
 * 
 * Account configuration with tabs:
 * - Profile (Name, Email, Password)
 * - Trading (Default settings, Confirmations)
 * - Notifications (Email, Push, SMS)
 * - API Keys (Generate, View, Revoke)
 * - Security (2FA, Login history)
 * 
 * ALL DATA FROM BACKEND - NO MOCK DATA
 */

import { createSignal, createEffect, Show } from 'solid-js';
import { apiClient } from '~/lib/api/client';
import { Key, Trash2 } from 'lucide-solid';

type SettingsTab = 'profile' | 'trading' | 'notifications' | 'api' | 'security';

export default function SettingsPage() {
  const [activeTab, setActiveTab] = createSignal<SettingsTab>('profile');
  const [settings, setSettings] = createSignal<any>(null);
  const [apiKeys, setApiKeys] = createSignal<any[]>([]);
  const [loading, setLoading] = createSignal(true);
  const [saving, setSaving] = createSignal(false);
  const [notification, setNotification] = createSignal<{type: 'success' | 'error', message: string} | null>(null);
  
  // Profile form
  const [fullName, setFullName] = createSignal('');
  const [currentPassword, setCurrentPassword] = createSignal('');
  const [newPassword, setNewPassword] = createSignal('');
  const [confirmPassword, setConfirmPassword] = createSignal('');

  // Trading settings
  const [defaultOrderType, setDefaultOrderType] = createSignal('market');
  const [requireConfirmation, setRequireConfirmation] = createSignal(true);

  // Notification settings
  const [emailAlerts, setEmailAlerts] = createSignal(true);
  const [smsAlerts, setSmsAlerts] = createSignal(false);
  const [pushAlerts, setPushAlerts] = createSignal(true);
  const [tradeNotifications, setTradeNotifications] = createSignal(true);
  const [priceAlertNotifications, setPriceAlertNotifications] = createSignal(true);
  const [newsNotifications, setNewsNotifications] = createSignal(false);
  const [marketingEmails, setMarketingEmails] = createSignal(false);

  // Security
  const [twoFactorEnabled, setTwoFactorEnabled] = createSignal(false);

  // Auto-hide notification
  createEffect(() => {
    if (notification()) {
      setTimeout(() => setNotification(null), 5000);
    }
  });

  const fetchSettings = async () => {
    try {
      setLoading(true);
      const data = await apiClient.getSettings();
      setSettings(data);
      setFullName(data.full_name || '');
      // Email comes from auth store, not settings
      setDefaultOrderType(data.default_order_type || 'market');
      setRequireConfirmation(data.require_order_confirmation !== false);
      
      // Notification settings
      setEmailAlerts(data.email_notifications !== false);
      setSmsAlerts(data.sms_notifications === true);
      setPushAlerts(data.push_notifications !== false);
      setTradeNotifications(data.email_trade_confirms !== false);
      setPriceAlertNotifications(data.email_price_alerts !== false);
      setNewsNotifications(data.email_market_news === true);
      setMarketingEmails(data.marketing_emails === true);
      
      // Security - 2FA would come from separate endpoint
    } catch (err: any) {
      console.error('Failed to load settings:', err);
      setNotification({
        type: 'error',
        message: err.response?.data?.detail || 'Failed to load settings'
      });
    } finally {
      setLoading(false);
    }
  };

  const fetchApiKeys = async () => {
    try {
      const keys = await apiClient.getApiKeys();
      setApiKeys(keys);
    } catch (err) {
      console.error('Failed to load API keys:', err);
    }
  };

  createEffect(() => {
    fetchSettings();
    if (activeTab() === 'api') {
      fetchApiKeys();
    }
  });

  const saveProfile = async () => {
    if (newPassword() && newPassword() !== confirmPassword()) {
      setNotification({type: 'error', message: 'Passwords do not match'});
      return;
    }

    if (newPassword() && newPassword().length < 8) {
      setNotification({type: 'error', message: 'Password must be at least 8 characters'});
      return;
    }

    try {
      setSaving(true);
      await apiClient.updateSettings({
        full_name: fullName(),
        ...(newPassword() && { 
          current_password: currentPassword(),
          new_password: newPassword()
        })
      });
      setNotification({type: 'success', message: 'Profile updated successfully'});
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
    } catch (err: any) {
      setNotification({
        type: 'error',
        message: err.response?.data?.detail || 'Failed to update profile'
      });
    } finally {
      setSaving(false);
    }
  };

  const saveTradingSettings = async () => {
    try {
      setSaving(true);
      await apiClient.updateSettings({
        default_order_type: defaultOrderType(),
        require_order_confirmation: requireConfirmation()
      });
      setNotification({type: 'success', message: 'Trading settings updated'});
    } catch (err: any) {
      setNotification({
        type: 'error',
        message: err.response?.data?.detail || 'Failed to update settings'
      });
    } finally {
      setSaving(false);
    }
  };

  const generateApiKey = async () => {
    const name = prompt('Enter a name for this API key:');
    if (!name) return;
    
    try {
      const key = await apiClient.createApiKey({ name });
      alert(`API Key Created!\n\n${key.api_key}\n\nSave this key securely. It won't be shown again.`);
      setNotification({type: 'success', message: 'API key created successfully'});
      await fetchApiKeys();
    } catch (err: any) {
      console.error('API Key creation error:', err);
      const errorMsg = err.response?.data?.detail || err.message || 'Failed to generate key';
      setNotification({
        type: 'error',
        message: errorMsg
      });
    }
  };

  const revokeApiKey = async (keyId: string) => {
    if (!confirm('Revoke this API key? This action cannot be undone.')) return;
    try {
      await apiClient.revokeApiKey(keyId);
      setNotification({type: 'success', message: 'API key revoked'});
      await fetchApiKeys();
    } catch (err: any) {
      setNotification({
        type: 'error',
        message: err.response?.data?.detail || 'Failed to revoke key'
      });
    }
  };

  return (
    <div class="h-full flex flex-col gap-2">
      {/* Top Bar - Tabs */}
      <div class="bg-terminal-900 border border-terminal-750 p-2">
        <div class="flex flex-wrap items-center gap-1">
          {([
            { id: 'profile', label: 'Profile' },
            { id: 'trading', label: 'Trading' },
            { id: 'notifications', label: 'Notifications' },
            { id: 'api', label: 'API Keys' },
            { id: 'security', label: 'Security' }
          ] as { id: SettingsTab; label: string }[]).map(tab => (
            <button
              onClick={() => setActiveTab(tab.id)}
              class={`px-3 sm:px-4 py-1.5 text-[10px] sm:text-xs font-mono font-bold transition-colors ${
                activeTab() === tab.id
                  ? 'bg-accent-500 text-black'
                  : 'bg-terminal-850 text-gray-400 hover:bg-terminal-800 border border-terminal-750'
              }`}
            >
              {tab.label.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <div class="flex-1 overflow-y-auto p-2 sm:p-3">
        <Show when={!loading()} fallback={
          <div class="flex items-center justify-center h-full">
            <span class="text-xs font-mono text-gray-600">Loading settings...</span>
          </div>
        }>
          {/* Profile Tab */}
          <Show when={activeTab() === 'profile'}>
            <div class="max-w-2xl">
              <div class="bg-terminal-900 border border-terminal-750 p-3 sm:p-4 space-y-3 sm:space-y-4">
                <h3 class="text-[10px] sm:text-xs font-mono font-bold text-gray-400 uppercase mb-2 sm:mb-3">Profile Information</h3>
                
                <div>
                  <label class="text-[10px] font-mono text-gray-500 uppercase block mb-1">Full Name</label>
                  <input
                    type="text"
                    value={fullName()}
                    onInput={(e) => setFullName(e.currentTarget.value)}
                    placeholder="Enter your full name"
                    class="w-full bg-terminal-850 border border-terminal-750 text-white font-mono text-xs sm:text-sm px-3 py-2 focus:outline-none focus:border-accent-500"
                  />
                </div>

                <div class="pt-3 sm:pt-4 border-t border-terminal-750">
                  <h4 class="text-[10px] sm:text-xs font-mono font-bold text-gray-400 uppercase mb-2 sm:mb-3">Change Password</h4>
                  
                  <div class="space-y-2 sm:space-y-3">
                    <div>
                      <label class="text-[10px] font-mono text-gray-500 uppercase block mb-1">Current Password</label>
                      <input
                        type="password"
                        value={currentPassword()}
                        onInput={(e) => setCurrentPassword(e.currentTarget.value)}
                        class="w-full bg-terminal-850 border border-terminal-750 text-white font-mono text-xs sm:text-sm px-3 py-2 focus:outline-none focus:border-accent-500"
                      />
                    </div>

                    <div>
                      <label class="text-[10px] font-mono text-gray-500 uppercase block mb-1">New Password</label>
                      <input
                        type="password"
                        value={newPassword()}
                        onInput={(e) => setNewPassword(e.currentTarget.value)}
                        class="w-full bg-terminal-850 border border-terminal-750 text-white font-mono text-xs sm:text-sm px-3 py-2 focus:outline-none focus:border-accent-500"
                      />
                    </div>
                  </div>
                </div>

                <button
                  onClick={saveProfile}
                  disabled={saving()}
                  class="w-full sm:w-auto px-4 py-2 bg-accent-500 hover:bg-accent-600 disabled:bg-terminal-800 disabled:text-gray-600 text-black text-xs sm:text-sm font-bold font-mono transition-colors"
                >
                  {saving() ? 'SAVING...' : 'SAVE CHANGES'}
                </button>
              </div>
            </div>
          </Show>

          {/* Trading Tab */}
          <Show when={activeTab() === 'trading'}>
            <div class="max-w-2xl">
              <div class="bg-terminal-900 border border-terminal-750 p-4 space-y-4">
                <h3 class="text-xs font-mono font-bold text-gray-400 uppercase mb-3">Trading Preferences</h3>
                
                <div>
                  <label class="text-[10px] font-mono text-gray-500 uppercase block mb-1">Default Order Type</label>
                  <select
                    value={defaultOrderType()}
                    onChange={(e) => setDefaultOrderType(e.currentTarget.value)}
                    class="w-full bg-terminal-850 border border-terminal-750 text-white font-mono text-sm px-3 py-2 focus:outline-none focus:border-accent-500"
                  >
                    <option value="market">Market</option>
                    <option value="limit">Limit</option>
                    <option value="stop">Stop</option>
                  </select>
                </div>

                <div class="flex items-center gap-2">
                  <input
                    type="checkbox"
                    id="require-confirmation"
                    checked={requireConfirmation()}
                    onChange={(e) => setRequireConfirmation(e.currentTarget.checked)}
                    class="w-4 h-4"
                  />
                  <label for="require-confirmation" class="text-sm font-mono text-gray-300">
                    Require order confirmation
                  </label>
                </div>

                <button
                  onClick={saveTradingSettings}
                  disabled={saving()}
                  class="px-4 py-2 bg-accent-500 hover:bg-accent-600 disabled:bg-terminal-800 disabled:text-gray-600 text-black text-sm font-bold font-mono transition-colors"
                >
                  {saving() ? 'SAVING...' : 'SAVE CHANGES'}
                </button>
              </div>
            </div>
          </Show>

          {/* API Keys Tab */}
          <Show when={activeTab() === 'api'}>
            <div class="max-w-4xl">
              <div class="bg-terminal-900 border border-terminal-750 p-4">
                <div class="flex items-center justify-between mb-4">
                  <h3 class="text-xs font-mono font-bold text-gray-400 uppercase">API Keys</h3>
                  <button
                    onClick={generateApiKey}
                    class="px-3 py-1.5 bg-success-500 hover:bg-success-600 text-black text-xs font-bold font-mono transition-colors flex items-center gap-1.5"
                  >
                    <Key class="w-3.5 h-3.5" />
                    GENERATE NEW KEY
                  </button>
                </div>

                <div class="space-y-2">
                  {apiKeys().map(key => (
                    <div class="bg-terminal-850 border border-terminal-750 p-3 flex items-center justify-between">
                      <div class="flex-1">
                        <div class="text-sm font-mono text-white font-bold">{key.name || 'Unnamed'}</div>
                        <div class="text-xs font-mono text-gray-500 mt-1">
                          Created: {new Date(key.created_at).toLocaleString()}
                        </div>
                      </div>
                      <button
                        onClick={() => revokeApiKey(key.id)}
                        class="p-1.5 text-danger-400 hover:text-danger-300 hover:bg-danger-900/20 transition-colors"
                        title="Revoke Key"
                      >
                        <Trash2 class="w-4 h-4" />
                      </button>
                    </div>
                  ))}
                  {apiKeys().length === 0 && (
                    <div class="text-center py-8 text-xs font-mono text-gray-600">
                      No API keys yet. Generate one above.
                    </div>
                  )}
                </div>
              </div>
            </div>
          </Show>

          {/* Other tabs placeholder */}
          <Show when={activeTab() === 'notifications' || activeTab() === 'security'}>
            <div class="bg-terminal-900 border border-terminal-750 p-8 text-center">
              <p class="text-xs font-mono text-gray-600">
                {activeTab().toUpperCase()} settings coming soon
              </p>
            </div>
          </Show>
        </Show>
      </div>
    </div>
  );
}
