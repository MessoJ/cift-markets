import { createSignal, createEffect, Show, For } from 'solid-js';
import { apiClient, ApiKey, SessionLog } from '~/lib/api/client';
import { 
  Key, 
  Trash2, 
  Shield, 
  Bell, 
  Smartphone, 
  Globe, 
  Monitor, 
  LogOut,
  Plus,
  CheckCircle2,
  AlertTriangle,
  Copy,
  X
} from 'lucide-solid';

type SettingsTab = 'security' | 'notifications' | 'api' | 'preferences';

export default function SettingsPage() {
  const [activeTab, setActiveTab] = createSignal<SettingsTab>('security');
  const [settings, setSettings] = createSignal<any>(null);
  const [apiKeys, setApiKeys] = createSignal<ApiKey[]>([]);
  const [sessions, setSessions] = createSignal<SessionLog[]>([]);
  const [_loading, setLoading] = createSignal(true);
  const [notification, setNotification] = createSignal<{type: 'success' | 'error', message: string} | null>(null);

  // API Key Form
  const [showCreateKeyModal, setShowCreateKeyModal] = createSignal(false);
  const [newKeyName, setNewKeyName] = createSignal('');
  const [newKeyPermissions, setNewKeyPermissions] = createSignal({ read: true, trade: false, withdraw: false });
  const [newKeyIps, setNewKeyIps] = createSignal('');
  const [createdKeySecret, setCreatedKeySecret] = createSignal<string | null>(null);

  // Notification Settings
  const [notifyConfig, setNotifyConfig] = createSignal({
    trade_email: true, trade_push: true, trade_sms: false,
    price_email: true, price_push: true, price_sms: false,
    security_email: true, security_push: true, security_sms: true,
    marketing_email: false
  });

  // 2FA State
  const [show2FASetup, setShow2FASetup] = createSignal(false);
  const [twoFactorCode, setTwoFactorCode] = createSignal('');

  const fetchSettings = async () => {
    try {
      setLoading(true);
      const [settingsData, sessionsData] = await Promise.all([
        apiClient.getSettings(),
        apiClient.getSessionHistory()
      ]);
      setSettings(settingsData);
      setSessions(sessionsData);
      
      // Map backend settings to local config state
      setNotifyConfig({
        trade_email: settingsData.email_trade_confirms !== false,
        trade_push: settingsData.push_notifications !== false,
        trade_sms: settingsData.sms_notifications === true,
        price_email: settingsData.email_price_alerts !== false,
        price_push: settingsData.push_notifications !== false,
        price_sms: settingsData.sms_notifications === true,
        security_email: true, // Always on
        security_push: true,
        security_sms: true,
        marketing_email: settingsData.marketing_emails === true
      });

    } catch (err: any) {
      console.error('Failed to load settings:', err);
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
    if (activeTab() === 'api') fetchApiKeys();
  });

  const handleCreateKey = async () => {
    const name = newKeyName().trim();
    if (!name) {
      setNotification({ type: 'error', message: 'API Key name is required' });
      return;
    }

    if (name.length > 50) {
      setNotification({ type: 'error', message: 'API Key name must be less than 50 characters' });
      return;
    }

    // Validate IP addresses if provided
    const ips = newKeyIps() ? newKeyIps().split(',').map(ip => ip.trim()) : [];
    const ipRegex = /^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/;
    
    for (const ip of ips) {
      if (ip && !ipRegex.test(ip)) {
        setNotification({ type: 'error', message: `Invalid IP address: ${ip}` });
        return;
      }
    }

    try {
      const response = await apiClient.createApiKey({
        name: name,
        permissions: Object.keys(newKeyPermissions()).filter(k => newKeyPermissions()[k as keyof typeof newKeyPermissions]),
        ip_whitelist: ips.length > 0 ? ips : undefined
      });
      setCreatedKeySecret(response.secret || response.api_key);
      fetchApiKeys();
      setNotification({ type: 'success', message: 'API Key created successfully' });
    } catch (err: any) {
      setNotification({ type: 'error', message: err.message || 'Failed to create API key' });
    }
  };

  const handleDeleteKey = async (id: string) => {
    if (!confirm('Are you sure you want to revoke this API key? This action cannot be undone.')) return;
    try {
      await apiClient.revokeApiKey(id);
      fetchApiKeys();
      setNotification({ type: 'success', message: 'API Key revoked' });
    } catch (err: any) {
      setNotification({ type: 'error', message: 'Failed to revoke key' });
    }
  };

  const handleTerminateSession = async (id: string) => {
    try {
      await apiClient.terminateSession(id);
      const sessionsData = await apiClient.getSessionHistory();
      setSessions(sessionsData);
      setNotification({ type: 'success', message: 'Session terminated' });
    } catch (err) {
      setNotification({ type: 'error', message: 'Failed to terminate session' });
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setNotification({ type: 'success', message: 'Copied to clipboard' });
  };

  return (
    <div class="h-full flex flex-col bg-black text-white overflow-hidden">
      {/* Header */}
      <div class="p-6 border-b border-terminal-800">
        <h1 class="text-2xl font-bold font-mono tracking-tight">SETTINGS</h1>
        <p class="text-gray-500 font-mono text-sm mt-1">Manage security, notifications, and API access</p>
      </div>

      {/* Notification Toast */}
      <Show when={notification()}>
        <div class={`absolute top-4 right-4 z-50 px-4 py-3 rounded-md border font-mono text-sm shadow-lg animate-in fade-in slide-in-from-top-2 ${
          notification()?.type === 'success' 
            ? 'bg-success-900/90 border-success-700 text-success-100' 
            : 'bg-danger-900/90 border-danger-700 text-danger-100'
        }`}>
          <div class="flex items-center gap-2">
            <Show when={notification()?.type === 'success'} fallback={<AlertTriangle size={16} />}>
              <CheckCircle2 size={16} />
            </Show>
            {notification()?.message}
          </div>
        </div>
      </Show>

      <div class="flex flex-col md:flex-row flex-1 overflow-hidden">
        {/* Sidebar Navigation (Desktop) / Top Bar (Mobile) */}
        <div class="w-full md:w-64 border-b md:border-b-0 md:border-r border-terminal-800 bg-terminal-900/50 flex flex-row md:flex-col overflow-x-auto md:overflow-visible shrink-0">
          <nav class="p-2 flex md:flex-col gap-1 min-w-max md:min-w-0">
            <button
              onClick={() => setActiveTab('security')}
              class={`flex items-center gap-3 px-4 py-3 text-sm font-mono rounded-md transition-colors whitespace-nowrap ${
                activeTab() === 'security' ? 'bg-terminal-800 text-accent-400' : 'text-gray-400 hover:bg-terminal-800/50 hover:text-gray-200'
              }`}
            >
              <Shield size={18} />
              Security
            </button>
            <button
              onClick={() => setActiveTab('notifications')}
              class={`flex items-center gap-3 px-4 py-3 text-sm font-mono rounded-md transition-colors whitespace-nowrap ${
                activeTab() === 'notifications' ? 'bg-terminal-800 text-accent-400' : 'text-gray-400 hover:bg-terminal-800/50 hover:text-gray-200'
              }`}
            >
              <Bell size={18} />
              Notifications
            </button>
            <button
              onClick={() => setActiveTab('api')}
              class={`flex items-center gap-3 px-4 py-3 text-sm font-mono rounded-md transition-colors whitespace-nowrap ${
                activeTab() === 'api' ? 'bg-terminal-800 text-accent-400' : 'text-gray-400 hover:bg-terminal-800/50 hover:text-gray-200'
              }`}
            >
              <Key size={18} />
              API Management
            </button>
            <button
              onClick={() => setActiveTab('preferences')}
              class={`flex items-center gap-3 px-4 py-3 text-sm font-mono rounded-md transition-colors whitespace-nowrap ${
                activeTab() === 'preferences' ? 'bg-terminal-800 text-accent-400' : 'text-gray-400 hover:bg-terminal-800/50 hover:text-gray-200'
              }`}
            >
              <Globe size={18} />
              Preferences
            </button>
          </nav>
        </div>

        {/* Content Area */}
        <div class="flex-1 overflow-y-auto p-4 md:p-8">
          
          {/* SECURITY TAB */}
          <Show when={activeTab() === 'security'}>
            <div class="max-w-3xl space-y-8">
              
              {/* 2FA Section */}
              <div class="bg-terminal-900 border border-terminal-800 rounded-lg p-6">
                <div class="flex items-start justify-between mb-6">
                  <div>
                    <h3 class="text-lg font-bold text-white font-mono flex items-center gap-2">
                      <Smartphone class="text-accent-500" size={20} />
                      Two-Factor Authentication
                    </h3>
                    <p class="text-gray-400 text-sm mt-1">Secure your account with an authenticator app.</p>
                  </div>
                  <div class={`px-3 py-1 rounded-full text-xs font-bold font-mono border ${
                    settings()?.two_factor_enabled 
                      ? 'bg-success-900/20 text-success-400 border-success-900/50' 
                      : 'bg-warning-900/20 text-warning-400 border-warning-900/50'
                  }`}>
                    {settings()?.two_factor_enabled ? 'ENABLED' : 'DISABLED'}
                  </div>
                </div>

                <Show when={!settings()?.two_factor_enabled}>
                  <div class="bg-terminal-800/50 rounded-lg p-4 border border-terminal-700">
                    <Show when={!show2FASetup()} fallback={
                      <div class="space-y-4">
                        <div class="flex gap-6">
                          <div class="bg-white p-2 rounded-lg w-32 h-32 flex items-center justify-center">
                            {/* Placeholder QR */}
                            <div class="w-full h-full bg-gray-900/10 grid grid-cols-6 grid-rows-6 gap-0.5">
                              <For each={Array(36)}>{() => <div class={`bg-black ${Math.random() > 0.5 ? 'opacity-100' : 'opacity-0'}`} />}</For>
                            </div>
                          </div>
                          <div class="flex-1 space-y-3">
                            <p class="text-sm text-gray-300">1. Scan this QR code with Google Authenticator or Authy.</p>
                            <p class="text-sm text-gray-300">2. Enter the 6-digit code below.</p>
                            <div class="flex gap-2">
                              <input 
                                type="text" 
                                placeholder="000 000" 
                                class="bg-terminal-900 border border-terminal-700 rounded px-3 py-2 font-mono text-center w-32 focus:border-accent-500 outline-none"
                                value={twoFactorCode()}
                                onInput={(e) => setTwoFactorCode(e.currentTarget.value)}
                              />
                              <button class="bg-accent-600 hover:bg-accent-500 text-white px-4 py-2 rounded font-mono text-sm font-bold">
                                Verify & Enable
                              </button>
                            </div>
                          </div>
                        </div>
                      </div>
                    }>
                      <div class="flex items-center justify-between">
                        <p class="text-sm text-gray-400">Protect your withdrawals and API keys.</p>
                        <button 
                          onClick={() => setShow2FASetup(true)}
                          class="bg-accent-600 hover:bg-accent-500 text-white px-4 py-2 rounded font-mono text-sm font-bold transition-colors"
                        >
                          Setup 2FA
                        </button>
                      </div>
                    </Show>
                  </div>
                </Show>
              </div>

              {/* Active Sessions */}
              <div class="bg-terminal-900 border border-terminal-800 rounded-lg p-6">
                <h3 class="text-lg font-bold text-white font-mono flex items-center gap-2 mb-4">
                  <Monitor class="text-accent-500" size={20} />
                  Active Sessions
                </h3>
                <div class="space-y-3">
                  <For each={sessions()}>
                    {(session) => (
                      <div class="flex items-center justify-between p-4 bg-terminal-800/30 border border-terminal-800 rounded-lg">
                        <div class="flex items-center gap-4">
                          <div class={`p-2 rounded-full ${session.is_current ? 'bg-success-900/20 text-success-400' : 'bg-terminal-700 text-gray-400'}`}>
                            <Monitor size={20} />
                          </div>
                          <div>
                            <div class="flex items-center gap-2">
                              <span class="font-bold text-white font-mono">{session.device || session.device_type || 'Unknown Device'}</span>
                              <Show when={session.is_current}>
                                <span class="text-[10px] bg-success-900/20 text-success-400 px-1.5 py-0.5 rounded border border-success-900/30 font-mono">CURRENT</span>
                              </Show>
                            </div>
                            <div class="text-xs text-gray-500 font-mono mt-1">
                              {session.ip_address} • {session.location || session.city || 'Unknown Location'} • Last active: {new Date(session.last_active || session.last_activity_at).toLocaleString()}
                            </div>
                          </div>
                        </div>
                        <Show when={!session.is_current}>
                          <button 
                            onClick={() => handleTerminateSession(session.id)}
                            class="text-danger-400 hover:text-danger-300 p-2 hover:bg-danger-900/20 rounded transition-colors"
                            title="Terminate Session"
                          >
                            <LogOut size={18} />
                          </button>
                        </Show>
                      </div>
                    )}
                  </For>
                  <Show when={sessions().length === 0}>
                    <div class="text-center py-8 text-gray-500 font-mono text-sm">
                      No active sessions found.
                    </div>
                  </Show>
                </div>
              </div>
            </div>
          </Show>

          {/* NOTIFICATIONS TAB */}
          <Show when={activeTab() === 'notifications'}>
            <div class="max-w-4xl bg-terminal-900 border border-terminal-800 rounded-lg overflow-hidden">
              <div class="grid grid-cols-4 gap-4 p-4 border-b border-terminal-800 bg-terminal-800/50 font-mono text-xs font-bold text-gray-400 uppercase tracking-wider">
                <div class="col-span-1">Notification Type</div>
                <div class="text-center">Email</div>
                <div class="text-center">Push</div>
                <div class="text-center">SMS</div>
              </div>
              
              <div class="divide-y divide-terminal-800">
                {/* Trade Confirmations */}
                <div class="grid grid-cols-4 gap-4 p-6 items-center hover:bg-terminal-800/30 transition-colors">
                  <div>
                    <div class="font-bold text-white font-mono">Trade Confirmations</div>
                    <div class="text-xs text-gray-500 mt-1">Fills, partial fills, and cancellations</div>
                  </div>
                  <div class="flex justify-center"><input type="checkbox" checked={notifyConfig().trade_email} class="accent-accent-500 w-4 h-4" /></div>
                  <div class="flex justify-center"><input type="checkbox" checked={notifyConfig().trade_push} class="accent-accent-500 w-4 h-4" /></div>
                  <div class="flex justify-center"><input type="checkbox" checked={notifyConfig().trade_sms} class="accent-accent-500 w-4 h-4" /></div>
                </div>

                {/* Price Alerts */}
                <div class="grid grid-cols-4 gap-4 p-6 items-center hover:bg-terminal-800/30 transition-colors">
                  <div>
                    <div class="font-bold text-white font-mono">Price Alerts</div>
                    <div class="text-xs text-gray-500 mt-1">Target price hits and volatility warnings</div>
                  </div>
                  <div class="flex justify-center"><input type="checkbox" checked={notifyConfig().price_email} class="accent-accent-500 w-4 h-4" /></div>
                  <div class="flex justify-center"><input type="checkbox" checked={notifyConfig().price_push} class="accent-accent-500 w-4 h-4" /></div>
                  <div class="flex justify-center"><input type="checkbox" checked={notifyConfig().price_sms} class="accent-accent-500 w-4 h-4" /></div>
                </div>

                {/* Security Alerts */}
                <div class="grid grid-cols-4 gap-4 p-6 items-center hover:bg-terminal-800/30 transition-colors">
                  <div>
                    <div class="font-bold text-white font-mono">Security Alerts</div>
                    <div class="text-xs text-gray-500 mt-1">New device logins and password changes</div>
                  </div>
                  <div class="flex justify-center"><input type="checkbox" checked={true} disabled class="accent-accent-500 w-4 h-4 opacity-50 cursor-not-allowed" /></div>
                  <div class="flex justify-center"><input type="checkbox" checked={true} disabled class="accent-accent-500 w-4 h-4 opacity-50 cursor-not-allowed" /></div>
                  <div class="flex justify-center"><input type="checkbox" checked={true} disabled class="accent-accent-500 w-4 h-4 opacity-50 cursor-not-allowed" /></div>
                </div>

                {/* Marketing */}
                <div class="grid grid-cols-4 gap-4 p-6 items-center hover:bg-terminal-800/30 transition-colors">
                  <div>
                    <div class="font-bold text-white font-mono">Marketing & News</div>
                    <div class="text-xs text-gray-500 mt-1">Product updates and daily newsletters</div>
                  </div>
                  <div class="flex justify-center"><input type="checkbox" checked={notifyConfig().marketing_email} class="accent-accent-500 w-4 h-4" /></div>
                  <div class="flex justify-center text-gray-600">-</div>
                  <div class="flex justify-center text-gray-600">-</div>
                </div>
              </div>
              
              <div class="p-4 bg-terminal-800/30 border-t border-terminal-800 flex justify-end">
                <button class="bg-accent-600 hover:bg-accent-500 text-white px-6 py-2 rounded font-mono text-sm font-bold transition-colors">
                  Save Preferences
                </button>
              </div>
            </div>
          </Show>

          {/* API MANAGEMENT TAB */}
          <Show when={activeTab() === 'api'}>
            <div class="max-w-4xl space-y-6">
              <div class="flex justify-between items-center">
                <div>
                  <h3 class="text-lg font-bold text-white font-mono">API Keys</h3>
                  <p class="text-gray-400 text-sm mt-1">Manage programmatic access to your account.</p>
                </div>
                <button 
                  onClick={() => setShowCreateKeyModal(true)}
                  class="flex items-center gap-2 bg-accent-600 hover:bg-accent-500 text-white px-4 py-2 rounded font-mono text-sm font-bold transition-colors"
                >
                  <Plus size={16} />
                  Create New Key
                </button>
              </div>

              {/* Created Key Modal/Display */}
              <Show when={createdKeySecret()}>
                <div class="bg-success-900/20 border border-success-900/50 rounded-lg p-6 mb-6">
                  <div class="flex items-start gap-3">
                    <CheckCircle2 class="text-success-400 shrink-0" size={24} />
                    <div class="flex-1">
                      <h4 class="text-lg font-bold text-success-400 font-mono mb-2">API Key Created Successfully</h4>
                      <p class="text-sm text-gray-300 mb-4">
                        Please copy your secret key now. For security reasons, it will never be shown again.
                      </p>
                      <div class="bg-black border border-success-900/30 rounded p-3 flex items-center justify-between font-mono text-sm text-white">
                        <span class="truncate">{createdKeySecret()}</span>
                        <button 
                          onClick={() => copyToClipboard(createdKeySecret()!)}
                          class="text-gray-400 hover:text-white transition-colors"
                        >
                          <Copy size={16} />
                        </button>
                      </div>
                      <button 
                        onClick={() => setCreatedKeySecret(null)}
                        class="mt-4 text-sm text-success-400 hover:text-success-300 font-mono underline"
                      >
                        I have saved my secret key
                      </button>
                    </div>
                  </div>
                </div>
              </Show>

              {/* Create Key Form */}
              <Show when={showCreateKeyModal() && !createdKeySecret()}>
                <div class="bg-terminal-900 border border-terminal-800 rounded-lg p-6 animate-in fade-in slide-in-from-top-4">
                  <div class="flex justify-between items-center mb-4">
                    <h4 class="font-bold text-white font-mono">Generate New API Key</h4>
                    <button onClick={() => setShowCreateKeyModal(false)} class="text-gray-500 hover:text-white"><X size={20} /></button>
                  </div>
                  
                  <div class="space-y-4">
                    <div>
                      <label class="block text-xs font-mono text-gray-400 mb-1">Key Label</label>
                      <input 
                        type="text" 
                        placeholder="e.g. Trading Bot 1"
                        class="w-full bg-terminal-800 border border-terminal-700 rounded px-3 py-2 text-white font-mono focus:border-accent-500 outline-none"
                        value={newKeyName()}
                        onInput={(e) => setNewKeyName(e.currentTarget.value)}
                      />
                    </div>

                    <div>
                      <label class="block text-xs font-mono text-gray-400 mb-2">Permissions</label>
                      <div class="flex gap-4">
                        <label class="flex items-center gap-2 cursor-pointer">
                          <input 
                            type="checkbox" 
                            checked={newKeyPermissions().read}
                            disabled
                            class="accent-accent-500"
                          />
                          <span class="text-sm text-gray-300 font-mono">Read Data</span>
                        </label>
                        <label class="flex items-center gap-2 cursor-pointer">
                          <input 
                            type="checkbox" 
                            checked={newKeyPermissions().trade}
                            onChange={(e) => setNewKeyPermissions(p => ({...p, trade: e.currentTarget.checked}))}
                            class="accent-accent-500"
                          />
                          <span class="text-sm text-gray-300 font-mono">Spot Trading</span>
                        </label>
                        <label class="flex items-center gap-2 cursor-pointer">
                          <input 
                            type="checkbox" 
                            checked={newKeyPermissions().withdraw}
                            onChange={(e) => setNewKeyPermissions(p => ({...p, withdraw: e.currentTarget.checked}))}
                            class="accent-accent-500"
                          />
                          <span class="text-sm text-gray-300 font-mono">Withdrawals</span>
                        </label>
                      </div>
                    </div>

                    <div>
                      <label class="block text-xs font-mono text-gray-400 mb-1">IP Whitelist (Optional)</label>
                      <input 
                        type="text" 
                        placeholder="e.g. 192.168.1.1, 10.0.0.1"
                        class="w-full bg-terminal-800 border border-terminal-700 rounded px-3 py-2 text-white font-mono focus:border-accent-500 outline-none"
                        value={newKeyIps()}
                        onInput={(e) => setNewKeyIps(e.currentTarget.value)}
                      />
                      <p class="text-[10px] text-gray-500 mt-1">Comma separated list of allowed IP addresses. Leave empty to allow all IPs (Not Recommended).</p>
                    </div>

                    <div class="flex justify-end gap-3 mt-6">
                      <button 
                        onClick={() => setShowCreateKeyModal(false)}
                        class="px-4 py-2 text-sm font-mono text-gray-400 hover:text-white transition-colors"
                      >
                        Cancel
                      </button>
                      <button 
                        onClick={handleCreateKey}
                        class="bg-accent-600 hover:bg-accent-500 text-white px-4 py-2 rounded font-mono text-sm font-bold transition-colors"
                      >
                        Generate Key
                      </button>
                    </div>
                  </div>
                </div>
              </Show>

              {/* Keys List */}
              <div class="space-y-3">
                <For each={apiKeys()}>
                  {(key) => (
                    <div class="bg-terminal-900 border border-terminal-800 rounded-lg p-4 flex items-center justify-between">
                      <div>
                        <div class="flex items-center gap-3">
                          <span class="font-bold text-white font-mono">{key.name}</span>
                          <div class="flex gap-1">
                            <For each={key.permissions || key.scopes}>
                              {(perm) => (
                                <span class="text-[10px] bg-terminal-800 text-gray-400 px-1.5 py-0.5 rounded border border-terminal-700 uppercase">{perm}</span>
                              )}
                            </For>
                          </div>
                        </div>
                        <div class="text-xs text-gray-500 font-mono mt-1 flex items-center gap-4">
                          <span>Prefix: {key.prefix || key.key_prefix}****</span>
                          <span>Created: {new Date(key.created_at).toLocaleDateString()}</span>
                          <span>Last Used: {key.last_used_at ? new Date(key.last_used_at).toLocaleDateString() : 'Never'}</span>
                        </div>
                      </div>
                      <button 
                        onClick={() => handleDeleteKey(key.id)}
                        class="text-gray-500 hover:text-danger-400 p-2 hover:bg-danger-900/10 rounded transition-colors"
                        title="Revoke Key"
                      >
                        <Trash2 size={18} />
                      </button>
                    </div>
                  )}
                </For>
                <Show when={apiKeys().length === 0}>
                  <div class="text-center py-12 border border-dashed border-terminal-800 rounded-lg">
                    <Key class="mx-auto text-terminal-700 mb-3" size={32} />
                    <p class="text-gray-500 font-mono text-sm">No API keys generated yet.</p>
                  </div>
                </Show>
              </div>
            </div>
          </Show>

          {/* PREFERENCES TAB */}
          <Show when={activeTab() === 'preferences'}>
            <div class="max-w-3xl bg-terminal-900 border border-terminal-800 rounded-lg p-6">
              <h3 class="text-lg font-bold text-white font-mono mb-6">Global Preferences</h3>
              
              <div class="space-y-6">
                <div class="grid grid-cols-3 gap-4 items-center">
                  <div class="col-span-1 text-sm text-gray-400 font-mono">Language</div>
                  <div class="col-span-2">
                    <select class="w-full bg-terminal-800 border border-terminal-700 rounded px-3 py-2 text-white font-mono focus:border-accent-500 outline-none">
                      <option value="en">English (US)</option>
                      <option value="es">Español</option>
                      <option value="fr">Français</option>
                      <option value="zh">中文</option>
                    </select>
                  </div>
                </div>

                <div class="grid grid-cols-3 gap-4 items-center">
                  <div class="col-span-1 text-sm text-gray-400 font-mono">Base Currency</div>
                  <div class="col-span-2">
                    <select class="w-full bg-terminal-800 border border-terminal-700 rounded px-3 py-2 text-white font-mono focus:border-accent-500 outline-none">
                      <option value="USD">USD - US Dollar</option>
                      <option value="EUR">EUR - Euro</option>
                      <option value="GBP">GBP - British Pound</option>
                      <option value="JPY">JPY - Japanese Yen</option>
                    </select>
                  </div>
                </div>

                <div class="grid grid-cols-3 gap-4 items-center">
                  <div class="col-span-1 text-sm text-gray-400 font-mono">Timezone</div>
                  <div class="col-span-2">
                    <select class="w-full bg-terminal-800 border border-terminal-700 rounded px-3 py-2 text-white font-mono focus:border-accent-500 outline-none">
                      <option value="UTC">UTC (Coordinated Universal Time)</option>
                      <option value="EST">EST (Eastern Standard Time)</option>
                      <option value="PST">PST (Pacific Standard Time)</option>
                      <option value="CET">CET (Central European Time)</option>
                    </select>
                  </div>
                </div>

                <div class="grid grid-cols-3 gap-4 items-center">
                  <div class="col-span-1 text-sm text-gray-400 font-mono">Theme</div>
                  <div class="col-span-2 flex gap-2">
                    <button class="flex-1 bg-terminal-800 border border-accent-500 text-accent-400 py-2 rounded text-xs font-bold font-mono">
                      Terminal Dark
                    </button>
                    <button class="flex-1 bg-terminal-800 border border-terminal-700 text-gray-400 py-2 rounded text-xs font-bold font-mono opacity-50 cursor-not-allowed">
                      Light (Coming Soon)
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </Show>

        </div>
      </div>
    </div>
  );
}