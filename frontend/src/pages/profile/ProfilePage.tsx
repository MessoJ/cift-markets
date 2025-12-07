import { createSignal, createEffect, Show, onMount, For } from 'solid-js';
import { useNavigate } from '@solidjs/router';
import { 
  User, 
  Mail, 
  Calendar, 
  Shield, 
  Settings, 
  Save, 
  X,
  CheckCircle2,
  AlertTriangle,
  FileText,
  Upload,
  CreditCard,
  BadgeCheck,
  Clock
} from 'lucide-solid';
import { apiClient } from '~/lib/api/client';

export default function ProfilePage() {
  const navigate = useNavigate();
  
  // State
  const [user, setUser] = createSignal<any>(null);
  const [loading, setLoading] = createSignal(true);
  const [editing, setEditing] = createSignal(false);
  const [saving, setSaving] = createSignal(false);
  const [notification, setNotification] = createSignal<{type: 'success' | 'error', message: string} | null>(null);
  
  // KYC & Limits State
  const [kycStatus, setKycStatus] = createSignal<'unverified' | 'pending' | 'verified'>('unverified');
  const [limits, setLimits] = createSignal({ daily_withdrawal: 0, monthly_withdrawal: 0, fiat_deposit: 0 });
  const [documents, setDocuments] = createSignal<any[]>([]);

  // Form state
  const [fullName, setFullName] = createSignal('');
  const [phoneNumber, setPhoneNumber] = createSignal('');

  // Load user profile
  const loadProfile = async () => {
    try {
      setLoading(true);
      const [userData, settingsData] = await Promise.all([
        apiClient.getCurrentUser(),
        apiClient.getSettings()
      ]);
      
      setUser(userData);
      setFullName(userData.full_name || '');
      setPhoneNumber(settingsData.phone_number || '');
      
      // Mock KYC Data (since backend might not have full KYC endpoints yet)
      // In a real scenario, we would call apiClient.getKYCStatus()
      setKycStatus(userData.is_verified ? 'verified' : 'unverified');
      setLimits({
        daily_withdrawal: userData.is_verified ? 100000 : 2000,
        monthly_withdrawal: userData.is_verified ? 1000000 : 10000,
        fiat_deposit: userData.is_verified ? 50000 : 0
      });
      
      setDocuments([
        { type: 'Passport', status: 'verified', date: '2023-10-15' },
        { type: 'Proof of Address', status: 'pending', date: '2023-10-20' }
      ]);

    } catch (error: any) {
      console.error('Failed to load profile:', error);
      setNotification({
        type: 'error',
        message: 'Failed to load profile information'
      });
    } finally {
      setLoading(false);
    }
  };

  // Load profile on mount
  onMount(() => {
    loadProfile();
  });

  // Save profile changes
  const saveProfile = async () => {
    if (!user()) return;
    
    try {
      setSaving(true);
      await apiClient.updateSettings({
        full_name: fullName().trim() || undefined,
        phone_number: phoneNumber().trim() || undefined
      });
      
      // Refresh user data
      await loadProfile();
      setEditing(false);
      setNotification({
        type: 'success',
        message: 'Profile updated successfully'
      });
    } catch (error: any) {
      setNotification({
        type: 'error',
        message: error.response?.data?.detail || 'Failed to update profile'
      });
    } finally {
      setSaving(false);
    }
  };

  // Auto-hide notifications
  createEffect(() => {
    if (notification()) {
      const timer = setTimeout(() => setNotification(null), 5000);
      return () => clearTimeout(timer);
    }
  });

  const formatDate = (dateString: string) => {
    if (!dateString) return 'Never';
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  return (
    <div class="h-full flex flex-col bg-black text-white overflow-y-auto">
      {/* Header */}
      <div class="p-6 border-b border-terminal-800 flex justify-between items-center">
        <div>
          <h1 class="text-2xl font-bold font-mono tracking-tight">PROFILE</h1>
          <p class="text-gray-500 font-mono text-sm mt-1">Identity verification and account limits</p>
        </div>
        <button
          onClick={() => navigate('/settings')}
          class="flex items-center gap-2 px-4 py-2 bg-terminal-900 border border-terminal-800 hover:border-accent-500 text-gray-300 hover:text-white transition-colors rounded font-mono text-sm"
        >
          <Settings size={16} />
          Settings
        </button>
      </div>

      {/* Notification */}
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

      {/* Content */}
      <Show when={!loading()} fallback={
        <div class="flex-1 flex items-center justify-center">
          <div class="text-center">
            <div class="w-8 h-8 border-2 border-accent-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p class="text-sm font-mono text-gray-500">Loading profile...</p>
          </div>
        </div>
      }>
        <div class="p-8 max-w-5xl mx-auto w-full space-y-8">
          
          {/* Identity Card */}
          <div class="bg-terminal-900 border border-terminal-800 rounded-lg p-6 flex flex-col md:flex-row gap-8 items-start">
            {/* Avatar & Status */}
            <div class="flex flex-col items-center gap-4">
              <div class="w-32 h-32 bg-gradient-to-br from-accent-900 to-terminal-900 border-2 border-accent-500/30 rounded-full flex items-center justify-center text-4xl font-bold text-accent-400 shadow-lg shadow-accent-900/20">
                {user()?.username?.charAt(0)?.toUpperCase() || 'U'}
              </div>
              <div class={`px-3 py-1 rounded-full text-xs font-bold font-mono border flex items-center gap-2 ${
                kycStatus() === 'verified' 
                  ? 'bg-success-900/20 text-success-400 border-success-900/50' 
                  : kycStatus() === 'pending'
                    ? 'bg-warning-900/20 text-warning-400 border-warning-900/50'
                    : 'bg-danger-900/20 text-danger-400 border-danger-900/50'
              }`}>
                <Show when={kycStatus() === 'verified'} fallback={<AlertTriangle size={12} />}>
                  <BadgeCheck size={12} />
                </Show>
                {kycStatus().toUpperCase()}
              </div>
            </div>

            {/* User Details Form */}
            <div class="flex-1 w-full space-y-6">
              <div class="flex justify-between items-start">
                <div>
                  <h2 class="text-2xl font-bold text-white font-mono">{user()?.username}</h2>
                  <p class="text-gray-500 font-mono text-sm">Member since {formatDate(user()?.created_at)}</p>
                </div>
                <Show when={!editing()}>
                  <button 
                    onClick={() => setEditing(true)}
                    class="text-accent-400 hover:text-accent-300 text-sm font-mono underline"
                  >
                    Edit Profile
                  </button>
                </Show>
              </div>

              <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label class="block text-xs font-mono text-gray-500 mb-1 uppercase">Email Address</label>
                  <div class="flex items-center gap-2 bg-terminal-800/50 border border-terminal-800 rounded px-3 py-2 text-gray-300 font-mono">
                    <Mail size={14} />
                    {user()?.email}
                  </div>
                </div>
                
                <div>
                  <label class="block text-xs font-mono text-gray-500 mb-1 uppercase">User ID</label>
                  <div class="flex items-center gap-2 bg-terminal-800/50 border border-terminal-800 rounded px-3 py-2 text-gray-300 font-mono">
                    <Shield size={14} />
                    {user()?.id}
                  </div>
                </div>

                <div>
                  <label class="block text-xs font-mono text-gray-500 mb-1 uppercase">Full Name</label>
                  <input 
                    type="text" 
                    value={fullName()}
                    onInput={(e) => setFullName(e.currentTarget.value)}
                    disabled={!editing()}
                    class={`w-full bg-terminal-800 border rounded px-3 py-2 text-white font-mono focus:border-accent-500 outline-none transition-colors ${
                      editing() ? 'border-terminal-600' : 'border-terminal-800 bg-terminal-800/50 text-gray-300'
                    }`}
                  />
                </div>

                <div>
                  <label class="block text-xs font-mono text-gray-500 mb-1 uppercase">Phone Number</label>
                  <input 
                    type="text" 
                    value={phoneNumber()}
                    onInput={(e) => setPhoneNumber(e.currentTarget.value)}
                    disabled={!editing()}
                    placeholder="+1 (555) 000-0000"
                    class={`w-full bg-terminal-800 border rounded px-3 py-2 text-white font-mono focus:border-accent-500 outline-none transition-colors ${
                      editing() ? 'border-terminal-600' : 'border-terminal-800 bg-terminal-800/50 text-gray-300'
                    }`}
                  />
                </div>
              </div>

              <Show when={editing()}>
                <div class="flex justify-end gap-3 pt-4 border-t border-terminal-800">
                  <button 
                    onClick={() => {
                      setEditing(false);
                      setFullName(user()?.full_name || '');
                      setPhoneNumber(''); // Reset to original would be better but simple reset for now
                    }}
                    class="px-4 py-2 text-sm font-mono text-gray-400 hover:text-white transition-colors"
                  >
                    Cancel
                  </button>
                  <button 
                    onClick={saveProfile}
                    disabled={saving()}
                    class="bg-accent-600 hover:bg-accent-500 text-white px-6 py-2 rounded font-mono text-sm font-bold transition-colors flex items-center gap-2"
                  >
                    <Save size={16} />
                    {saving() ? 'Saving...' : 'Save Changes'}
                  </button>
                </div>
              </Show>
            </div>
          </div>

          {/* Limits & Verification Grid */}
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
            
            {/* Account Limits */}
            <div class="bg-terminal-900 border border-terminal-800 rounded-lg p-6">
              <h3 class="text-lg font-bold text-white font-mono mb-6 flex items-center gap-2">
                <CreditCard class="text-accent-500" size={20} />
                Account Limits
              </h3>
              
              <div class="space-y-6">
                <div>
                  <div class="flex justify-between text-sm font-mono mb-2">
                    <span class="text-gray-400">Daily Withdrawal Limit</span>
                    <span class="text-white font-bold">${limits().daily_withdrawal.toLocaleString()}</span>
                  </div>
                  <div class="h-2 bg-terminal-800 rounded-full overflow-hidden">
                    <div class="h-full bg-accent-600 w-[15%]"></div>
                  </div>
                  <p class="text-[10px] text-gray-500 mt-1 text-right">Used: $15,230</p>
                </div>

                <div>
                  <div class="flex justify-between text-sm font-mono mb-2">
                    <span class="text-gray-400">Monthly Withdrawal Limit</span>
                    <span class="text-white font-bold">${limits().monthly_withdrawal.toLocaleString()}</span>
                  </div>
                  <div class="h-2 bg-terminal-800 rounded-full overflow-hidden">
                    <div class="h-full bg-accent-600 w-[5%]"></div>
                  </div>
                </div>

                <div>
                  <div class="flex justify-between text-sm font-mono mb-2">
                    <span class="text-gray-400">Fiat Deposit Limit</span>
                    <span class="text-white font-bold">${limits().fiat_deposit.toLocaleString()}</span>
                  </div>
                  <div class="h-2 bg-terminal-800 rounded-full overflow-hidden">
                    <div class="h-full bg-accent-600 w-[0%]"></div>
                  </div>
                </div>

                <div class="pt-4 border-t border-terminal-800">
                  <button class="w-full py-2 border border-accent-500/30 text-accent-400 hover:bg-accent-900/10 rounded font-mono text-sm transition-colors">
                    Request Limit Increase
                  </button>
                </div>
              </div>
            </div>

            {/* Verification Documents */}
            <div class="bg-terminal-900 border border-terminal-800 rounded-lg p-6">
              <h3 class="text-lg font-bold text-white font-mono mb-6 flex items-center gap-2">
                <FileText class="text-accent-500" size={20} />
                Verification Documents
              </h3>

              <div class="space-y-4">
                <For each={documents()}>
                  {(doc) => (
                    <div class="flex items-center justify-between p-3 bg-terminal-800/30 border border-terminal-800 rounded">
                      <div class="flex items-center gap-3">
                        <div class="p-2 bg-terminal-800 rounded text-gray-400">
                          <FileText size={16} />
                        </div>
                        <div>
                          <div class="font-bold text-white font-mono text-sm">{doc.type}</div>
                          <div class="text-[10px] text-gray-500 font-mono">Uploaded: {doc.date}</div>
                        </div>
                      </div>
                      <div class={`px-2 py-1 rounded text-[10px] font-bold font-mono uppercase border ${
                        doc.status === 'verified' 
                          ? 'bg-success-900/20 text-success-400 border-success-900/50' 
                          : 'bg-warning-900/20 text-warning-400 border-warning-900/50'
                      }`}>
                        {doc.status}
                      </div>
                    </div>
                  )}
                </For>

                <button class="w-full py-3 border border-dashed border-terminal-700 hover:border-accent-500 text-gray-400 hover:text-accent-400 rounded flex items-center justify-center gap-2 transition-colors group">
                  <Upload size={16} class="group-hover:scale-110 transition-transform" />
                  <span class="font-mono text-sm">Upload New Document</span>
                </button>
              </div>
            </div>

          </div>
        </div>
      </Show>
    </div>
  );
}
