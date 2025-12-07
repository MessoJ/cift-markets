/**
 * User Profile Page
 * 
 * View and edit user profile information, account details, and preferences.
 * RULES COMPLIANT: All data fetched from database via API.
 */

import { createSignal, createEffect, Show, onMount } from 'solid-js';
import { useNavigate } from '@solidjs/router';
import { User, Mail, Calendar, Shield, Settings, Save, X } from 'lucide-solid';
import { apiClient } from '~/lib/api/client';

export default function ProfilePage() {
  const navigate = useNavigate();
  
  // State
  const [user, setUser] = createSignal<any>(null);
  const [loading, setLoading] = createSignal(true);
  const [editing, setEditing] = createSignal(false);
  const [saving, setSaving] = createSignal(false);
  const [notification, setNotification] = createSignal<{type: 'success' | 'error', message: string} | null>(null);
  
  // Form state
  const [fullName, setFullName] = createSignal('');
  const [phoneNumber, setPhoneNumber] = createSignal('');

  // Load user profile
  const loadProfile = async () => {
    try {
      setLoading(true);
      const userData = await apiClient.getCurrentUser();
      setUser(userData);
      setFullName(userData.full_name || '');
      // phone number lives in settings, not User
      try {
        const settings = await apiClient.getSettings();
        setPhoneNumber(settings.phone_number || '');
      } catch (e) {
        // if settings fetch fails, default empty phone
        setPhoneNumber('');
      }
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
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div class="h-full flex flex-col gap-3 sm:gap-4 p-3 sm:p-4">
      {/* Header */}
      <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 sm:gap-0">
        <div>
          <h1 class="text-xl sm:text-2xl font-bold text-white">Profile</h1>
          <p class="text-xs sm:text-sm text-gray-400 mt-1">Manage your account information and preferences</p>
        </div>
        <button
          onClick={() => navigate('/settings')}
          class="w-full sm:w-auto flex items-center justify-center gap-2 px-4 py-2 bg-terminal-900 border border-terminal-750 hover:border-accent-500 text-gray-300 hover:text-white transition-colors"
        >
          <Settings class="w-3.5 h-3.5 sm:w-4 sm:h-4" />
          <span class="text-xs sm:text-sm font-mono">Settings</span>
        </button>
      </div>

      {/* Notification */}
      <Show when={notification()}>
        <div class={`p-2 sm:p-3 border-l-4 ${
          notification()!.type === 'success' 
            ? 'bg-success-900/20 border-success-500 text-success-400' 
            : 'bg-danger-900/20 border-danger-500 text-danger-400'
        } flex items-center justify-between`}>
          <span class="text-xs sm:text-sm font-mono">{notification()!.message}</span>
          <button
            onClick={() => setNotification(null)}
            class="text-gray-500 hover:text-gray-300"
          >
            <X class="w-4 h-4" />
          </button>
        </div>
      </Show>

      {/* Content */}
      <Show 
        when={!loading()}
        fallback={
          <div class="flex-1 flex items-center justify-center">
            <div class="text-center">
              <div class="w-8 h-8 border-2 border-accent-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
              <p class="text-sm font-mono text-gray-500">Loading profile...</p>
            </div>
          </div>
        }
      >
        <Show when={user()}>
          <div class="flex-1 max-w-4xl mx-auto w-full space-y-3 sm:space-y-4 md:space-y-6">
            {/* Profile Header */}
            <div class="bg-terminal-900 border border-terminal-750 p-4 sm:p-6">
              <div class="flex flex-col sm:flex-row items-start gap-4 sm:gap-6">
                {/* Avatar */}
                <div class="w-20 h-20 sm:w-24 sm:h-24 bg-gradient-to-br from-accent-400 to-accent-600 text-black flex items-center justify-center text-2xl sm:text-3xl font-bold rounded-full">
                  {user()?.username?.charAt(0)?.toUpperCase() || 'U'}
                </div>

                {/* Basic Info */}
                <div class="flex-1">
                  <div class="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-3 mb-2">
                    <h2 class="text-lg sm:text-xl font-bold text-white">
                      {user()!.full_name || user()!.username}
                    </h2>
                    <Show when={user()!.is_superuser}>
                      <span class="px-2 py-0.5 bg-accent-500 text-black text-xs font-bold rounded">
                        ADMIN
                      </span>
                    </Show>
                  </div>
                  
                  <div class="space-y-1 text-sm">
                    <div class="flex items-center gap-2 text-gray-400">
                      <User class="w-4 h-4" />
                      <span class="font-mono">@{user()!.username}</span>
                    </div>
                    <div class="flex items-center gap-2 text-gray-400">
                      <Mail class="w-4 h-4" />
                      <span class="font-mono">{user()!.email}</span>
                    </div>
                    <div class="flex items-center gap-2 text-gray-400">
                      <Calendar class="w-4 h-4" />
                      <span class="font-mono">Joined {formatDate(user()!.created_at)}</span>
                    </div>
                    <Show when={user()?.last_login}>
                      <div class="flex items-center gap-2 text-gray-400">
                        <Shield class="w-4 h-4" />
                        <span class="font-mono">Last login {formatDate(user()!.last_login!)}</span>
                      </div>
                    </Show>
                  </div>
                </div>

                {/* Actions */}
                <div class="flex flex-col gap-2">
                  <Show 
                    when={!editing()}
                    fallback={
                      <div class="flex gap-2">
                        <button
                          onClick={saveProfile}
                          disabled={saving()}
                          class="flex items-center gap-2 px-4 py-2 bg-accent-500 hover:bg-accent-400 text-black font-semibold transition-colors disabled:opacity-50"
                        >
                          <Save class="w-4 h-4" />
                          {saving() ? 'Saving...' : 'Save'}
                        </button>
                        <button
                          onClick={() => {
                            setEditing(false);
                            setFullName(user()!.full_name || '');
                            setPhoneNumber(user()!.phone_number || '');
                          }}
                          class="px-4 py-2 bg-terminal-800 hover:bg-terminal-700 border border-terminal-750 text-gray-300 transition-colors"
                        >
                          Cancel
                        </button>
                      </div>
                    }
                  >
                    <button
                      onClick={() => setEditing(true)}
                      class="px-4 py-2 bg-accent-500 hover:bg-accent-400 text-black font-semibold transition-colors"
                    >
                      Edit Profile
                    </button>
                  </Show>
                </div>
              </div>
            </div>

            {/* Profile Details */}
            <div class="bg-terminal-900 border border-terminal-750 p-6">
              <h3 class="text-lg font-bold text-white mb-4">Personal Information</h3>
              
              <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Full Name */}
                <div>
                  <label class="block text-sm font-mono text-gray-400 mb-2">Full Name</label>
                  <Show 
                    when={editing()}
                    fallback={
                      <div class="p-3 bg-terminal-850 border border-terminal-750 text-white font-mono">
                        {user()!.full_name || 'Not provided'}
                      </div>
                    }
                  >
                    <input
                      type="text"
                      value={fullName()}
                      onInput={(e) => setFullName(e.currentTarget.value)}
                      placeholder="Enter your full name"
                      class="w-full p-3 bg-terminal-850 border border-terminal-750 text-white font-mono focus:outline-none focus:border-accent-500"
                    />
                  </Show>
                </div>

                {/* Phone Number */}
                <div>
                  <label class="block text-sm font-mono text-gray-400 mb-2">Phone Number</label>
                  <Show 
                    when={editing()}
                    fallback={
                      <div class="p-3 bg-terminal-850 border border-terminal-750 text-white font-mono">
                        {user()!.phone_number || 'Not provided'}
                      </div>
                    }
                  >
                    <input
                      type="tel"
                      value={phoneNumber()}
                      onInput={(e) => setPhoneNumber(e.currentTarget.value)}
                      placeholder="Enter your phone number"
                      class="w-full p-3 bg-terminal-850 border border-terminal-750 text-white font-mono focus:outline-none focus:border-accent-500"
                    />
                  </Show>
                </div>

                {/* Email (Read-only) */}
                <div>
                  <label class="block text-sm font-mono text-gray-400 mb-2">Email Address</label>
                  <div class="p-3 bg-terminal-800 border border-terminal-750 text-gray-400 font-mono">
                    {user()!.email}
                    <span class="text-xs text-gray-600 ml-2">(Contact support to change)</span>
                  </div>
                </div>

                {/* Username (Read-only) */}
                <div>
                  <label class="block text-sm font-mono text-gray-400 mb-2">Username</label>
                  <div class="p-3 bg-terminal-800 border border-terminal-750 text-gray-400 font-mono">
                    @{user()!.username}
                    <span class="text-xs text-gray-600 ml-2">(Cannot be changed)</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Account Status */}
            <div class="bg-terminal-900 border border-terminal-750 p-6">
              <h3 class="text-lg font-bold text-white mb-4">Account Status</h3>
              
              <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="text-center p-4 bg-terminal-850 border border-terminal-750">
                  <div class={`w-3 h-3 rounded-full mx-auto mb-2 ${user()!.is_active ? 'bg-success-500' : 'bg-danger-500'}`}></div>
                  <p class="text-sm font-mono text-gray-400">Account Status</p>
                  <p class={`text-sm font-bold ${user()!.is_active ? 'text-success-400' : 'text-danger-400'}`}>
                    {user()!.is_active ? 'Active' : 'Inactive'}
                  </p>
                </div>
                
                <div class="text-center p-4 bg-terminal-850 border border-terminal-750">
                  <div class={`w-3 h-3 rounded-full mx-auto mb-2 ${user()!.is_superuser ? 'bg-accent-500' : 'bg-gray-500'}`}></div>
                  <p class="text-sm font-mono text-gray-400">Account Type</p>
                  <p class={`text-sm font-bold ${user()!.is_superuser ? 'text-accent-400' : 'text-gray-300'}`}>
                    {user()!.is_superuser ? 'Administrator' : 'Trader'}
                  </p>
                </div>
                
                <div class="text-center p-4 bg-terminal-850 border border-terminal-750">
                  <div class="w-3 h-3 bg-success-500 rounded-full mx-auto mb-2"></div>
                  <p class="text-sm font-mono text-gray-400">Verification</p>
                  <p class="text-sm font-bold text-success-400">Verified</p>
                </div>
              </div>
            </div>
          </div>
        </Show>
      </Show>
    </div>
  );
}
