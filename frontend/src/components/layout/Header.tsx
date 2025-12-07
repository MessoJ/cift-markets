/**
 * Professional Trading Platform Header - v3.0
 * 
 * Institutional-grade Command Center.
 * Features:
 * - Real-time Market Session & Latency
 * - Global Command Center (Search)
 * - Portfolio Summary (Privacy Mode)
 * - Breadcrumb Navigation
 * - Quick Actions
 * - High-density data display
 */

import { createSignal, createEffect, Show, onMount, onCleanup, For } from 'solid-js';
import { useNavigate, useLocation, A } from '@solidjs/router';
import { 
  Bell, 
  ChevronDown, 
  User, 
  Settings, 
  LogOut, 
  Menu, 
  Plus,
  Wallet,
  Eye,
  EyeOff,
  Activity,
  ChevronRight,
  HelpCircle,
  CreditCard,
  FileText,
  Search,
  X
} from 'lucide-solid';
import { MarketIndicesBar } from '~/components/market/MarketIndicesBar';
import { GlobalSearch } from '~/components/search/GlobalSearch';
import { apiClient, type Notification, type PortfolioSummary } from '~/lib/api/client';
import { authStore } from '~/stores/auth.store';

interface HeaderProps {
  onMobileMenuToggle?: () => void;
}

export function Header(props: HeaderProps) {
  const navigate = useNavigate();
  const location = useLocation();
  
  // State
  const [notifications, setNotifications] = createSignal<Notification[]>([]);
  const [unreadCount, setUnreadCount] = createSignal(0);
  const [marketSession, setMarketSession] = createSignal('REGULAR');
  const [portfolio, setPortfolio] = createSignal<PortfolioSummary | null>(null);
  const [showBalance, setShowBalance] = createSignal(false);
  const [showMobileSearch, setShowMobileSearch] = createSignal(false);
  
  // Dropdown states
  const [showNotifications, setShowNotifications] = createSignal(false);
  const [showProfile, setShowProfile] = createSignal(false);
  const [showQuickActions, setShowQuickActions] = createSignal(false);

  // Load Data
  createEffect(async () => {
    if (authStore.user()) {
      await Promise.all([
        loadNotifications(),
        loadPortfolio()
      ]);
    }
  });
  
  const loadNotifications = async () => {
    try {
      const [notificationsData, countData] = await Promise.all([
        apiClient.getNotifications(10, false),
        apiClient.getUnreadCount()
      ]);
      setNotifications(notificationsData);
      setUnreadCount(countData.count);
    } catch (error) {
      console.warn('Failed to load notifications:', error);
    }
  };

  const loadPortfolio = async () => {
    try {
      const data = await apiClient.getPortfolioSummary();
      setPortfolio(data);
    } catch (error) {
      console.warn('Failed to load portfolio:', error);
    }
  };

  // Clock & Session Logic
  onMount(() => {
    const timer = setInterval(() => {
      const now = new Date();
      
      // Session Logic (EST)
      const est = new Date(now.toLocaleString("en-US", {timeZone: "America/New_York"}));
      const hour = est.getHours();
      const min = est.getMinutes();
      const day = est.getDay();
      
      if (day === 0 || day === 6) setMarketSession('CLOSED');
      else if (hour < 4) setMarketSession('CLOSED');
      else if (hour < 9 || (hour === 9 && min < 30)) setMarketSession('PRE-MKT');
      else if (hour < 16) setMarketSession('REGULAR');
      else if (hour < 20) setMarketSession('POST-MKT');
      else setMarketSession('CLOSED');
    }, 1000);

    return () => clearInterval(timer);
  });

  // Click Outside Handler
  const handleClickOutside = (event: MouseEvent) => {
    const target = event.target as Element;
    if (!target.closest('.dropdown-container')) {
      setShowNotifications(false);
      setShowProfile(false);
      setShowQuickActions(false);
    }
  };

  onMount(() => document.addEventListener('click', handleClickOutside));
  onCleanup(() => document.removeEventListener('click', handleClickOutside));

  // Helpers
  const formatCurrency = (val: number) => {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val);
  };

  const getBreadcrumbs = () => {
    const path = location.pathname.split('/').filter(Boolean);
    return path.map((segment, index) => ({
      label: segment.charAt(0).toUpperCase() + segment.slice(1),
      path: '/' + path.slice(0, index + 1).join('/')
    }));
  };

  const handleNotificationClick = async (notification: Notification) => {
    if (!notification.is_read) {
      try {
        await apiClient.markNotificationRead(notification.id);
        await loadNotifications();
      } catch (error) {
        console.warn('Failed to mark notification as read:', error);
      }
    }
    if (notification.link) {
      navigate(notification.link);
      setShowNotifications(false);
    }
  };

  const handleMarkAllRead = async () => {
    try {
      await apiClient.markAllNotificationsRead();
      await loadNotifications();
    } catch (error) {
      console.warn('Failed to mark all notifications as read:', error);
    }
  };

  return (
    <header class="h-16 bg-terminal-950 border-b border-terminal-800 flex items-center justify-between px-4 shadow-lg z-40 relative">
      {/* Mobile Search Overlay */}
      <Show when={showMobileSearch()}>
        <div class="absolute inset-0 bg-terminal-950 z-50 flex items-center px-4 gap-2 animate-in fade-in slide-in-from-top-2">
          <div class="flex-1">
            <GlobalSearch />
          </div>
          <button 
            onClick={() => setShowMobileSearch(false)} 
            class="p-2 text-gray-400 hover:text-white"
            aria-label="Close search"
          >
            <X size={20} />
          </button>
        </div>
      </Show>

      {/* Left: Mobile Menu, Logo, Breadcrumbs */}
      <div class="flex items-center gap-4 flex-1">
        <button
          onClick={() => props.onMobileMenuToggle?.()}
          class="lg:hidden p-2 text-gray-400 hover:text-white hover:bg-terminal-900 rounded-md transition-colors"
          aria-label="Toggle mobile menu"
        >
          <Menu size={20} />
        </button>

        {/* Breadcrumbs (Desktop) */}
        <div class="hidden md:flex items-center gap-2 text-sm text-gray-400">
          <A href="/dashboard" class="hover:text-white transition-colors">
            <Activity size={16} />
          </A>
          <For each={getBreadcrumbs()}>
            {(crumb) => (
              <>
                <ChevronRight size={14} class="text-terminal-600" />
                <A 
                  href={crumb.path}
                  class="hover:text-white transition-colors font-medium"
                  classList={{ 'text-white': location.pathname === crumb.path }}
                >
                  {crumb.label}
                </A>
              </>
            )}
          </For>
        </div>
      </div>

      {/* Center: Global Search & Indices */}
      <div class="hidden lg:flex flex-col items-center gap-1 flex-1 max-w-2xl">
        <div class="w-full max-w-md">
          <GlobalSearch />
        </div>
        <MarketIndicesBar />
      </div>

      {/* Right: Actions & Profile */}
      <div class="flex items-center gap-3 flex-1 justify-end">
        
        {/* Mobile Search Toggle */}
        <button 
          onClick={() => setShowMobileSearch(true)}
          class="lg:hidden p-2 text-gray-400 hover:text-white hover:bg-terminal-900 rounded-lg transition-colors"
          aria-label="Open search"
        >
          <Search size={20} />
        </button>

        {/* Quick Actions */}
        <div class="relative dropdown-container hidden md:block">
          <button 
            onClick={(e) => { e.stopPropagation(); setShowQuickActions(!showQuickActions()); setShowNotifications(false); setShowProfile(false); }}
            class="flex items-center gap-2 px-3 py-1.5 bg-accent-600 hover:bg-accent-500 text-white rounded text-xs font-bold transition-colors"
          >
            <Plus size={14} />
            <span>NEW</span>
          </button>
          
          <Show when={showQuickActions()}>
            <div class="absolute right-0 top-full mt-2 w-48 bg-terminal-900 border border-terminal-700 rounded-lg shadow-xl py-1 animate-in fade-in slide-in-from-top-2">
              <button onClick={() => navigate('/trading')} class="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-terminal-800 hover:text-white flex items-center gap-2">
                <Activity size={14} /> New Order
              </button>
              <button onClick={() => navigate('/funding')} class="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-terminal-800 hover:text-white flex items-center gap-2">
                <CreditCard size={14} /> Deposit Funds
              </button>
              <button onClick={() => navigate('/support')} class="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-terminal-800 hover:text-white flex items-center gap-2">
                <HelpCircle size={14} /> Create Ticket
              </button>
            </div>
          </Show>
        </div>

        {/* Wallet Balance */}
        <div class="hidden xl:flex items-center gap-3 px-3 py-1.5 bg-terminal-900 border border-terminal-800 rounded-lg">
          <div class="flex items-center gap-2">
            <Wallet size={14} class="text-gray-400" />
            <span class="text-xs font-mono font-bold text-white">
              {showBalance() ? formatCurrency(portfolio()?.total_value || 0) : '••••••'}
            </span>
          </div>
          <button 
            onClick={() => setShowBalance(!showBalance())}
            class="text-gray-500 hover:text-white transition-colors"
          >
            {showBalance() ? <EyeOff size={14} /> : <Eye size={14} />}
          </button>
        </div>

        {/* Notifications */}
        <div class="relative dropdown-container">
          <button 
            onClick={(e) => { e.stopPropagation(); setShowNotifications(!showNotifications()); setShowProfile(false); setShowQuickActions(false); }}
            class="p-2 text-gray-400 hover:text-white hover:bg-terminal-900 rounded-lg transition-colors relative"
          >
            <Bell size={20} />
            <Show when={unreadCount() > 0}>
              <span class="absolute top-1.5 right-1.5 w-2 h-2 bg-accent-500 rounded-full animate-pulse" />
            </Show>
          </button>

          <Show when={showNotifications()}>
            <div class="absolute right-0 top-full mt-2 w-80 bg-terminal-900 border border-terminal-700 rounded-lg shadow-xl overflow-hidden animate-in fade-in slide-in-from-top-2">
              <div class="p-3 border-b border-terminal-800 flex justify-between items-center">
                <h3 class="text-sm font-bold text-white">Notifications</h3>
                <button onClick={handleMarkAllRead} class="text-xs text-accent-400 hover:text-accent-300">Mark all read</button>
              </div>
              <div class="max-h-64 overflow-y-auto">
                <For each={notifications()} fallback={
                  <div class="p-8 text-center text-gray-500 text-xs">No new notifications</div>
                }>
                  {(notification) => (
                    <div 
                      onClick={() => handleNotificationClick(notification)}
                      class="p-3 border-b border-terminal-800 hover:bg-terminal-800 transition-colors cursor-pointer"
                    >
                      <div class="flex justify-between items-start mb-1">
                        <span class="font-bold text-xs text-white">{notification.title}</span>
                        <span class="text-[10px] text-gray-500">{new Date(notification.created_at).toLocaleTimeString()}</span>
                      </div>
                      <p class="text-xs text-gray-400 line-clamp-2">{notification.message}</p>
                    </div>
                  )}
                </For>
              </div>
            </div>
          </Show>
        </div>

        {/* Profile */}
        <div class="relative dropdown-container pl-2 border-l border-terminal-800">
          <button 
            onClick={(e) => { e.stopPropagation(); setShowProfile(!showProfile()); setShowNotifications(false); setShowQuickActions(false); }}
            class="flex items-center gap-2 hover:bg-terminal-900 p-1.5 rounded-lg transition-colors"
          >
            <div class="w-8 h-8 bg-terminal-800 rounded flex items-center justify-center text-accent-500 font-bold border border-terminal-700">
              {authStore.user()?.username.charAt(0).toUpperCase()}
            </div>
            <div class="hidden md:block text-left">
              <div class="text-xs font-bold text-white leading-none mb-1">{authStore.user()?.username}</div>
              <div class="text-[10px] text-gray-500 font-mono leading-none flex items-center gap-1">
                <span class={`w-1.5 h-1.5 rounded-full ${marketSession() === 'REGULAR' ? 'bg-success-500' : 'bg-warning-500'}`} />
                {marketSession()}
              </div>
            </div>
            <ChevronDown size={14} class="text-gray-500" />
          </button>

          <Show when={showProfile()}>
            <div class="absolute right-0 top-full mt-2 w-56 bg-terminal-900 border border-terminal-700 rounded-lg shadow-xl py-1 animate-in fade-in slide-in-from-top-2">
              <div class="px-4 py-3 border-b border-terminal-800">
                <p class="text-sm font-bold text-white">{authStore.user()?.full_name || authStore.user()?.username}</p>
                <p class="text-xs text-gray-500 truncate">{authStore.user()?.email}</p>
              </div>
              
              <div class="py-1">
                <button onClick={() => navigate('/profile')} class="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-terminal-800 hover:text-white flex items-center gap-2">
                  <User size={14} /> Profile & KYC
                </button>
                <button onClick={() => navigate('/settings')} class="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-terminal-800 hover:text-white flex items-center gap-2">
                  <Settings size={14} /> Settings
                </button>
                <button onClick={() => navigate('/statements')} class="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-terminal-800 hover:text-white flex items-center gap-2">
                  <FileText size={14} /> Statements
                </button>
              </div>

              <div class="border-t border-terminal-800 py-1">
                <button onClick={() => authStore.logout()} class="w-full text-left px-4 py-2 text-sm text-danger-400 hover:bg-terminal-800 hover:text-danger-300 flex items-center gap-2">
                  <LogOut size={14} /> Sign Out
                </button>
              </div>
            </div>
          </Show>
        </div>
      </div>
    </header>
  );
}
