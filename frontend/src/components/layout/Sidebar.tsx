/**
 * Professional Trading Platform Sidebar - v2.0
 * 
 * Industry-standard navigation inspired by TradingView & Bloomberg.
 * Features:
 * - Grouped collapsible sections
 * - Pinnable items with persistence
 * - Keyboard shortcuts display
 * - Badge notifications
 * - Tooltips in collapsed mode
 * - Quick trade action
 * - Recent navigation tracking
 */

import { A, useLocation } from '@solidjs/router';
import { createSignal, createEffect, For, Show, onMount } from 'solid-js';
import {
  LayoutDashboard,
  TrendingUp,
  Wallet,
  BarChart3,
  ListOrdered,
  Star,
  Receipt,
  Settings,
  LogOut,
  DollarSign,
  HelpCircle,
  BarChart2,
  Newspaper,
  FileText,
  Filter,
  Bell,
  Zap,
  ChevronDown,
  ChevronUp,
  ChevronLeft,
  ChevronRight,
  Pin,
  PieChart,
  CandlestickChart,
  Activity,
} from 'lucide-solid';
import { Logo } from './Logo';
import { AIIcon } from '~/components/icons/AIIcon';
import { authStore } from '~/stores/auth.store';
import { twMerge } from 'tailwind-merge';

// Navigation Section Types
interface NavItem {
  label: string;
  href: string;
  icon: any;
  shortcut?: string;
  badge?: number | string;
  isNew?: boolean;
}

interface NavSection {
  id: string;
  label: string;
  icon: any;
  items: NavItem[];
  defaultOpen?: boolean;
}

// Grouped Navigation Structure (Industry Standard)
const navSections: NavSection[] = [
  {
    id: 'core',
    label: 'Core',
    icon: Zap,
    defaultOpen: true,
    items: [
      { label: 'Dashboard', href: '/dashboard', icon: LayoutDashboard, shortcut: '⌘D' },
      { label: 'Trading', href: '/trading', icon: TrendingUp, shortcut: '⌘T' },
      { label: 'Portfolio', href: '/portfolio', icon: Wallet, shortcut: '⌘P' },
    ]
  },
  {
    id: 'markets',
    label: 'Markets',
    icon: Activity,
    defaultOpen: true,
    items: [
      { label: 'Analysis', href: '/analysis', icon: AIIcon, isNew: true },
      { label: 'Charts', href: '/charts', icon: CandlestickChart, shortcut: '⌘K' },
      { label: 'Screener', href: '/screener', icon: Filter },
      { label: 'Watchlists', href: '/watchlists', icon: Star },
      { label: 'News', href: '/news', icon: Newspaper },
    ]
  },
  {
    id: 'analytics',
    label: 'Analytics',
    icon: PieChart,
    defaultOpen: false,
    items: [
      { label: 'Performance', href: '/analytics', icon: BarChart3 },
      { label: 'Orders', href: '/orders', icon: ListOrdered },
      { label: 'Alerts', href: '/alerts', icon: Bell, badge: 3 },
    ]
  },
  {
    id: 'account',
    label: 'Account',
    icon: Wallet,
    defaultOpen: false,
    items: [
      { label: 'Transactions', href: '/transactions', icon: Receipt },
      { label: 'Funding', href: '/funding', icon: DollarSign },
      { label: 'Statements', href: '/statements', icon: FileText },
    ]
  },
];

interface SidebarProps {
  collapsed?: boolean;
  onToggleCollapse?: () => void;
  onMobileClose?: () => void;
}

export function Sidebar(props: SidebarProps) {
  const location = useLocation();
  
  // Section open/close state
  const [openSections, setOpenSections] = createSignal<Record<string, boolean>>({});
  
  // Pinned items (stored in localStorage)
  const [pinnedItems, setPinnedItems] = createSignal<string[]>([]);
  
  // Hover state for tooltips in collapsed mode
  const [hoveredItem, setHoveredItem] = createSignal<string | null>(null);

  onMount(() => {
    // Initialize section states based on current path
    const initialState: Record<string, boolean> = {};
    const currentPath = location.pathname;
    
    navSections.forEach(section => {
      // Check if any item in this section matches the current path
      const hasActiveItem = section.items.some(item => 
        currentPath === item.href || currentPath.startsWith(item.href + '/')
      );
      
      // Open if it has active item OR if it's default open
      initialState[section.id] = hasActiveItem || (section.defaultOpen ?? false);
    });
    setOpenSections(initialState);
    
    // Load pinned items from localStorage
    const stored = localStorage.getItem('cift_pinned_nav');
    if (stored) {
      try {
        setPinnedItems(JSON.parse(stored));
      } catch (e) {
        setPinnedItems([]);
      }
    }
  });

  const isActive = (href: string) => {
    if (href === '/dashboard') {
      return location.pathname === '/' || location.pathname === '/dashboard';
    }
    return location.pathname === href || location.pathname.startsWith(href + '/');
  };
  
  const toggleSection = (sectionId: string) => {
    setOpenSections(prev => ({
      ...prev,
      [sectionId]: !prev[sectionId]
    }));
  };
  
  const togglePin = (href: string, e: MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const current = pinnedItems();
    let updated: string[];
    if (current.includes(href)) {
      updated = current.filter(p => p !== href);
    } else {
      updated = [...current, href].slice(0, 5);
    }
    setPinnedItems(updated);
    localStorage.setItem('cift_pinned_nav', JSON.stringify(updated));
  };
  
  const isPinned = (href: string) => pinnedItems().includes(href);
  
  // Get all items flat for pinned lookup
  const allItems = navSections.flatMap(s => s.items);
  const getPinnedItemData = (href: string) => allItems.find(i => i.href === href);

  const handleLogout = async () => {
    await authStore.logout();
  };

  return (
    <aside
      class={twMerge(
        'flex flex-col h-full bg-terminal-950 border-r border-terminal-750 transition-all duration-300 ease-out',
        props.collapsed ? 'w-16' : 'w-60'
      )}
    >
      {/* Logo Header */}
      <div class="h-14 flex items-center px-4 border-b border-terminal-750/50">
        <Logo size="sm" variant={props.collapsed ? 'icon-only' : 'compact'} theme="dark" />
      </div>

      {/* Quick Trade Button */}
      <Show when={!props.collapsed}>
        <div class="px-3 py-3 border-b border-terminal-750/50">
          <A
            href="/trading"
            onClick={() => props.onMobileClose?.()}
            class="flex items-center justify-center gap-2 w-full px-4 py-2.5 bg-accent-500 hover:bg-accent-600 text-white rounded-lg font-medium text-sm transition-all duration-200 shadow-lg shadow-accent-500/20 hover:shadow-accent-500/30"
          >
            <Zap class="w-4 h-4" />
            <span>Quick Trade</span>
          </A>
        </div>
      </Show>
      <Show when={props.collapsed}>
        <div class="px-2 py-3 border-b border-terminal-750/50">
          <A
            href="/trading"
            onClick={() => props.onMobileClose?.()}
            class="flex items-center justify-center w-12 h-10 mx-auto bg-accent-500 hover:bg-accent-600 text-white rounded-lg transition-all duration-200"
            title="Quick Trade"
          >
            <Zap class="w-4 h-4" />
          </A>
        </div>
      </Show>

      {/* Pinned Items */}
      <Show when={pinnedItems().length > 0}>
        <div class={twMerge('border-b border-terminal-750/50', props.collapsed ? 'py-2' : 'px-3 py-2')}>
          <Show when={!props.collapsed}>
            <div class="flex items-center gap-2 px-2 mb-2">
              <Pin class="w-3 h-3 text-gray-500" />
              <span class="text-[10px] font-semibold uppercase tracking-wider text-gray-500">Pinned</span>
            </div>
          </Show>
          <For each={pinnedItems()}>
            {(href) => {
              const item = getPinnedItemData(href);
              if (!item) return null;
              return (
                <A
                  href={item.href}
                  onClick={() => props.onMobileClose?.()}
                  class={twMerge(
                    'flex items-center gap-3 py-3 md:py-2 rounded-md transition-all duration-200 group relative',
                    props.collapsed ? 'justify-center mx-2 px-2' : 'mx-1 px-3',
                    isActive(item.href)
                      ? 'bg-accent-500/15 text-accent-400'
                      : 'text-gray-400 hover:text-white hover:bg-terminal-800/50'
                  )}
                  onMouseEnter={() => props.collapsed && setHoveredItem(item.href)}
                  onMouseLeave={() => setHoveredItem(null)}
                >
                  <div class="relative">
                    <item.icon class="w-4 h-4 flex-shrink-0" />
                  </div>
                  <Show when={!props.collapsed}>
                    <span class="text-sm font-medium truncate flex-1">{item.label}</span>
                    <Pin class="w-3 h-3 text-accent-500 opacity-60" />
                  </Show>
                  {/* Active indicator */}
                  <Show when={isActive(item.href)}>
                    <div class="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 bg-accent-500 rounded-r-full" />
                  </Show>
                  {/* Tooltip for collapsed mode */}
                  <Show when={props.collapsed && hoveredItem() === item.href}>
                    <div class="absolute left-full ml-3 px-3 py-1.5 bg-terminal-800 text-white text-sm rounded-md shadow-xl border border-terminal-750 whitespace-nowrap z-50">
                      {item.label}
                    </div>
                  </Show>
                </A>
              );
            }}
          </For>
        </div>
      </Show>

      {/* Navigation Sections */}
      <nav class="flex-1 py-2 overflow-y-auto">
        <For each={navSections}>
          {(section) => (
            <div class="mb-1">
              {/* Section Header */}
              <Show when={!props.collapsed}>
                <button
                  onClick={() => toggleSection(section.id)}
                  class="w-full flex items-center justify-between px-4 py-2 text-gray-500 hover:text-gray-300 transition-colors"
                >
                  <div class="flex items-center gap-2">
                    <section.icon class="w-3.5 h-3.5" />
                    <span class="text-[11px] font-semibold uppercase tracking-wider">{section.label}</span>
                  </div>
                  {openSections()[section.id] 
                    ? <ChevronUp class="w-3.5 h-3.5" />
                    : <ChevronDown class="w-3.5 h-3.5" />
                  }
                </button>
              </Show>
              
              {/* Section Items */}
              <Show when={props.collapsed || openSections()[section.id]}>
                <ul class={twMerge('space-y-0.5', props.collapsed ? 'px-2' : 'px-2')}>
                  <For each={section.items}>
                    {(item) => (
                      <li class="relative group">
                        <A
                          href={item.href}
                          onClick={() => props.onMobileClose?.()}
                          class={twMerge(
                            'flex items-center gap-3 py-3 md:py-2 rounded-md transition-all duration-200 relative',
                            props.collapsed ? 'justify-center px-2' : 'px-3',
                            isActive(item.href)
                              ? 'bg-accent-500/15 text-accent-400'
                              : 'text-gray-400 hover:text-white hover:bg-terminal-800/50'
                          )}
                          onMouseEnter={() => props.collapsed && setHoveredItem(item.href)}
                          onMouseLeave={() => setHoveredItem(null)}
                        >
                          {/* Active indicator */}
                          <Show when={isActive(item.href)}>
                            <div class="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 bg-accent-500 rounded-r-full" />
                          </Show>
                          
                          <div class="relative">
                            <item.icon class="w-4 h-4 flex-shrink-0" />
                            {/* Badge */}
                            <Show when={item.badge}>
                              <span class="absolute -top-1.5 -right-1.5 min-w-[14px] h-[14px] flex items-center justify-center px-1 bg-accent-500 text-white text-[9px] font-bold rounded-full">
                                {item.badge}
                              </span>
                            </Show>
                          </div>
                          
                          <Show when={!props.collapsed}>
                            <span class="text-sm font-medium truncate flex-1">{item.label}</span>
                            
                            {/* New badge */}
                            <Show when={item.isNew}>
                              <span class="px-1.5 py-0.5 bg-success-500/20 text-success-400 text-[9px] font-bold uppercase rounded">
                                New
                              </span>
                            </Show>
                            
                            {/* Pin button */}
                            <button
                              onClick={(e) => togglePin(item.href, e)}
                              class={twMerge(
                                'p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity',
                                isPinned(item.href) ? 'text-accent-500' : 'text-gray-500 hover:text-white'
                              )}
                              title={isPinned(item.href) ? 'Unpin' : 'Pin to top'}
                            >
                              <Pin class="w-3 h-3" />
                            </button>
                            
                            {/* Keyboard shortcut */}
                            <Show when={item.shortcut}>
                              <span class="text-[10px] text-gray-600 font-mono">{item.shortcut}</span>
                            </Show>
                          </Show>
                          
                          {/* Tooltip for collapsed mode */}
                          <Show when={props.collapsed && hoveredItem() === item.href}>
                            <div class="absolute left-full ml-3 px-3 py-1.5 bg-terminal-800 text-white text-sm rounded-md shadow-xl border border-terminal-750 whitespace-nowrap z-50 flex items-center gap-2">
                              {item.label}
                              <Show when={item.badge}>
                                <span class="px-1.5 py-0.5 bg-accent-500 text-white text-[10px] font-bold rounded">
                                  {item.badge}
                                </span>
                              </Show>
                            </div>
                          </Show>
                        </A>
                      </li>
                    )}
                  </For>
                </ul>
              </Show>
            </div>
          )}
        </For>
      </nav>

      {/* Bottom Actions */}
      <div class="border-t border-terminal-750/50 py-2 space-y-0.5">
        {/* Support */}
        <A
          href="/support"
          onClick={() => props.onMobileClose?.()}
          class={twMerge(
            'flex items-center gap-3 py-3 md:py-2 rounded-md transition-all duration-200 relative',
            props.collapsed ? 'justify-center mx-2 px-2' : 'mx-2 px-3',
            isActive('/support')
              ? 'bg-accent-500/15 text-accent-400'
              : 'text-gray-400 hover:text-white hover:bg-terminal-800/50'
          )}
          onMouseEnter={() => props.collapsed && setHoveredItem('/support')}
          onMouseLeave={() => setHoveredItem(null)}
        >
          <Show when={isActive('/support')}>
            <div class="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 bg-accent-500 rounded-r-full" />
          </Show>
          <HelpCircle class="w-4 h-4 flex-shrink-0" />
          <Show when={!props.collapsed}>
            <span class="text-sm font-medium">Support</span>
          </Show>
          <Show when={props.collapsed && hoveredItem() === '/support'}>
            <div class="absolute left-full ml-3 px-3 py-1.5 bg-terminal-800 text-white text-sm rounded-md shadow-xl border border-terminal-750 whitespace-nowrap z-50">
              Support
            </div>
          </Show>
        </A>
        
        {/* Settings */}
        <A
          href="/settings"
          onClick={() => props.onMobileClose?.()}
          class={twMerge(
            'flex items-center gap-3 py-3 md:py-2 rounded-md transition-all duration-200 relative',
            props.collapsed ? 'justify-center mx-2 px-2' : 'mx-2 px-3',
            isActive('/settings')
              ? 'bg-accent-500/15 text-accent-400'
              : 'text-gray-400 hover:text-white hover:bg-terminal-800/50'
          )}
          onMouseEnter={() => props.collapsed && setHoveredItem('/settings')}
          onMouseLeave={() => setHoveredItem(null)}
        >
          <Show when={isActive('/settings')}>
            <div class="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 bg-accent-500 rounded-r-full" />
          </Show>
          <Settings class="w-4 h-4 flex-shrink-0" />
          <Show when={!props.collapsed}>
            <span class="text-sm font-medium">Settings</span>
          </Show>
          <Show when={props.collapsed && hoveredItem() === '/settings'}>
            <div class="absolute left-full ml-3 px-3 py-1.5 bg-terminal-800 text-white text-sm rounded-md shadow-xl border border-terminal-750 whitespace-nowrap z-50">
              Settings
            </div>
          </Show>
        </A>
        
        {/* Logout */}
        <button
          onClick={handleLogout}
          class={twMerge(
            'flex items-center gap-3 py-3 md:py-2 rounded-md text-gray-400 hover:text-danger-400 hover:bg-danger-500/10 transition-all duration-200',
            props.collapsed ? 'justify-center mx-2 px-2 w-12' : 'mx-2 px-3 w-[calc(100%-16px)]'
          )}
          aria-label="Logout"
          onMouseEnter={() => props.collapsed && setHoveredItem('/logout')}
          onMouseLeave={() => setHoveredItem(null)}
        >
          <LogOut class="w-4 h-4 flex-shrink-0" />
          <Show when={!props.collapsed}>
            <span class="text-sm font-medium">Logout</span>
          </Show>
          <Show when={props.collapsed && hoveredItem() === '/logout'}>
            <div class="absolute left-full ml-3 px-3 py-1.5 bg-terminal-800 text-white text-sm rounded-md shadow-xl border border-terminal-750 whitespace-nowrap z-50">
              Logout
            </div>
          </Show>
        </button>

        {/* User Badge */}
        <Show when={!props.collapsed && authStore.user()}>
          <div class="mt-2 pt-2 border-t border-terminal-750/50 px-4">
            <div class="flex items-center gap-3">
              <div class="w-8 h-8 rounded-full bg-accent-500/20 flex items-center justify-center text-accent-400 text-sm font-semibold">
                {authStore.user()?.username?.charAt(0).toUpperCase() || 'U'}
              </div>
              <div class="flex-1 min-w-0">
                <p class="text-sm font-medium text-gray-200 truncate">
                  {authStore.user()?.username}
                </p>
                <p class="text-[10px] text-gray-500 font-mono uppercase">
                  {authStore.user()?.is_superuser ? 'Administrator' : 'Trader'}
                </p>
              </div>
            </div>
          </div>
        </Show>
      </div>

      {/* Collapse Toggle */}
      <Show when={props.onToggleCollapse}>
        <button
          onClick={props.onToggleCollapse}
          class="h-11 flex items-center justify-center border-t border-terminal-750/50 text-gray-500 hover:text-white hover:bg-terminal-800/50 transition-all duration-200 group"
          title={props.collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          aria-label={props.collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          <Show when={props.collapsed} fallback={
            <div class="flex items-center gap-2">
              <ChevronLeft class="w-4 h-4" />
              <span class="text-xs opacity-0 group-hover:opacity-100 transition-opacity">Collapse</span>
            </div>
          }>
            <ChevronRight class="w-4 h-4" />
          </Show>
        </button>
      </Show>
    </aside>
  );
}
