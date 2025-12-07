/**
 * Professional Trading Platform Main Layout
 * 
 * Institutional-grade layout with:
 * - Terminal black background
 * - Compact spacing
 * - Professional styling
 * - Maximum content area
 * - Mobile responsive with drawer navigation
 */

import { JSX, createSignal, createEffect, onCleanup, Show } from 'solid-js';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { StatusBar } from './StatusBar';

interface MainLayoutProps {
  children: JSX.Element;
}

export function MainLayout(props: MainLayoutProps) {
  const [sidebarCollapsed, setSidebarCollapsed] = createSignal(false);
  const [mobileMenuOpen, setMobileMenuOpen] = createSignal(false);
  
  // Close mobile menu on route change
  createEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 768) {
        setMobileMenuOpen(false);
      }
    };
    
    window.addEventListener('resize', handleResize);
    onCleanup(() => window.removeEventListener('resize', handleResize));
  });

  return (
    <div class="flex h-screen bg-terminal-950 text-gray-300 overflow-hidden">
      {/* Mobile Menu Overlay */}
      <Show when={mobileMenuOpen()}>
        <div 
          class="fixed inset-0 bg-black/60 z-40 md:hidden backdrop-blur-sm"
          onClick={() => setMobileMenuOpen(false)}
        />
      </Show>
      
      {/* Sidebar - Hidden on mobile by default, drawer on open */}
      <div class={`
        fixed md:static inset-y-0 left-0 z-50 transform transition-transform duration-300 ease-in-out
        md:translate-x-0 shadow-2xl md:shadow-none
        ${mobileMenuOpen() ? 'translate-x-0' : '-translate-x-full'}
      `}>
        <Sidebar
          collapsed={sidebarCollapsed()}
          onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed())}
          onMobileClose={() => setMobileMenuOpen(false)}
        />
      </div>

      {/* Main Content Area */}
      <div class="flex-1 flex flex-col min-w-0 bg-terminal-950">
        {/* Header - Compact */}
        <Header onMobileMenuToggle={() => setMobileMenuOpen(!mobileMenuOpen())} />

        {/* Page Content - Dense, Professional, Mobile Responsive */}
        <main class="flex-1 overflow-y-auto p-2 sm:p-3 md:p-4 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-terminal-750">
          {props.children}
        </main>

        {/* Status Bar - Fixed at bottom */}
        <StatusBar />
      </div>
    </div>
  );
}
