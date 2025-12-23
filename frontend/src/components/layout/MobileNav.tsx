import { A, useLocation } from '@solidjs/router';
import { LayoutDashboard, TrendingUp, Wallet, Menu } from 'lucide-solid';

interface MobileNavProps {
  onMenuClick: () => void;
}

export function MobileNav(props: MobileNavProps) {
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path || location.pathname.startsWith(path + '/');

  return (
    <div class="md:hidden fixed bottom-0 left-0 right-0 bg-terminal-900 border-t border-terminal-800 z-50 pb-safe">
      <div class="flex justify-around items-center h-16">
        <A href="/dashboard" class={`flex flex-col items-center gap-1 p-2 transition-colors ${isActive('/dashboard') ? 'text-accent-500' : 'text-gray-500 hover:text-gray-300'}`}>
          <LayoutDashboard class="w-5 h-5" />
          <span class="text-[10px] font-medium">Home</span>
        </A>
        <A href="/trading" class={`flex flex-col items-center gap-1 p-2 transition-colors ${isActive('/trading') ? 'text-accent-500' : 'text-gray-500 hover:text-gray-300'}`}>
          <TrendingUp class="w-5 h-5" />
          <span class="text-[10px] font-medium">Trade</span>
        </A>
        <A href="/portfolio" class={`flex flex-col items-center gap-1 p-2 transition-colors ${isActive('/portfolio') ? 'text-accent-500' : 'text-gray-500 hover:text-gray-300'}`}>
          <Wallet class="w-5 h-5" />
          <span class="text-[10px] font-medium">Portfolio</span>
        </A>
        <button 
          onClick={props.onMenuClick}
          class="flex flex-col items-center gap-1 p-2 text-gray-500 hover:text-gray-300 transition-colors"
        >
          <Menu class="w-5 h-5" />
          <span class="text-[10px] font-medium">Menu</span>
        </button>
      </div>
    </div>
  );
}
