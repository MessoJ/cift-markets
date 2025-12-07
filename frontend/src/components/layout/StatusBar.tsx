import { createSignal, onMount, onCleanup, Show } from 'solid-js';
import { Wifi, WifiOff, Database, Activity, Server, GitBranch, Clock } from 'lucide-solid';

export function StatusBar() {
  const [latency, setLatency] = createSignal<number>(0);
  const [isConnected, setIsConnected] = createSignal(true);
  const [dbStatus, setDbStatus] = createSignal<'connected' | 'disconnected'>('connected');
  const [systemTime, setSystemTime] = createSignal(new Date());

  // Simulate latency updates
  onMount(() => {
    const interval = setInterval(() => {
      setLatency(Math.floor(Math.random() * 40) + 10); // 10-50ms
      setSystemTime(new Date());
    }, 1000);

    onCleanup(() => clearInterval(interval));
  });

  return (
    <footer class="h-6 bg-terminal-950 border-t border-terminal-750 flex items-center justify-between px-3 text-[10px] font-mono text-gray-500 select-none">
      {/* Left: System Status */}
      <div class="flex items-center gap-4">
        <div class="flex items-center gap-1.5" title="WebSocket Connection">
          <Show when={isConnected()} fallback={<WifiOff class="w-3 h-3 text-danger-500" />}>
            <Wifi class="w-3 h-3 text-success-500" />
          </Show>
          <span class={isConnected() ? 'text-gray-400' : 'text-danger-500'}>
            {isConnected() ? 'CNTD' : 'DISC'}
          </span>
        </div>

        <div class="flex items-center gap-1.5" title="Network Latency">
          <Activity class="w-3 h-3 text-gray-600" />
          <span class={latency() > 100 ? 'text-warning-500' : 'text-gray-400'}>
            {latency()}ms
          </span>
        </div>

        <div class="flex items-center gap-1.5" title="Database Connection">
          <Database class="w-3 h-3 text-gray-600" />
          <span class={dbStatus() === 'connected' ? 'text-gray-400' : 'text-danger-500'}>
            {dbStatus() === 'connected' ? 'DB: OK' : 'DB: ERR'}
          </span>
        </div>
      </div>

      {/* Center: System Messages / Ticker (Placeholder) */}
      <div class="hidden md:flex items-center gap-2 opacity-50">
        <span>CIFT MARKETS TERMINAL v2.4.0</span>
      </div>

      {/* Right: Server & Time */}
      <div class="flex items-center gap-4">
        <div class="flex items-center gap-1.5" title="Server Region">
          <Server class="w-3 h-3 text-gray-600" />
          <span>US-EAST-1</span>
        </div>
        
        <div class="flex items-center gap-1.5" title="Build Version">
          <GitBranch class="w-3 h-3 text-gray-600" />
          <span>v2.4.0-stable</span>
        </div>

        <div class="flex items-center gap-1.5 pl-2 border-l border-terminal-800">
          <Clock class="w-3 h-3 text-gray-600" />
          <span>{systemTime().toISOString().split('T')[1].split('.')[0]} UTC</span>
        </div>
      </div>
    </footer>
  );
}
