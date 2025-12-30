import { createSignal, onMount, onCleanup, Show } from 'solid-js';
import { Play, Square, Activity, RefreshCw, Cpu, Zap, AlertTriangle } from 'lucide-solid';
import { apiClient } from '~/lib/api/client';

interface SystemStatus {
  status: string;
  pipeline_running: boolean;
  active_symbols: string[];
  total_predictions: number;
  uptime_seconds: number;
}

export default function HFTControlPanel() {
  const [status, setStatus] = createSignal<SystemStatus | null>(null);
  const [loading, setLoading] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [symbols, setSymbols] = createSignal('SPY,QQQ,IWM');
  const [unavailable, setUnavailable] = createSignal(false);

  const fetchStatus = async () => {
    try {
      const data = await apiClient.get('/inference/status');
      setStatus(data);
      setError(null);
      // Check if models are not loaded (no GPU)
      if (data.status === 'error' && data.models?.every((m: any) => !m.loaded)) {
        setUnavailable(true);
      }
    } catch (err: any) {
      // If 404 or 500, the service is unavailable
      if (err.status === 404 || err.status === 500) {
        setUnavailable(true);
      }
    }
  };

  const togglePipeline = async () => {
    if (unavailable()) return;
    setLoading(true);
    setError(null);
    try {
      if (status()?.pipeline_running) {
        await apiClient.post('/inference/pipeline/stop');
      } else {
        const symbolList = symbols().split(',').map(s => s.trim()).filter(s => s);
        await apiClient.post('/inference/pipeline/start', { symbols: symbolList });
      }
      setTimeout(fetchStatus, 1000);
    } catch (err: any) {
      const detail = err.detail?.detail || err.message || 'Failed to toggle pipeline';
      if (detail.includes('NVIDIA') || detail.includes('GPU') || detail.includes('CUDA')) {
        setUnavailable(true);
        setError('GPU not available on this server');
      } else {
        setError(detail);
      }
    } finally {
      setLoading(false);
    }
  };

  onMount(() => {
    fetchStatus();
    // Only poll if not unavailable
    const interval = setInterval(() => {
      if (!unavailable()) fetchStatus();
    }, 10000);
    onCleanup(() => clearInterval(interval));
  });

  const formatUptime = (seconds: number) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return `${h}h ${m}m ${s}s`;
  };

  return (
    <div class="bg-terminal-900 border border-terminal-800 rounded-lg p-4 space-y-4">
      <div class="flex items-center justify-between">
        <h3 class="text-sm font-bold text-white uppercase flex items-center gap-2">
          <Cpu class="w-4 h-4 text-accent-500" />
          HFT Engine
        </h3>
        <Show when={!unavailable()} fallback={
          <span class="text-[10px] text-gray-500 bg-gray-800 px-2 py-0.5 rounded">UNAVAILABLE</span>
        }>
          <div class="flex items-center gap-2">
            <div class={`w-2 h-2 rounded-full ${status()?.pipeline_running ? 'bg-success-500' : 'bg-gray-600'}`} />
            <span class="text-xs text-gray-400">{status()?.pipeline_running ? 'RUNNING' : 'STOPPED'}</span>
          </div>
        </Show>
      </div>

      <Show when={unavailable()}>
        <div class="text-xs text-warning-400 bg-warning-900/20 p-3 rounded border border-warning-900/50 flex items-start gap-2">
          <AlertTriangle class="w-4 h-4 shrink-0 mt-0.5" />
          <div>
            <div class="font-bold mb-1">GPU Required</div>
            <div class="text-gray-400">The HFT engine requires NVIDIA GPU hardware which is not available on this server. Contact support for GPU-enabled deployment.</div>
          </div>
        </div>
      </Show>

      <Show when={!unavailable()}>
        <Show when={error()}>
          <div class="text-xs text-danger-400 bg-danger-900/20 p-2 rounded border border-danger-900/50">
            {error()}
          </div>
        </Show>

        <div class="grid grid-cols-2 gap-4 text-xs">
          <div class="bg-terminal-950 p-2 rounded border border-terminal-800">
            <div class="text-gray-500 mb-1">Predictions</div>
            <div class="text-white font-mono text-lg">{status()?.total_predictions || 0}</div>
          </div>
          <div class="bg-terminal-950 p-2 rounded border border-terminal-800">
            <div class="text-gray-500 mb-1">Uptime</div>
            <div class="text-white font-mono text-lg">{status()?.uptime_seconds ? formatUptime(status()?.uptime_seconds!) : '0s'}</div>
          </div>
        </div>

        <div class="space-y-2">
          <label class="text-[10px] text-gray-500 uppercase">Target Symbols</label>
          <input 
            type="text" 
            value={symbols()}
            onInput={(e) => setSymbols(e.currentTarget.value)}
            disabled={status()?.pipeline_running}
            class="w-full bg-terminal-950 border border-terminal-700 rounded px-2 py-1.5 text-xs text-white focus:border-accent-500 focus:outline-none disabled:opacity-50"
            placeholder="SPY, QQQ..."
          />
        </div>

        <button
          onClick={togglePipeline}
          disabled={loading()}
          class={`w-full py-2 rounded text-xs font-bold flex items-center justify-center gap-2 transition-all ${
            status()?.pipeline_running
              ? 'bg-danger-500/20 text-danger-400 hover:bg-danger-500 hover:text-white border border-danger-500/30'
              : 'bg-success-500/20 text-success-400 hover:bg-success-500 hover:text-black border border-success-500/30'
          }`}
        >
          <Show when={loading()} fallback={
            status()?.pipeline_running ? <><Square class="w-3 h-3" /> STOP ENGINE</> : <><Play class="w-3 h-3" /> START ENGINE</>
          }>
            <RefreshCw class="w-3 h-3 animate-spin" /> PROCESSING...
          </Show>
        </button>
      </Show>
    </div>
  );
}
