import { Show, createSignal } from 'solid-js';
import { Brain, ArrowRight, Loader2 } from 'lucide-solid';
import { useNavigate } from '@solidjs/router';
import { apiClient } from '~/lib/api/client';
import { Modal } from '~/components/ui/Modal';
import { Button } from '~/components/ui/Button';

interface QuickAnalyzeButtonProps {
  symbol: string;
  className?: string;
  variant?: 'icon' | 'button' | 'text';
  size?: 'sm' | 'md' | 'lg';
  compact?: boolean;
}

export function QuickAnalyzeButton(props: QuickAnalyzeButtonProps) {
  const navigate = useNavigate();
  const [isOpen, setIsOpen] = createSignal(false);
  const [loading, setLoading] = createSignal(false);
  const [summary, setSummary] = createSignal<any>(null);

  const handleAnalyze = async (e: Event) => {
    e.stopPropagation();
    e.preventDefault();
    setIsOpen(true);
    
    if (!summary()) {
      setLoading(true);
      try {
        // We use the same endpoint but we'll just display the summary
        const data = await apiClient.get(`/analysis/${props.symbol}`);
        setSummary(data);
      } catch (err) {
        console.error('Quick analysis failed:', err);
      } finally {
        setLoading(false);
      }
    }
  };

  const goToFullAnalysis = (e: Event) => {
    e.stopPropagation();
    e.preventDefault();
    setIsOpen(false);
    navigate(`/analysis/${props.symbol}`);
  };

  return (
    <>
      <button 
        onClick={handleAnalyze}
        class={props.className || "p-1.5 text-cyan-400 hover:bg-cyan-500/10 rounded-lg transition-colors"}
        title={`Analyze ${props.symbol}`}
      >
        <Brain class={props.size === 'sm' ? "w-3 h-3" : "w-4 h-4"} />
        <Show when={props.variant === 'button'}>
          <span class="ml-2">Analyze</span>
        </Show>
      </button>

      <Modal
        open={isOpen()}
        onClose={() => setIsOpen(false)}
        title={`Quick Analysis: ${props.symbol}`}
        size="md"
        className="bg-slate-900 border-slate-800"
      >
        <div class="p-6">
          <Show when={loading()}>
            <div class="flex flex-col items-center justify-center py-8">
              <Loader2 class="w-8 h-8 text-cyan-400 animate-spin mb-4" />
              <p class="text-slate-400">Analyzing market data...</p>
            </div>
          </Show>

          <Show when={!loading() && summary()}>
            <div class="space-y-6">
              {/* Score & Signal */}
              <div class="flex items-center justify-between bg-slate-800/50 p-4 rounded-xl border border-slate-700">
                <div class="text-center">
                  <div class="text-3xl font-bold text-white">{summary()?.overall_score}</div>
                  <div class="text-xs text-slate-400">AI Score</div>
                </div>
                <div class="h-8 w-px bg-slate-700"></div>
                <div class="text-center">
                  <div class={`text-xl font-bold ${
                    summary()?.suggested_action === 'BUY' ? 'text-green-400' :
                    summary()?.suggested_action === 'SELL' ? 'text-red-400' : 'text-yellow-400'
                  }`}>
                    {summary()?.suggested_action}
                  </div>
                  <div class="text-xs text-slate-400">Suggestion</div>
                </div>
                <div class="h-8 w-px bg-slate-700"></div>
                <div class="text-center">
                  <div class="text-xl font-bold text-white">{summary()?.risk?.risk_level?.toUpperCase()}</div>
                  <div class="text-xs text-slate-400">Risk Level</div>
                </div>
              </div>

              {/* Key Factors */}
              <div class="space-y-3">
                <h4 class="text-sm font-medium text-slate-400">Key Drivers</h4>
                <div class="grid grid-cols-2 gap-3">
                  <div class="bg-slate-800/30 p-3 rounded-lg border border-slate-700/50">
                    <div class="text-xs text-slate-500 mb-1">Technical</div>
                    <div class={`font-medium ${
                      summary()?.technical?.signal?.includes('BULL') ? 'text-green-400' : 
                      summary()?.technical?.signal?.includes('BEAR') ? 'text-red-400' : 'text-yellow-400'
                    }`}>
                      {summary()?.technical?.signal?.replace('_', ' ')}
                    </div>
                  </div>
                  <div class="bg-slate-800/30 p-3 rounded-lg border border-slate-700/50">
                    <div class="text-xs text-slate-500 mb-1">Sentiment</div>
                    <div class={`font-medium ${
                      summary()?.sentiment?.signal?.includes('BULL') ? 'text-green-400' : 
                      summary()?.sentiment?.signal?.includes('BEAR') ? 'text-red-400' : 'text-yellow-400'
                    }`}>
                      {summary()?.sentiment?.signal?.replace('_', ' ')}
                    </div>
                  </div>
                </div>
              </div>

              {/* Action Button */}
              <Button 
                onClick={goToFullAnalysis}
                class="w-full bg-cyan-600 hover:bg-cyan-500 text-white py-3 rounded-lg flex items-center justify-center gap-2"
              >
                View Full Analysis <ArrowRight class="w-4 h-4" />
              </Button>
            </div>
          </Show>
        </div>
      </Modal>
    </>
  );
}
