/**
 * Chart Template Manager
 * 
 * UI for saving, loading, and managing chart templates.
 * Integrates with backend API for persistence.
 */

import { createSignal, For, Show } from 'solid-js';
import { Save, FolderOpen, Trash2, Star, Plus } from 'lucide-solid';
import type { IndicatorConfig } from './IndicatorPanel';

export interface ChartTemplate {
  id: string;
  name: string;
  description?: string;
  config: {
    symbol: string;
    timeframe: string;
    chartType: string;
    indicators: any[];
    viewMode: string;
    multiLayout?: string;
    multiTimeframes?: string[];
  };
  is_default: boolean;
  created_at: string;
  updated_at: string;
}

export interface TemplateManagerProps {
  symbol: string;
  timeframe: string;
  chartType: string;
  indicators: IndicatorConfig[];
  viewMode: 'single' | 'multi';
  multiLayout?: string;
  multiTimeframes?: string[];
  onLoadTemplate?: (template: ChartTemplate) => void;
}

export default function TemplateManager(props: TemplateManagerProps) {
  const [templates, setTemplates] = createSignal<ChartTemplate[]>([]);
  const [showSaveDialog, setShowSaveDialog] = createSignal(false);
  const [showLoadDialog, setShowLoadDialog] = createSignal(false);
  const [loading, setLoading] = createSignal(false);
  
  // Save dialog fields
  const [saveName, setSaveName] = createSignal('');
  const [saveDescription, setSaveDescription] = createSignal('');
  const [saveAsDefault, setSaveAsDefault] = createSignal(false);

  /**
   * Load templates from API
   */
  const loadTemplates = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/v1/chart-templates', {
        credentials: 'include',
      });
      
      if (response.ok) {
        const data = await response.json();
        setTemplates(data);
        console.log(`📁 Loaded ${data.length} templates`);
      }
    } catch (error) {
      console.error('Failed to load templates:', error);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Save current chart as template
   */
  const saveTemplate = async () => {
    if (!saveName().trim()) {
      alert('Please enter a template name');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/v1/chart-templates', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          name: saveName(),
          description: saveDescription() || null,
          is_default: saveAsDefault(),
          config: {
            symbol: props.symbol,
            timeframe: props.timeframe,
            chartType: props.chartType,
            indicators: props.indicators,
            viewMode: props.viewMode,
            multiLayout: props.multiLayout,
            multiTimeframes: props.multiTimeframes,
          },
        }),
      });

      if (response.ok) {
        const newTemplate = await response.json();
        setTemplates([newTemplate, ...templates()]);
        setShowSaveDialog(false);
        setSaveName('');
        setSaveDescription('');
        setSaveAsDefault(false);
        console.log(`💾 Saved template: ${newTemplate.name}`);
      } else {
        const error = await response.json();
        alert(`Failed to save template: ${error.detail}`);
      }
    } catch (error) {
      console.error('Failed to save template:', error);
      alert('Failed to save template');
    } finally {
      setLoading(false);
    }
  };

  /**
   * Load a template
   */
  const loadTemplate = (template: ChartTemplate) => {
    props.onLoadTemplate?.(template);
    setShowLoadDialog(false);
    console.log(`📂 Loaded template: ${template.name}`);
  };

  /**
   * Delete a template
   */
  const deleteTemplate = async (id: string, name: string) => {
    if (!confirm(`Delete template "${name}"?`)) return;

    setLoading(true);
    try {
      const response = await fetch(`/api/v1/chart-templates/${id}`, {
        method: 'DELETE',
        credentials: 'include',
      });

      if (response.ok) {
        setTemplates(templates().filter(t => t.id !== id));
        console.log(`🗑️ Deleted template: ${name}`);
      }
    } catch (error) {
      console.error('Failed to delete template:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div class="space-y-2">
      {/* Action Buttons */}
      <div class="flex gap-2">
        <button
          class="flex-1 px-3 py-2 bg-primary-600 hover:bg-primary-700 text-white text-sm rounded flex items-center justify-center gap-2 transition-colors"
          onClick={() => setShowSaveDialog(true)}
        >
          <Save size={16} />
          Save Template
        </button>
        <button
          class="flex-1 px-3 py-2 bg-terminal-800 hover:bg-terminal-750 text-gray-300 text-sm rounded flex items-center justify-center gap-2 transition-colors"
          onClick={() => {
            loadTemplates();
            setShowLoadDialog(true);
          }}
        >
          <FolderOpen size={16} />
          Load Template
        </button>
      </div>

      {/* Save Dialog */}
      <Show when={showSaveDialog()}>
        <div class="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center" onClick={() => setShowSaveDialog(false)}>
          <div class="bg-terminal-900 border border-terminal-750 rounded-lg p-6 w-full max-w-md" onClick={(e) => e.stopPropagation()}>
            <h3 class="text-lg font-semibold text-white mb-4">Save Chart Template</h3>
            
            <div class="space-y-4">
              <div>
                <label class="block text-sm text-gray-400 mb-1">Template Name *</label>
                <input
                  type="text"
                  class="w-full px-3 py-2 bg-terminal-800 border border-terminal-750 rounded text-white"
                  placeholder="My Trading Setup"
                  value={saveName()}
                  onInput={(e) => setSaveName(e.currentTarget.value)}
                />
              </div>

              <div>
                <label class="block text-sm text-gray-400 mb-1">Description (Optional)</label>
                <textarea
                  class="w-full px-3 py-2 bg-terminal-800 border border-terminal-750 rounded text-white resize-none"
                  rows="3"
                  placeholder="Brief description of this template..."
                  value={saveDescription()}
                  onInput={(e) => setSaveDescription(e.currentTarget.value)}
                />
              </div>

              <label class="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  class="rounded"
                  checked={saveAsDefault()}
                  onChange={(e) => setSaveAsDefault(e.currentTarget.checked)}
                />
                <span class="text-sm text-gray-300">Set as default template</span>
              </label>

              <div class="flex gap-2 pt-2">
                <button
                  class="flex-1 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded transition-colors"
                  onClick={saveTemplate}
                  disabled={loading()}
                >
                  {loading() ? 'Saving...' : 'Save'}
                </button>
                <button
                  class="px-4 py-2 bg-terminal-800 hover:bg-terminal-750 text-gray-300 rounded transition-colors"
                  onClick={() => setShowSaveDialog(false)}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      </Show>

      {/* Load Dialog */}
      <Show when={showLoadDialog()}>
        <div class="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center" onClick={() => setShowLoadDialog(false)}>
          <div class="bg-terminal-900 border border-terminal-750 rounded-lg p-6 w-full max-w-2xl max-h-[80vh] overflow-hidden flex flex-col" onClick={(e) => e.stopPropagation()}>
            <h3 class="text-lg font-semibold text-white mb-4">Load Chart Template</h3>
            
            <Show when={loading()}>
              <div class="text-center py-8 text-gray-500">Loading templates...</div>
            </Show>

            <Show when={!loading() && templates().length === 0}>
              <div class="text-center py-8 text-gray-500">
                <p>No templates saved yet.</p>
                <p class="text-sm mt-2">Save your first template to get started!</p>
              </div>
            </Show>

            <Show when={!loading() && templates().length > 0}>
              <div class="flex-1 overflow-y-auto space-y-2">
                <For each={templates()}>
                  {(template) => (
                    <div class="p-3 bg-terminal-800 border border-terminal-750 rounded hover:border-primary-500 transition-colors">
                      <div class="flex items-start justify-between">
                        <div class="flex-1 cursor-pointer" onClick={() => loadTemplate(template)}>
                          <div class="flex items-center gap-2">
                            <h4 class="font-medium text-white">{template.name}</h4>
                            <Show when={template.is_default}>
                              <Star size={14} class="text-yellow-500 fill-yellow-500" />
                            </Show>
                          </div>
                          <Show when={template.description}>
                            <p class="text-sm text-gray-400 mt-1">{template.description}</p>
                          </Show>
                          <div class="flex gap-3 mt-2 text-xs text-gray-500">
                            <span>{template.config.symbol} • {template.config.timeframe}</span>
                            <span>{template.config.indicators.length} indicators</span>
                            <span>{new Date(template.created_at).toLocaleDateString()}</span>
                          </div>
                        </div>
                        <button
                          class="p-1 text-red-500 hover:bg-red-500/10 rounded transition-colors"
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteTemplate(template.id, template.name);
                          }}
                          title="Delete template"
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>
                    </div>
                  )}
                </For>
              </div>
            </Show>

            <button
              class="mt-4 px-4 py-2 bg-terminal-800 hover:bg-terminal-750 text-gray-300 rounded transition-colors"
              onClick={() => setShowLoadDialog(false)}
            >
              Close
            </button>
          </div>
        </div>
      </Show>
    </div>
  );
}
