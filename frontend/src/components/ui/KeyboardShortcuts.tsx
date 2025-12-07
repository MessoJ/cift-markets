/**
 * KeyboardShortcuts Component & System
 * 
 * Global keyboard shortcuts for power users.
 * Bloomberg/TradingView style command system.
 * 
 * Design System: Professional trading UI
 */

import { createSignal, onCleanup, For, Show, onMount } from 'solid-js';
import { Command, X, Search } from 'lucide-solid';

export interface Shortcut {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  description: string;
  action: () => void;
  category?: string;
}

// Global shortcut registry
const [shortcuts, setShortcuts] = createSignal<Shortcut[]>([]);
const [isHelpOpen, setIsHelpOpen] = createSignal(false);
const [isCommandPaletteOpen, setIsCommandPaletteOpen] = createSignal(false);

/**
 * Register a keyboard shortcut
 */
export function registerShortcut(shortcut: Shortcut) {
  setShortcuts((prev) => [...prev, shortcut]);
  
  // Return unregister function
  return () => {
    setShortcuts((prev) => prev.filter((s) => s !== shortcut));
  };
}

/**
 * Register multiple shortcuts at once
 */
export function registerShortcuts(newShortcuts: Shortcut[]) {
  setShortcuts((prev) => [...prev, ...newShortcuts]);
  
  return () => {
    setShortcuts((prev) => prev.filter((s) => !newShortcuts.includes(s)));
  };
}

/**
 * Format shortcut key for display
 */
function formatShortcutKey(shortcut: Shortcut): string {
  const parts: string[] = [];
  if (shortcut.ctrl) parts.push('Ctrl');
  if (shortcut.shift) parts.push('Shift');
  if (shortcut.alt) parts.push('Alt');
  parts.push(shortcut.key.toUpperCase());
  return parts.join(' + ');
}

/**
 * Global keyboard handler - mount this once in your app root
 */
export function KeyboardShortcutsProvider(props: { children: any }) {
  onMount(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Skip if user is typing in input
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) {
        // Allow escape to close modals even when in inputs
        if (e.key !== 'Escape') return;
      }
      
      // Built-in shortcuts
      // ? or Ctrl+/ = Show help
      if ((e.key === '?' && !e.ctrlKey) || (e.key === '/' && e.ctrlKey)) {
        e.preventDefault();
        setIsHelpOpen(true);
        return;
      }
      
      // Ctrl+K or Cmd+K = Command palette
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        setIsCommandPaletteOpen(true);
        return;
      }
      
      // Escape = Close any open panel
      if (e.key === 'Escape') {
        if (isHelpOpen()) {
          setIsHelpOpen(false);
          return;
        }
        if (isCommandPaletteOpen()) {
          setIsCommandPaletteOpen(false);
          return;
        }
      }
      
      // Check registered shortcuts
      for (const shortcut of shortcuts()) {
        const ctrlMatch = shortcut.ctrl ? (e.ctrlKey || e.metaKey) : !e.ctrlKey && !e.metaKey;
        const shiftMatch = shortcut.shift ? e.shiftKey : !e.shiftKey;
        const altMatch = shortcut.alt ? e.altKey : !e.altKey;
        const keyMatch = e.key.toLowerCase() === shortcut.key.toLowerCase();
        
        if (ctrlMatch && shiftMatch && altMatch && keyMatch) {
          e.preventDefault();
          shortcut.action();
          return;
        }
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    
    onCleanup(() => {
      window.removeEventListener('keydown', handleKeyDown);
    });
  });

  return (
    <>
      {props.children}
      <KeyboardShortcutsHelp />
      <CommandPalette />
    </>
  );
}

/**
 * Keyboard shortcuts help modal
 */
function KeyboardShortcutsHelp() {
  const groupedShortcuts = () => {
    const groups: Record<string, Shortcut[]> = {};
    
    // Add built-in shortcuts
    const builtIn: Shortcut[] = [
      { key: '?', description: 'Show this help', action: () => {}, category: 'General' },
      { key: 'k', ctrl: true, description: 'Command palette', action: () => {}, category: 'General' },
      { key: 'Escape', description: 'Close panels/modals', action: () => {}, category: 'General' },
    ];
    
    [...builtIn, ...shortcuts()].forEach((s) => {
      const category = s.category || 'General';
      if (!groups[category]) groups[category] = [];
      groups[category].push(s);
    });
    
    return groups;
  };

  return (
    <Show when={isHelpOpen()}>
      <div class="fixed inset-0 z-50 flex items-center justify-center">
        {/* Backdrop */}
        <div 
          class="absolute inset-0 bg-black/70" 
          onClick={() => setIsHelpOpen(false)}
        />
        
        {/* Modal */}
        <div class="relative bg-terminal-900 border border-terminal-700 rounded-lg shadow-2xl w-full max-w-2xl max-h-[80vh] overflow-hidden">
          {/* Header */}
          <div class="flex items-center justify-between px-6 py-4 border-b border-terminal-700">
            <div class="flex items-center gap-3">
              <Command class="w-5 h-5 text-accent-500" />
              <h2 class="text-lg font-semibold text-white">Keyboard Shortcuts</h2>
            </div>
            <button
              class="p-1 text-gray-500 hover:text-white transition-colors"
              onClick={() => setIsHelpOpen(false)}
            >
              <X class="w-5 h-5" />
            </button>
          </div>
          
          {/* Content */}
          <div class="p-6 overflow-y-auto max-h-[60vh]">
            <For each={Object.entries(groupedShortcuts())}>
              {([category, categoryShortcuts]) => (
                <div class="mb-6 last:mb-0">
                  <h3 class="text-xs font-semibold text-gray-500 uppercase mb-3">
                    {category}
                  </h3>
                  <div class="space-y-2">
                    <For each={categoryShortcuts}>
                      {(shortcut) => (
                        <div class="flex items-center justify-between py-2 px-3 bg-terminal-850 rounded">
                          <span class="text-sm text-gray-300">{shortcut.description}</span>
                          <kbd class="px-2 py-1 bg-terminal-800 border border-terminal-700 rounded text-xs font-mono text-gray-400">
                            {formatShortcutKey(shortcut)}
                          </kbd>
                        </div>
                      )}
                    </For>
                  </div>
                </div>
              )}
            </For>
          </div>
          
          {/* Footer */}
          <div class="px-6 py-3 border-t border-terminal-700 bg-terminal-850">
            <p class="text-xs text-gray-500 text-center">
              Press <kbd class="px-1 py-0.5 bg-terminal-700 rounded text-gray-400">?</kbd> anytime to show this help
            </p>
          </div>
        </div>
      </div>
    </Show>
  );
}

/**
 * Command Palette - Spotlight-style command launcher
 */
function CommandPalette() {
  const [search, setSearch] = createSignal('');
  
  const filteredShortcuts = () => {
    const query = search().toLowerCase();
    if (!query) return shortcuts();
    return shortcuts().filter((s) => 
      s.description.toLowerCase().includes(query) ||
      s.category?.toLowerCase().includes(query)
    );
  };
  
  const handleSelect = (shortcut: Shortcut) => {
    shortcut.action();
    setIsCommandPaletteOpen(false);
    setSearch('');
  };

  return (
    <Show when={isCommandPaletteOpen()}>
      <div class="fixed inset-0 z-50 flex items-start justify-center pt-[15vh]">
        {/* Backdrop */}
        <div 
          class="absolute inset-0 bg-black/70" 
          onClick={() => {
            setIsCommandPaletteOpen(false);
            setSearch('');
          }}
        />
        
        {/* Palette */}
        <div class="relative bg-terminal-900 border border-terminal-700 rounded-lg shadow-2xl w-full max-w-lg overflow-hidden">
          {/* Search input */}
          <div class="flex items-center gap-3 px-4 py-3 border-b border-terminal-700">
            <Search class="w-5 h-5 text-gray-500" />
            <input
              type="text"
              value={search()}
              onInput={(e) => setSearch(e.currentTarget.value)}
              placeholder="Type a command..."
              class="flex-1 bg-transparent text-white placeholder-gray-500 text-sm focus:outline-none"
              autofocus
            />
            <kbd class="px-2 py-0.5 bg-terminal-800 border border-terminal-700 rounded text-[10px] font-mono text-gray-500">
              ESC
            </kbd>
          </div>
          
          {/* Results */}
          <div class="max-h-[300px] overflow-y-auto">
            <Show when={filteredShortcuts().length > 0} fallback={
              <div class="py-8 text-center text-sm text-gray-500">
                No commands found
              </div>
            }>
              <For each={filteredShortcuts()}>
                {(shortcut) => (
                  <button
                    class="w-full flex items-center justify-between px-4 py-3 hover:bg-terminal-800 transition-colors text-left"
                    onClick={() => handleSelect(shortcut)}
                  >
                    <div>
                      <div class="text-sm text-white">{shortcut.description}</div>
                      <Show when={shortcut.category}>
                        <div class="text-xs text-gray-500 mt-0.5">{shortcut.category}</div>
                      </Show>
                    </div>
                    <kbd class="px-2 py-1 bg-terminal-850 border border-terminal-700 rounded text-xs font-mono text-gray-400">
                      {formatShortcutKey(shortcut)}
                    </kbd>
                  </button>
                )}
              </For>
            </Show>
          </div>
        </div>
      </div>
    </Show>
  );
}

/**
 * Shortcut hint component - shows keyboard shortcut inline
 */
interface ShortcutHintProps {
  keys: string | string[];
  className?: string;
}

export function ShortcutHint(props: ShortcutHintProps) {
  const keyArray = () => Array.isArray(props.keys) ? props.keys : [props.keys];
  
  return (
    <span class={`inline-flex items-center gap-0.5 ${props.className || ''}`}>
      <For each={keyArray()}>
        {(key, index) => (
          <>
            <kbd class="px-1.5 py-0.5 bg-terminal-800 border border-terminal-700 rounded text-[10px] font-mono text-gray-500">
              {key}
            </kbd>
            <Show when={index() < keyArray().length - 1}>
              <span class="text-gray-600 text-[10px]">+</span>
            </Show>
          </>
        )}
      </For>
    </span>
  );
}

// Export utilities
export { isHelpOpen, setIsHelpOpen, isCommandPaletteOpen, setIsCommandPaletteOpen };
export default KeyboardShortcutsProvider;
