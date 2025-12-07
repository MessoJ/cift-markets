/**
 * Modal Component
 * 
 * Accessible modal dialog with animations and keyboard support.
 * Updated with "Terminal" theme and glassmorphism.
 */

import { JSX, Show, createEffect, onCleanup } from 'solid-js';
import { Portal } from 'solid-js/web';
import { X } from 'lucide-solid';
import { twMerge } from 'tailwind-merge';

// ============================================================================
// TYPES
// ============================================================================

interface ModalProps {
  open: boolean;
  onClose: () => void;
  title?: string;
  subtitle?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  children: JSX.Element;
  footer?: JSX.Element;
  closeOnOverlayClick?: boolean;
  showCloseButton?: boolean;
  className?: string;
}

// ============================================================================
// COMPONENT
// ============================================================================

export function Modal(props: ModalProps) {
  let modalRef: HTMLDivElement | undefined;

  const sizeClasses = {
    sm: 'max-w-sm',
    md: 'max-w-lg',
    lg: 'max-w-2xl',
    xl: 'max-w-4xl',
    full: 'max-w-7xl',
  };

  // Handle escape key
  createEffect(() => {
    if (!props.open) return;

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        props.onClose();
      }
    };

    document.addEventListener('keydown', handleEscape);
    
    // Prevent body scroll
    document.body.style.overflow = 'hidden';

    onCleanup(() => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = '';
    });
  });

  const handleOverlayClick = (e: MouseEvent) => {
    if (props.closeOnOverlayClick !== false && e.target === e.currentTarget) {
      props.onClose();
    }
  };

  return (
    <Show when={props.open}>
      <Portal>
        {/* Overlay */}
        <div
          class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 sm:p-6 animate-in fade-in duration-200"
          onClick={handleOverlayClick}
          role="presentation"
        >
          {/* Modal Content */}
          <div
            ref={modalRef}
            class={twMerge(
              'relative w-full bg-terminal-900 border border-terminal-700 shadow-2xl shadow-black/50 rounded-xl overflow-hidden transform transition-all animate-in fade-in zoom-in-95 duration-200 flex flex-col max-h-[90vh]',
              sizeClasses[props.size || 'md'],
              props.className
            )}
            role="dialog"
            aria-modal="true"
            aria-labelledby={props.title ? 'modal-title' : undefined}
          >
            {/* Header */}
            <Show when={props.title || props.showCloseButton !== false}>
              <div class="flex items-start justify-between p-6 border-b border-terminal-800 bg-terminal-800/30 shrink-0">
                <div class="flex-1 pr-4">
                  <Show when={props.title}>
                    <h2 id="modal-title" class="text-xl font-semibold text-white tracking-tight">
                      {props.title}
                    </h2>
                  </Show>
                  <Show when={props.subtitle}>
                    <p class="text-sm text-terminal-400 mt-1">{props.subtitle}</p>
                  </Show>
                </div>
                <Show when={props.showCloseButton !== false}>
                  <button
                    onClick={props.onClose}
                    class="p-2 text-terminal-400 hover:text-white transition-colors rounded-lg hover:bg-terminal-800/50"
                    aria-label="Close modal"
                  >
                    <X class="w-5 h-5" />
                  </button>
                </Show>
              </div>
            </Show>

            {/* Body */}
            <div class="p-6 overflow-y-auto custom-scrollbar">
              {props.children}
            </div>

            {/* Footer */}
            <Show when={props.footer}>
              <div class="flex items-center justify-end gap-3 p-6 border-t border-terminal-800 bg-terminal-800/30 shrink-0">
                {props.footer}
              </div>
            </Show>
          </div>
        </div>
      </Portal>
    </Show>
  );
}
