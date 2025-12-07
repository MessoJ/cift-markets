/**
 * Card Component
 * 
 * Reusable card container with variants.
 */

import { JSX, Show, splitProps } from 'solid-js';
import { twMerge } from 'tailwind-merge';

// ============================================================================
// TYPES
// ============================================================================

interface CardProps extends JSX.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'glass' | 'interactive';
  padding?: 'none' | 'sm' | 'md' | 'lg';
  title?: string;
  subtitle?: string;
  headerAction?: JSX.Element;
}

// ============================================================================
// COMPONENT
// ============================================================================

export function Card(props: CardProps) {
  const [local, divProps] = splitProps(props, [
    'variant',
    'padding',
    'title',
    'subtitle',
    'headerAction',
    'class',
    'children',
  ]);

  const variantClasses = {
    default: 'card',
    glass: 'card-glass',
    interactive: 'card-interactive',
  };

  const paddingClasses = {
    none: 'p-0',
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8',
  };

  const hasHeader = () => local.title || local.subtitle || local.headerAction;

  const classes = twMerge(
    variantClasses[local.variant || 'default'],
    local.padding && paddingClasses[local.padding],
    local.class
  );

  return (
    <div {...divProps} class={classes}>
      <Show when={hasHeader()}>
        <div class="flex items-start justify-between mb-6">
          <div class="flex-1">
            <Show when={local.title}>
              <h3 class="text-lg font-semibold text-white">{local.title}</h3>
            </Show>
            <Show when={local.subtitle}>
              <p class="text-sm text-gray-400 mt-1">{local.subtitle}</p>
            </Show>
          </div>
          <Show when={local.headerAction}>
            <div class="ml-4">{local.headerAction}</div>
          </Show>
        </div>
      </Show>

      {local.children}
    </div>
  );
}
