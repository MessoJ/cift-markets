/**
 * Button Component
 * 
 * Reusable button with variants, sizes, and loading states.
 */

import { JSX, Show, splitProps } from 'solid-js';
import { Loader2 } from 'lucide-solid';
import { twMerge } from 'tailwind-merge';

// ============================================================================
// TYPES
// ============================================================================

interface ButtonProps extends JSX.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'success' | 'danger' | 'ghost' | 'link';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  icon?: JSX.Element;
  iconPosition?: 'left' | 'right';
  fullWidth?: boolean;
}

// ============================================================================
// COMPONENT
// ============================================================================

export function Button(props: ButtonProps) {
  const [local, buttonProps] = splitProps(props, [
    'variant',
    'size',
    'loading',
    'icon',
    'iconPosition',
    'fullWidth',
    'class',
    'children',
    'disabled',
  ]);

  const variantClasses = {
    primary: 'btn-primary',
    success: 'btn-success',
    danger: 'btn-danger',
    ghost: 'btn-ghost',
    link: 'btn-link',
  };

  const sizeClasses = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-sm',
    lg: 'px-6 py-3 text-base',
  };

  const classes = twMerge(
    'btn',
    variantClasses[local.variant || 'primary'],
    sizeClasses[local.size || 'md'],
    local.fullWidth && 'w-full',
    local.class
  );

  const isDisabled = () => local.disabled || local.loading;

  return (
    <button
      {...buttonProps}
      class={classes}
      disabled={isDisabled()}
      aria-busy={local.loading}
    >
      <Show when={local.loading}>
        <Loader2 class="w-4 h-4 animate-spin" />
      </Show>
      
      <Show when={!local.loading && local.icon && local.iconPosition !== 'right'}>
        {local.icon}
      </Show>

      <Show when={local.children}>
        <span>{local.children}</span>
      </Show>

      <Show when={!local.loading && local.icon && local.iconPosition === 'right'}>
        {local.icon}
      </Show>
    </button>
  );
}
