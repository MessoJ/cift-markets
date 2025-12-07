/**
 * Input Component
 * 
 * Reusable input with validation states and icons.
 */

import { JSX, Show, splitProps } from 'solid-js';
import { twMerge } from 'tailwind-merge';

// ============================================================================
// TYPES
// ============================================================================

interface InputProps extends JSX.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  helperText?: string;
  leftIcon?: JSX.Element;
  rightIcon?: JSX.Element;
  fullWidth?: boolean;
}

// ============================================================================
// COMPONENT
// ============================================================================

export function Input(props: InputProps) {
  const [local, inputProps] = splitProps(props, [
    'label',
    'error',
    'helperText',
    'leftIcon',
    'rightIcon',
    'fullWidth',
    'class',
  ]);

  const inputClasses = twMerge(
    'input',
    local.error && 'input-error',
    local.leftIcon && 'pl-10',
    local.rightIcon && 'pr-10',
    local.fullWidth && 'w-full',
    local.class
  );

  return (
    <div class={twMerge('flex flex-col gap-1.5', local.fullWidth && 'w-full')}>
      <Show when={local.label}>
        <label class="text-sm font-medium text-gray-300">
          {local.label}
        </label>
      </Show>

      <div class="relative">
        <Show when={local.leftIcon}>
          <div class="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500">
            {local.leftIcon}
          </div>
        </Show>

        <input
          {...inputProps}
          class={inputClasses}
          aria-invalid={!!local.error}
          aria-describedby={local.error ? 'input-error' : local.helperText ? 'input-helper' : undefined}
        />

        <Show when={local.rightIcon}>
          <div class="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500">
            {local.rightIcon}
          </div>
        </Show>
      </div>

      <Show when={local.error}>
        <p id="input-error" class="text-sm text-danger-400">
          {local.error}
        </p>
      </Show>

      <Show when={local.helperText && !local.error}>
        <p id="input-helper" class="text-sm text-gray-500">
          {local.helperText}
        </p>
      </Show>
    </div>
  );
}
