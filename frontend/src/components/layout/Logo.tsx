/**
 * CIFT MARKETS - Enterprise-Grade Wordmark Logo System
 *
 * DESIGN PHILOSOPHY (Research-Based Fintech Branding):
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * 
 * 1. TWO-TIER HIERARCHY: Establishes visual weight distribution
 *    - "CIFT" = Bold, heavyweight, institutional trust
 *    - "MARKETS" = Regular weight, professional refinement
 * 
 * 2. TYPOGRAPHIC EXCELLENCE:
 *    - Sans-serif (Inter/System UI) for modern fintech aesthetic
 *    - Tight letter-spacing for premium feel
 *    - Balanced x-height for digital scalability
 * 
 * 3. COLOR PSYCHOLOGY (Fintech Standards):
 *    - White (#FFFFFF): Trust, clarity, professionalism
 *    - Orange Accent (#F97316): Innovation, energy, accessibility
 *    - Used strategically on "CIFT" to create brand anchor
 * 
 * 4. ADAPTIVE VARIANTS:
 *    - Full: Complete wordmark for headers/landing pages
 *    - Compact: Space-efficient for sidebars/mobile
 *    - Icon-only: "C|M" monogram for favicons/avatars
 * 
 * 5. SEMANTIC STRUCTURE:
 *    - Vertical divider "│" separates tiers (visual breathing room)
 *    - Creates left-right balance without icons
 *    - Maintains readability at all sizes (16px to 240px)
 * 
 * COMPETITIVE ANALYSIS INSIGHTS:
 * - Bloomberg: Bold primary + light secondary wordmark
 * - Robinhood: Single-color minimalist typography
 * - Stripe: Clean sans-serif with strategic weight variation
 * - Our approach: Combines Bloomberg's hierarchy with Stripe's simplicity
 * 
 * TECHNICAL SPECIFICATIONS:
 * - Font: System UI stack (Inter fallback)
 * - Weight distribution: 700 (CIFT) / 400 (MARKETS)
 * - Letter-spacing: -0.02em (optical tightness)
 * - No gradients, no icons, no shadows (per requirements)
 * 
 * ACCESSIBILITY:
 * - WCAG AAA contrast ratio (white on dark: 21:1)
 * - Readable at 16px minimum
 * - Screen reader friendly semantic structure
 */

import { twMerge } from 'tailwind-merge';
import { Show } from 'solid-js';

interface LogoProps {
  size?: 'sm' | 'md' | 'lg' | 'xl';
  variant?: 'full' | 'compact' | 'icon-only';
  theme?: 'dark' | 'light';
  animated?: boolean;
  showText?: boolean;
  class?: string;
}

export function Logo(props: LogoProps) {
  const variant = () => props.variant || 'full';
  const theme = () => props.theme || 'dark';
  
  // Size system (responsive scale)
  const sizeClasses = () => {
    const baseSize = props.size || 'md';
    
    if (variant() === 'icon-only') {
      switch (baseSize) {
        case 'sm': return 'text-base';
        case 'md': return 'text-lg';
        case 'lg': return 'text-xl';
        case 'xl': return 'text-2xl';
        default: return 'text-lg';
      }
    }
    
    switch (baseSize) {
      case 'sm': return 'text-base';
      case 'md': return 'text-xl';
      case 'lg': return 'text-3xl';
      case 'xl': return 'text-5xl';
      default: return 'text-xl';
    }
  };
  
  // Theme-based color system
  const primaryColor = () => theme() === 'dark' ? 'text-white' : 'text-gray-900';
  const secondaryColor = () => theme() === 'dark' ? 'text-gray-400' : 'text-gray-600';
  const accentColor = () => 'text-accent-500';
  const dividerColor = () => theme() === 'dark' ? 'text-gray-700' : 'text-gray-300';
  
  // Animation classes
  const animationClass = () => props.animated ? 'animate-fade-in' : '';
  const showSecondary = () => props.showText !== false;
  
  /**
   * FULL VARIANT - Complete wordmark with hierarchy
   * Best for: Headers, landing pages, auth screens
   */
  const renderFull = () => (
    <div class={twMerge(
      'inline-flex items-baseline gap-2 font-sans',
      sizeClasses(),
      animationClass(),
      props.class
    )}>
      {/* Primary wordmark - Bold weight establishes trust */}
      <span class={twMerge('font-bold tracking-tight', accentColor())}>
        CIFT
      </span>
      <Show when={showSecondary()}>
        {/* Visual separator - Creates tier distinction */}
        <span class={twMerge('font-thin', dividerColor())}>
          │
        </span>
        
        {/* Secondary wordmark - Regular weight for refinement */}
        <span class={twMerge('font-normal tracking-tight', primaryColor())}>
          MARKETS
        </span>
      </Show>
    </div>
  );
  
  /**
   * COMPACT VARIANT - Space-efficient design
   * Best for: Sidebars, mobile headers, tight layouts
   */
  const renderCompact = () => (
    <div class={twMerge(
      'inline-flex items-center gap-1.5 font-sans',
      sizeClasses(),
      animationClass(),
      props.class
    )}>
      {/* Condensed primary */}
      <span class={twMerge('font-bold tracking-tighter', accentColor())}>
        CIFT
      </span>
      <Show when={showSecondary()}>
        {/* Micro separator */}
        <span class={twMerge('font-extralight text-[0.6em]', dividerColor())}>
          │
        </span>
        
        {/* Full secondary - MARKETS, not MKT */}
        <span class={twMerge('font-medium tracking-tight', secondaryColor())}>
          MARKETS
        </span>
      </Show>
    </div>
  );
  
  /**
   * ICON-ONLY VARIANT - Monogram for constrained spaces
   * Best for: Favicons, avatars, mobile tabs, notifications
   */
  const renderIconOnly = () => (
    <div class={twMerge(
      'inline-flex items-center justify-center',
      'w-10 h-10 rounded-lg',
      'bg-accent-500/10 border border-accent-500/20',
      'font-sans font-bold',
      sizeClasses(),
      animationClass(),
      props.class
    )}>
      <span class={accentColor()}>C</span>
      <span class={twMerge('mx-0.5 text-[0.5em]', dividerColor())}>│</span>
      <span class={primaryColor()}>M</span>
    </div>
  );
  
  return (
    <Show
      when={variant() === 'full'}
      fallback={
        <Show when={variant() === 'compact'} fallback={renderIconOnly()}>
          {renderCompact()}
        </Show>
      }
    >
      {renderFull()}
    </Show>
  );
}
