/**
 * Custom AI Icon - CIFT Markets Brand
 * 
 * A unique brain/circuit design representing AI-powered analysis.
 * Uses brand accent orange (#f97316) as primary color.
 */

import { JSX } from 'solid-js';

interface AIIconProps {
  class?: string;
  size?: number | string;
  color?: string;
}

export function AIIcon(props: AIIconProps): JSX.Element {
  const size = props.size || 24;
  // Brand accent orange
  const color = props.color || '#f97316';
  
  return (
    <svg 
      xmlns="http://www.w3.org/2000/svg" 
      viewBox="0 0 64 64" 
      width={size} 
      height={size}
      class={props.class}
      fill="none"
    >
      {/* Brain outline with circuit pattern */}
      <defs>
        <linearGradient id="aiGradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style={`stop-color:${color};stop-opacity:1`} />
          <stop offset="100%" style={`stop-color:${color};stop-opacity:0.7`} />
        </linearGradient>
      </defs>
      
      {/* Outer brain shape */}
      <path 
        d="M32 8C20 8 12 18 12 28c0 6 3 11 7 14v8c0 2 1.5 3.5 3.5 3.5h19c2 0 3.5-1.5 3.5-3.5v-8c4-3 7-8 7-14 0-10-8-20-20-20z" 
        stroke="url(#aiGradient)" 
        stroke-width="2.5" 
        stroke-linecap="round"
        stroke-linejoin="round"
      />
      
      {/* Brain center line */}
      <path 
        d="M32 12v38" 
        stroke={color} 
        stroke-width="1.5" 
        stroke-linecap="round"
        stroke-dasharray="3 2"
        opacity="0.6"
      />
      
      {/* Left brain circuits */}
      <path 
        d="M20 20h6M20 28h8M20 36h6" 
        stroke={color} 
        stroke-width="2" 
        stroke-linecap="round"
      />
      
      {/* Right brain circuits */}
      <path 
        d="M38 20h6M36 28h8M38 36h6" 
        stroke={color} 
        stroke-width="2" 
        stroke-linecap="round"
      />
      
      {/* Neural connection dots - left */}
      <circle cx="18" cy="20" r="2" fill={color} />
      <circle cx="18" cy="28" r="2" fill={color} />
      <circle cx="18" cy="36" r="2" fill={color} />
      
      {/* Neural connection dots - right */}
      <circle cx="46" cy="20" r="2" fill={color} />
      <circle cx="46" cy="28" r="2" fill={color} />
      <circle cx="46" cy="36" r="2" fill={color} />
      
      {/* Center brain node */}
      <circle cx="32" cy="24" r="3" fill={color} />
      
      {/* Pulse rings around center node */}
      <circle 
        cx="32" 
        cy="24" 
        r="5" 
        stroke={color} 
        stroke-width="1" 
        fill="none" 
        opacity="0.5"
      />
      <circle 
        cx="32" 
        cy="24" 
        r="7" 
        stroke={color} 
        stroke-width="0.5" 
        fill="none" 
        opacity="0.3"
      />
      
      {/* Bottom connectors (processing) */}
      <rect x="22" y="50" width="6" height="4" rx="1" fill={color} opacity="0.8" />
      <rect x="36" y="50" width="6" height="4" rx="1" fill={color} opacity="0.8" />
    </svg>
  );
}

// Animated variant for loading states
export function AIIconAnimated(props: AIIconProps): JSX.Element {
  const size = props.size || 24;
  const color = props.color || '#f97316';
  
  return (
    <div class="relative" style={{ width: `${size}px`, height: `${size}px` }}>
      <AIIcon {...props} />
      {/* Animated pulse overlay */}
      <div 
        class="absolute inset-0 rounded-full animate-ping opacity-20"
        style={{ background: `radial-gradient(circle, ${color} 0%, transparent 70%)` }}
      />
    </div>
  );
}

export default AIIcon;
