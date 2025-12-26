import { JSX, mergeProps } from 'solid-js';

interface AIIconProps {
  class?: string;
  animate?: boolean;
  size?: number | string;
  color?: string;
}

export const AIIcon = (props: AIIconProps): JSX.Element => {
  const merged = mergeProps({ size: 24, animate: false, color: '#f97316' }, props);
  
  return (
    <svg 
      width={merged.size} 
      height={merged.size} 
      viewBox='0 0 200 200' 
      fill='none' 
      xmlns='http://www.w3.org/2000/svg'
      class={merged.class}
      style={merged.animate ? {
        animation: 'spin 2s linear infinite'
      } : {}}
    >
      {/* Outer ring */}
      <circle 
        cx='100' 
        cy='100' 
        r='80' 
        stroke={merged.color} 
        stroke-width='3' 
        fill='none'
        opacity='0.3'
      />
      
      {/* Middle ring with gradient */}
      <circle 
        cx='100' 
        cy='100' 
        r='60' 
        stroke='url(#gradient1)' 
        stroke-width='4' 
        fill='none'
        stroke-dasharray='20 10'
      />
      
      {/* Inner hexagon */}
      <path 
        d='M 100 30 L 130 50 L 130 90 L 100 110 L 70 90 L 70 50 Z' 
        fill='url(#gradient2)'
        stroke={merged.color}
        stroke-width='2'
        opacity='0.8'
      />
      
      {/* Center core */}
      <circle 
        cx='100' 
        cy='100' 
        r='20' 
        fill={merged.color}
      />
      
      {/* Center dot */}
      <circle 
        cx='100' 
        cy='100' 
        r='8' 
        fill='#000'
        opacity='0.3'
      />
      
      {/* Neural network lines */}
      <g opacity='0.5'>
        <line x1='100' y1='80' x2='100' y2='40' stroke={merged.color} stroke-width='2'/>
        <line x1='115' y1='90' x2='130' y2='70' stroke={merged.color} stroke-width='2'/>
        <line x1='120' y1='100' x2='150' y2='100' stroke={merged.color} stroke-width='2'/>
        <line x1='115' y1='110' x2='130' y2='130' stroke={merged.color} stroke-width='2'/>
        <line x1='100' y1='120' x2='100' y2='160' stroke={merged.color} stroke-width='2'/>
        <line x1='85' y1='110' x2='70' y2='130' stroke={merged.color} stroke-width='2'/>
        <line x1='80' y1='100' x2='50' y2='100' stroke={merged.color} stroke-width='2'/>
        <line x1='85' y1='90' x2='70' y2='70' stroke={merged.color} stroke-width='2'/>
      </g>
      
      {/* Node points */}
      <g>
        <circle cx='100' cy='40' r='5' fill={merged.color}/>
        <circle cx='130' cy='70' r='5' fill={merged.color}/>
        <circle cx='150' cy='100' r='5' fill={merged.color}/>
        <circle cx='130' cy='130' r='5' fill={merged.color}/>
        <circle cx='100' cy='160' r='5' fill={merged.color}/>
        <circle cx='70' cy='130' r='5' fill={merged.color}/>
        <circle cx='50' cy='100' r='5' fill={merged.color}/>
        <circle cx='70' cy='70' r='5' fill={merged.color}/>
      </g>
      
      {/* Gradients */}
      <defs>
        <linearGradient id='gradient1' x1='0%' y1='0%' x2='100%' y2='100%'>
          <stop offset='0%' stop-color={merged.color} stop-opacity='0.8'/>
          <stop offset='100%' stop-color={merged.color} stop-opacity='0.2'/>
        </linearGradient>
        <linearGradient id='gradient2' x1='0%' y1='0%' x2='100%' y2='100%'>
          <stop offset='0%' stop-color={merged.color} stop-opacity='0.3'/>
          <stop offset='50%' stop-color={merged.color} stop-opacity='0.1'/>
          <stop offset='100%' stop-color={merged.color} stop-opacity='0.3'/>
        </linearGradient>
      </defs>
      
      <style>
        {`
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
        `}
      </style>
    </svg>
  );
};


