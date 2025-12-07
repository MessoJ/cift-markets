/**
 * useECharts Hook
 * 
 * Advanced ECharts integration for SolidJS with proper lifecycle management.
 * Handles initialization, updates, resizing, and cleanup.
 */

import { onMount, onCleanup, createEffect, on } from 'solid-js';
import * as echarts from 'echarts';
import type { EChartsOption, ECharts } from 'echarts';

export interface UseEChartsOptions {
  /**
   * Chart DOM container reference
   */
  container: HTMLElement | undefined;
  
  /**
   * Initial chart options
   */
  options: EChartsOption;
  
  /**
   * Enable automatic resize on window resize
   */
  autoResize?: boolean;
  
  /**
   * Debounce delay for resize events (ms)
   */
  resizeDebounce?: number;
  
  /**
   * Theme (dark/light)
   */
  theme?: 'dark' | 'light';
  
  /**
   * Loading state
   */
  loading?: boolean;
}

export function useECharts(getOptions: () => UseEChartsOptions) {
  let chartInstance: ECharts | null = null;
  let resizeObserver: ResizeObserver | null = null;
  let resizeTimeout: number | null = null;

  /**
   * Initialize chart instance
   */
  const initChart = () => {
    const opts = getOptions();
    
    if (!opts.container) {
      console.warn('ECharts container not ready');
      return;
    }

    // Dispose existing instance
    if (chartInstance) {
      chartInstance.dispose();
    }

    // Initialize new chart
    chartInstance = echarts.init(opts.container, opts.theme || 'dark', {
      renderer: 'canvas', // Use canvas for better performance with large datasets
      useDirtyRect: true, // Enable dirty rect rendering for better performance
    });

    // Set initial options
    chartInstance.setOption(opts.options, {
      notMerge: false,
      lazyUpdate: false,
      silent: false,
    });

    // Handle loading state
    if (opts.loading) {
      chartInstance.showLoading('default', {
        text: 'Loading...',
        color: '#f97316',
        textColor: '#ffffff',
        maskColor: 'rgba(0, 0, 0, 0.8)',
        zlevel: 0,
      });
    }

    // Setup resize observer if autoResize is enabled
    if (opts.autoResize !== false) {
      setupResizeObserver(opts.container);
    }

    console.info('✅ ECharts initialized');
  };

  /**
   * Update chart options (reactive)
   */
  const updateChart = (options: EChartsOption, merge = true) => {
    if (!chartInstance) {
      console.warn('Chart instance not initialized');
      return;
    }

    chartInstance.setOption(options, {
      notMerge: !merge,
      replaceMerge: merge ? undefined : ['series', 'xAxis', 'yAxis'],
      lazyUpdate: false,
    });
  };

  /**
   * Setup resize observer for responsive charts
   */
  const setupResizeObserver = (container: HTMLElement) => {
    if (typeof ResizeObserver === 'undefined') {
      // Fallback to window resize event
      window.addEventListener('resize', handleResize);
      return;
    }

    resizeObserver = new ResizeObserver(() => {
      handleResize();
    });

    resizeObserver.observe(container);
  };

  /**
   * Handle resize with debouncing
   */
  const handleResize = () => {
    const opts = getOptions();
    const debounce = opts.resizeDebounce || 150;

    if (resizeTimeout) {
      clearTimeout(resizeTimeout);
    }

    resizeTimeout = window.setTimeout(() => {
      if (chartInstance && !chartInstance.isDisposed()) {
        chartInstance.resize({
          animation: {
            duration: 300,
            easing: 'cubicOut',
          },
        });
      }
    }, debounce);
  };

  /**
   * Show loading overlay
   */
  const showLoading = () => {
    if (chartInstance && !chartInstance.isDisposed()) {
      chartInstance.showLoading('default', {
        text: 'Loading chart data...',
        color: '#f97316',
        textColor: '#ffffff',
        maskColor: 'rgba(0, 0, 0, 0.8)',
      });
    }
  };

  /**
   * Hide loading overlay
   */
  const hideLoading = () => {
    if (chartInstance && !chartInstance.isDisposed()) {
      chartInstance.hideLoading();
    }
  };

  /**
   * Get chart instance (for advanced operations)
   */
  const getInstance = (): ECharts | null => {
    return chartInstance;
  };

  /**
   * Dispose chart and cleanup
   */
  const dispose = () => {
    if (resizeTimeout) {
      clearTimeout(resizeTimeout);
      resizeTimeout = null;
    }

    if (resizeObserver) {
      resizeObserver.disconnect();
      resizeObserver = null;
    }

    window.removeEventListener('resize', handleResize);

    if (chartInstance && !chartInstance.isDisposed()) {
      chartInstance.dispose();
      chartInstance = null;
      console.info('✅ ECharts disposed');
    }
  };

  // Mount lifecycle
  onMount(() => {
    const opts = getOptions();
    if (opts.container) {
      initChart();
    }
  });

  // Reactive update when options change
  createEffect(
    on(
      () => getOptions().options,
      (options) => {
        if (chartInstance && !chartInstance.isDisposed()) {
          updateChart(options);
        }
      },
      { defer: true }
    )
  );

  // Reactive loading state
  createEffect(
    on(
      () => getOptions().loading,
      (loading) => {
        if (loading) {
          showLoading();
        } else {
          hideLoading();
        }
      }
    )
  );

  // Cleanup
  onCleanup(() => {
    dispose();
  });

  return {
    getInstance,
    updateChart,
    showLoading,
    hideLoading,
    dispose,
  };
}
