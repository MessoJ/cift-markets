import { defineConfig } from 'vite';
import solidPlugin from 'vite-plugin-solid';
import path from 'path';

export default defineConfig({
  plugins: [solidPlugin()],
  resolve: {
    alias: {
      '~': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    strictPort: false,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
  build: {
    target: 'esnext',
    minify: 'esbuild',
    sourcemap: false,
    chunkSizeWarningLimit: 1600,
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-solid': ['solid-js', '@solidjs/router'],
          'vendor-charts': ['echarts'],
          'vendor-ui': ['lucide-solid'],
          'vendor-three': ['three'],
        },
      },
    },
  },
  optimizeDeps: {
    exclude: ['@tauri-apps/api'],
  },
  clearScreen: false,
  envPrefix: ['VITE_', 'TAURI_'],
});
