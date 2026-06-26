import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    // Proxy all /api requests to the FastAPI backend.
    // This eliminates CORS entirely — the browser sees one origin.
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        // Keep the /api prefix — FastAPI routes start with /api/v1/
      },
    },
  },
})
