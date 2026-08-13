import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/query':       { target: 'http://localhost:8000', changeOrigin: true },
      '/ingest':      { target: 'http://localhost:8000', changeOrigin: true },
      '/documents':   { target: 'http://localhost:8000', changeOrigin: true },
      '/graph':       { target: 'http://localhost:8000', changeOrigin: true },
      '/health':      { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
