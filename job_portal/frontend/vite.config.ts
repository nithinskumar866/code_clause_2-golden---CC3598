import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // 5174, not Vite's default 5173: the recruiter platform's own frontend already
  // uses 5173, and both are expected to run at once during development. The API's
  // CORS allow-list has both.
  server: { port: 5174 },
})
