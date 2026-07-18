import axios from 'axios';

/**
 * Single source of truth for the backend base URL. Previously the literal
 * `http://localhost:8000` was duplicated across five files; changing the host
 * now means editing one place (or setting `VITE_API_BASE` at build time).
 *
 * `import.meta.env.VITE_API_BASE` wins when provided, so the same bundle can
 * target a deployed backend without code changes; it falls back to the local
 * dev server.
 */
export const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, '') ??
  'http://localhost:8000';

/** Full `/api/v1` root, e.g. `${API_V1}/analysis/evaluate`. */
export const API_V1 = `${API_BASE}/api/v1`;

/**
 * Shared axios instance pre-pointed at `/api/v1`. Endpoints are passed as
 * relative paths (e.g. `api.get('/resume')`). Kept minimal — the response
 * envelope (`{success, message, data, meta}`) is unwrapped at each call site,
 * exactly as the existing code does.
 */
export const api = axios.create({ baseURL: API_V1 });
