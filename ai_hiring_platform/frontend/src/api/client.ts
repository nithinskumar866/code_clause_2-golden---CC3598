import axios from 'axios';

/**
 * Single source of truth for the backend base URL.
 *
 * Resolution order:
 *   1. An explicit `VITE_API_BASE` (set at build time) always wins — point the
 *      same bundle at any deployed backend.
 *   2. Otherwise: in a dev build use the local backend (`http://localhost:8000`);
 *      in a production build use **same-origin** (empty base → `/api/v1`), which is
 *      how the single-container deploy serves the API alongside the frontend.
 */
const _explicit = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, '');
export const API_BASE: string =
  _explicit !== undefined && _explicit !== ''
    ? _explicit
    : import.meta.env.DEV
      ? 'http://localhost:8000'
      : '';

/** Full `/api/v1` root, e.g. `${API_V1}/analysis/evaluate`. */
export const API_V1 = `${API_BASE}/api/v1`;

/**
 * Shared axios instance pre-pointed at `/api/v1`. Endpoints are passed as
 * relative paths (e.g. `api.get('/resume')`). Kept minimal — the response
 * envelope (`{success, message, data, meta}`) is unwrapped at each call site,
 * exactly as the existing code does.
 */
export const api = axios.create({ baseURL: API_V1 });
