import type {
  WebChatResponse,
  WebCompany,
  WebCompanyStatus,
  WebHealth,
} from '../types';

/**
 * Client for the `company_intel` crawler service.
 *
 * This talks to a DIFFERENT PROCESS than every other file in `api/` — a standalone
 * service that crawls company websites into its own Qdrant collections. It deliberately
 * does not use the shared axios instance from `client.ts`: that one is pinned to the
 * hiring backend, and routing these calls through it would mean adding a proxy there
 * and tying the two projects together. The whole point of company_intel is that it
 * stands alone.
 *
 * Base URL comes from `VITE_COMPANY_INTEL_URL`, defaulting to the local dev port. The
 * service sends permissive CORS headers, so the browser reaches it directly.
 */
const _explicit = (import.meta.env.VITE_COMPANY_INTEL_URL as string | undefined)?.replace(
  /\/$/,
  '',
);
export const WEB_BASE: string = _explicit || 'http://localhost:8123';
export const WEB_V1 = `${WEB_BASE}/api/v1`;

/** The service's `{success, message, data}` envelope. */
interface Envelope<T> {
  success: boolean;
  message: string;
  data: T | null;
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${WEB_V1}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...init,
    });
  } catch {
    // A network-level failure here almost always means the service is not running,
    // which is a different problem from a bad request — say so rather than surfacing
    // "Failed to fetch".
    throw new Error(
      `Cannot reach the company crawler at ${WEB_BASE}. Start it with run.ps1 in company_intel/.`,
    );
  }

  let body: Envelope<T> | { detail?: string } | null = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }

  if (!response.ok) {
    const detail = (body as { detail?: string } | null)?.detail;
    throw new Error(detail || `The crawler returned ${response.status}.`);
  }

  const envelope = body as Envelope<T> | null;
  if (envelope && envelope.data !== undefined && envelope.data !== null) {
    return envelope.data;
  }
  throw new Error(envelope?.message || 'The crawler returned an empty response.');
}

/** Is the crawler running, and what does its index hold? */
export function getWebHealth(): Promise<WebHealth> {
  return call<WebHealth>('/health');
}

/** Every registered company. */
export function listWebCompanies(): Promise<WebCompany[]> {
  return call<WebCompany[]>('/companies');
}

/** What has actually been crawled for one company. */
export function getWebCompanyStatus(companyId: string): Promise<WebCompanyStatus> {
  return call<WebCompanyStatus>(`/companies/${encodeURIComponent(companyId)}`);
}

export interface RegisterWebCompanyInput {
  name: string;
  domain: string;
  linkedin_urls?: string[];
  deny_patterns?: string[];
}

/** Add (or update) a company. Idempotent on the domain. */
export function registerWebCompany(input: RegisterWebCompanyInput): Promise<WebCompany> {
  return call<WebCompany>('/companies', {
    method: 'POST',
    body: JSON.stringify({
      name: input.name,
      domain: input.domain,
      linkedin_urls: input.linkedin_urls ?? [],
      deny_patterns: input.deny_patterns ?? [],
      seed_urls: [],
      allow_patterns: [],
      enabled: true,
    }),
  });
}

/**
 * Start a crawl. Returns as soon as the worker thread starts, not when it finishes —
 * a full crawl takes minutes, and holding an HTTP request open for that is how a
 * browser timeout turns into a half-indexed company.
 */
export function startWebCrawl(companyId: string, maxPages = 20): Promise<{ started: boolean }> {
  return call<{ started: boolean }>(`/companies/${encodeURIComponent(companyId)}/crawl`, {
    method: 'POST',
    body: JSON.stringify({ max_pages: maxPages, background: true, force: false }),
  });
}

/** Remove a company and everything crawled for it. */
export function deleteWebCompany(companyId: string): Promise<Record<string, number>> {
  return call<Record<string, number>>(`/companies/${encodeURIComponent(companyId)}`, {
    method: 'DELETE',
  });
}

/** Ask one question against the crawled pages. */
export function askWeb(
  message: string,
  companyId?: string,
  limit = 8,
): Promise<WebChatResponse> {
  return call<WebChatResponse>('/chat/query', {
    method: 'POST',
    body: JSON.stringify({
      message,
      company_id: companyId ?? null,
      limit,
      use_llm: true,
    }),
  });
}

/**
 * Ask a question and receive real pipeline progress while it runs.
 *
 * Server-Sent Events over POST, so `fetch` + a stream reader rather than EventSource
 * (which is GET-only). Mirrors `askCompanyStreaming` frame for frame so all three chat
 * modes behave identically to the user. Falls back to the plain request whenever
 * streaming is unavailable — the answer is the same either way, only the progress
 * display is lost.
 */
export async function askWebStreaming(
  message: string,
  onStage: (stage: string, detail: string) => void,
  companyId?: string,
  limit = 8,
): Promise<WebChatResponse> {
  let response: Response;
  try {
    response = await fetch(`${WEB_V1}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        company_id: companyId ?? null,
        limit,
        use_llm: true,
      }),
    });
  } catch {
    return askWeb(message, companyId, limit);
  }
  if (!response.ok || !response.body) {
    return askWeb(message, companyId, limit);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let result: WebChatResponse | null = null;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line; keep any partial frame buffered.
    const frames = buffer.split('\n\n');
    buffer = frames.pop() ?? '';

    for (const frame of frames) {
      const lines = frame.split('\n');
      const eventLine = lines.find((l) => l.startsWith('event:'));
      const dataLine = lines.find((l) => l.startsWith('data:'));
      if (!dataLine) continue;
      const payload = JSON.parse(dataLine.slice(5).trim());
      const kind = eventLine?.slice(6).trim();
      if (kind === 'stage') onStage(payload.stage, payload.detail);
      else if (kind === 'result') result = payload as WebChatResponse;
      else if (kind === 'error') throw new Error(payload.message || 'The crawler failed.');
    }
  }

  if (!result) throw new Error('The crawler closed the connection without answering.');
  return result;
}
