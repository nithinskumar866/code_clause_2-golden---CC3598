import { api, API_V1 } from './client';
import type { CompanyChatResponse, CompanyStoreStatus } from '../types';

/**
 * Client for the company knowledge base (`/api/v1/company-chat`).
 *
 * Kept separate from `api/chat.ts` because it talks to a different corpus with a
 * different response shape. The streaming helper mirrors `askChatStreaming` exactly —
 * same SSE framing, same fall back to the plain request — so both chat modes behave
 * identically to the user even though nothing is shared between the two pipelines.
 */

/** Ask one question against the company database. */
export async function askCompany(
  message: string,
  session_id: string,
  limit = 5,
  use_llm = true,
): Promise<CompanyChatResponse> {
  const res = await api.post('/company-chat/query', { message, session_id, limit, use_llm });
  if (res.data && res.data.success) {
    return res.data.data as CompanyChatResponse;
  }
  throw new Error(res.data?.message || 'Company request failed');
}

/**
 * Ask a question and receive real pipeline progress while it runs.
 *
 * Server-Sent Events over POST, so `fetch` + a stream reader rather than EventSource
 * (which is GET-only). Falls back to the plain request when streaming is unavailable.
 */
export async function askCompanyStreaming(
  message: string,
  session_id: string,
  onStage: (stage: string, detail: string) => void,
  limit = 5,
  use_llm = true,
): Promise<CompanyChatResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_V1}/company-chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id, limit, use_llm }),
    });
  } catch {
    return askCompany(message, session_id, limit, use_llm);
  }
  if (!response.ok || !response.body) {
    return askCompany(message, session_id, limit, use_llm);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let result: CompanyChatResponse | null = null;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line; keep any partial frame buffered.
    const frames = buffer.split('\n\n');
    buffer = frames.pop() ?? '';

    for (const frame of frames) {
      const eventLine = frame.split('\n').find((l) => l.startsWith('event:'));
      const dataLine = frame.split('\n').find((l) => l.startsWith('data:'));
      if (!dataLine) continue;
      const payload = JSON.parse(dataLine.slice(5).trim());
      const kind = eventLine?.slice(6).trim();
      if (kind === 'stage') onStage(payload.stage, payload.detail);
      else if (kind === 'result') result = payload as CompanyChatResponse;
      else if (kind === 'error') throw new Error(payload.message || 'The assistant failed.');
    }
  }

  if (!result) throw new Error('The assistant closed the connection without answering.');
  return result;
}

/** Clear the company conversation's follow-up memory on the backend. */
export async function resetCompanyChat(session_id: string): Promise<void> {
  await api.post('/company-chat/reset', null, { params: { session_id } });
}

/**
 * Whether the company database is configured, reachable and populated.
 *
 * Read before offering the toggle, so a recruiter is told the store is empty rather
 * than being handed an assistant that answers nothing.
 */
export async function getCompanyStatus(): Promise<CompanyStoreStatus> {
  const res = await api.get('/company-chat/status');
  if (res.data && res.data.success) {
    return res.data.data as CompanyStoreStatus;
  }
  throw new Error(res.data?.message || 'Could not read company store status');
}
