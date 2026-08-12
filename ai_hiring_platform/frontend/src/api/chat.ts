import { api, API_V1 } from './client';
import type { ChatResponse, CorpusStatus, CorpusSyncResult } from '../types';

/** Ask the recruiter chatbot one question against the whole indexed resume pool. */
export async function askChat(
  message: string,
  session_id: string,
  limit = 5,
  use_llm = true,
  embedding_engine?: string,
): Promise<ChatResponse> {
  const res = await api.post('/chat/query', {
    message, session_id, limit, use_llm, embedding_engine,
  });
  if (res.data && res.data.success) {
    return res.data.data as ChatResponse;
  }
  throw new Error(res.data?.message || 'Chat request failed');
}

/**
 * Ask a question and receive REAL pipeline progress while it runs.
 *
 * Server-Sent Events over POST, so `fetch` + a stream reader rather than EventSource
 * (which is GET-only). `onStage` fires as each actual backend step begins; the promise
 * resolves with the final answer. Falls back to the plain request if streaming is
 * unavailable, so an older backend or a proxy that buffers still works.
 */
export async function askChatStreaming(
  message: string,
  session_id: string,
  onStage: (stage: string, detail: string) => void,
  limit = 5,
  use_llm = true,
  embedding_engine?: string,
): Promise<ChatResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_V1}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id, limit, use_llm, embedding_engine }),
    });
  } catch {
    return askChat(message, session_id, limit, use_llm, embedding_engine);
  }
  if (!response.ok || !response.body) {
    return askChat(message, session_id, limit, use_llm, embedding_engine);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let result: ChatResponse | null = null;

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
      else if (kind === 'result') result = payload as ChatResponse;
      else if (kind === 'error') throw new Error(payload.message || 'The assistant failed.');
    }
  }

  if (!result) throw new Error('The assistant closed the connection without answering.');
  return result;
}

/** Clear the conversation's follow-up memory on the backend. */
export async function resetChat(session_id: string): Promise<void> {
  await api.post('/chat/reset', null, { params: { session_id } });
}

/**
 * What the selected model can currently see, from the shared store.
 *
 * Coverage is per MODEL, not per screen: every section reads the same
 * `documents.db` + `vectors-<model>.faiss`, so a resume indexed here is indexed
 * everywhere. Passing the engine matters — a model whose endpoint was down when
 * some resumes arrived legitimately holds fewer of them.
 */
export async function getCorpusStatus(engine?: string): Promise<CorpusStatus> {
  const res = await api.get('/chat/corpus/status', { params: engine ? { engine } : undefined });
  if (res.data && res.data.success) {
    return res.data.data as CorpusStatus;
  }
  throw new Error(res.data?.message || 'Could not read corpus status');
}

/**
 * Manually top up ONE model against the shared document layer.
 *
 * Uploads index themselves in the background, so this is a catch-up for the case
 * automation cannot cover — a model that was unreachable when its resumes arrived.
 */
export async function syncCorpus(force = false, engine?: string): Promise<CorpusSyncResult> {
  const res = await api.post('/chat/corpus/sync', null, {
    params: engine ? { force, engine } : { force },
  });
  if (res.data && res.data.success) {
    return res.data.data as CorpusSyncResult;
  }
  throw new Error(res.data?.message || 'Corpus sync failed');
}
