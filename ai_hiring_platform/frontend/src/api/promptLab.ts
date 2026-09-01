import { api } from './client';
import type {
  PromptCase,
  PromptRunDetail,
  PromptRunResult,
  PromptRunSummary,
  PromptRuleInfo,
  PromptScoreResult,
  PromptSuite,
  PromptVariant,
  StarterSuite,
} from '../types';

/** Every compliance rule the lab can apply, with what each one measures. */
export async function getRules(): Promise<PromptRuleInfo[]> {
  const res = await api.get('/prompt-lab/rules');
  if (res.data?.success) return res.data.data as PromptRuleInfo[];
  throw new Error(res.data?.message || 'Could not load the rule catalog');
}

/** The prompt in the field today, with the turns that exercise each of its clauses. */
export async function getStarterSuite(): Promise<StarterSuite> {
  const res = await api.get('/prompt-lab/starter');
  if (res.data?.success) return res.data.data as StarterSuite;
  throw new Error(res.data?.message || 'Could not load the starter suite');
}

/**
 * Run one or two prompts over the same cases and grade both.
 *
 * `llm_available: false` comes back as a successful response with no answers — the lab
 * reports a missing LLM rather than faking one, because an invented answer would make a
 * prompt look compliant when it was never exercised.
 */
export async function runPrompts(
  variants: PromptVariant[],
  cases: PromptCase[],
  options: { temperature?: number; suiteId?: number | null; persist?: boolean } = {},
): Promise<PromptRunResult> {
  const res = await api.post('/prompt-lab/run', {
    variants,
    cases,
    temperature: options.temperature ?? 0.2,
    suite_id: options.suiteId ?? null,
    persist: options.persist ?? true,
  });
  if (res.data?.success) return res.data.data as PromptRunResult;
  throw new Error(res.data?.message || 'The prompt run failed');
}

/** Grade an answer that already exists — no LLM needed, so it works with no key set. */
export async function scoreAnswer(
  prompt: string,
  promptCase: PromptCase,
  answer: string,
): Promise<PromptScoreResult> {
  const res = await api.post('/prompt-lab/score', { prompt, case: promptCase, answer });
  if (res.data?.success) return res.data.data as PromptScoreResult;
  throw new Error(res.data?.message || 'Could not grade the answer');
}

export async function listSuites(): Promise<PromptSuite[]> {
  const res = await api.get('/prompt-lab/suites');
  if (res.data?.success) return res.data.data as PromptSuite[];
  throw new Error(res.data?.message || 'Could not load saved suites');
}

export async function createSuite(
  name: string,
  prompt: string,
  cases: PromptCase[],
  description = '',
): Promise<PromptSuite> {
  const res = await api.post('/prompt-lab/suites', { name, description, prompt, cases });
  if (res.data?.success) return res.data.data as PromptSuite;
  throw new Error(res.data?.message || 'Could not save the suite');
}

export async function updateSuite(
  id: number,
  patch: Partial<Pick<PromptSuite, 'name' | 'description' | 'prompt' | 'cases'>>,
): Promise<PromptSuite> {
  const res = await api.put(`/prompt-lab/suites/${id}`, patch);
  if (res.data?.success) return res.data.data as PromptSuite;
  throw new Error(res.data?.message || 'Could not update the suite');
}

export async function deleteSuite(id: number): Promise<void> {
  const res = await api.delete(`/prompt-lab/suites/${id}`);
  if (!res.data?.success) throw new Error(res.data?.message || 'Could not delete the suite');
}

/** Newest first — the regression history a prompt edit is judged against. */
export async function listRuns(suiteId?: number | null, limit = 50): Promise<PromptRunSummary[]> {
  const res = await api.get('/prompt-lab/runs', {
    params: { ...(suiteId ? { suite_id: suiteId } : {}), limit },
  });
  if (res.data?.success) return res.data.data as PromptRunSummary[];
  throw new Error(res.data?.message || 'Could not load the run history');
}

export async function getRun(id: number): Promise<PromptRunDetail> {
  const res = await api.get(`/prompt-lab/runs/${id}`);
  if (res.data?.success) return res.data.data as PromptRunDetail;
  throw new Error(res.data?.message || 'Could not load the run');
}
