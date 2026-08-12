import { api } from './client';
import type {
  ComparisonResponse,
  IndexProgressInfo,
  StoreCoverage,
} from '../types';

/** What each model holds, against the one shared document set. */
export async function getStoreCoverage(): Promise<StoreCoverage> {
  const res = await api.get('/embeddings/coverage');
  if (res.data?.success) return res.data.data as StoreCoverage;
  throw new Error(res.data?.message || 'Could not read embedding coverage');
}

/**
 * Start indexing the shared chunk set with the chosen models.
 *
 * Returns as soon as the work is queued — a full pass takes minutes for the larger
 * models, so the caller polls `getIndexProgress` rather than waiting on this.
 */
export async function startIndexing(models: string[], syncDocuments = true): Promise<string[]> {
  const res = await api.post('/embeddings/index', { models, sync_documents: syncDocuments });
  if (res.data?.success) return res.data.data.queued as string[];
  throw new Error(res.data?.message || 'Could not start indexing');
}

export async function getIndexProgress(): Promise<Record<string, IndexProgressInfo>> {
  const res = await api.get('/embeddings/progress');
  if (res.data?.success) return res.data.data as Record<string, IndexProgressInfo>;
  throw new Error(res.data?.message || 'Could not read indexing progress');
}

/** Ask several models the same question and get their answers side by side. */
export async function compareModels(
  message: string,
  models: string[],
  useLlm = false,
  fairMode = true,
  limit = 5,
): Promise<ComparisonResponse> {
  const res = await api.post('/embeddings/compare', {
    message,
    models,
    use_llm: useLlm,
    fair_mode: fairMode,
    limit,
  });
  if (res.data?.success) return res.data.data as ComparisonResponse;
  throw new Error(res.data?.message || 'Comparison failed');
}
