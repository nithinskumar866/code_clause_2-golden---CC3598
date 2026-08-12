import { useCallback, useEffect, useRef, useState, type FC, type FormEvent } from 'react';
import {
  FlaskConical, RefreshCw, Send, Database, AlertTriangle, CheckCircle2,
  Sparkles, Scale, Layers,
} from 'lucide-react';
import type { ComparisonResponse, ModelAnswer, StoreCoverage } from '../../types';
import { compareModels, getStoreCoverage, startIndexing, getIndexProgress } from '../../api/embeddings';
import { PageHeader } from '../../components/ui/PageHeader';
import { Card } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Badge } from '../../components/ui/Badge';
import { Spinner } from '../../components/ui/Spinner';

/** Questions that separate the models rather than ones every model answers alike. */
const SAMPLE_QUESTIONS = [
  'who has kubernetes in real projects',
  'who is strongest in cloud infrastructure',
  'candidates who led a migration',
  'who has 10+ years in java',
];

const MODEL_BLURB: Record<string, string> = {
  bge: 'BGE-large · 1024d · local, accurate',
  mxbai: 'mxbai-large · 1024d · local, heavier',
  gpu: 'nomic · 768d · remote GPU',
};

/**
 * Side-by-side comparison of the embedding models, over one shared question.
 *
 * WHY THIS IS ITS OWN PAGE
 * The Recruiter Assistant answers with ONE model, which is what a recruiter wants. The
 * question "is the bigger model actually better on our resumes?" is a different job:
 * it needs the same question sent to every model, over the same candidates, with the
 * differences measured rather than eyeballed. Mixing the two would make the assistant
 * slower and noisier for the people who just want an answer.
 *
 * Two things keep the comparison honest, and both are visible on screen:
 *  - **Equal population.** Fair mode confines every model to the resumes they have ALL
 *    indexed. Without it a model wins by having seen more resumes, which measures
 *    indexing coverage rather than retrieval quality.
 *  - **Retrieval first.** The deterministic answer is always shown; the LLM summary is
 *    an opt-in extra, because its wording varies run to run and would blur the very
 *    difference this page exists to show.
 */
export const ModelLab: FC = () => {
  const [coverage, setCoverage] = useState<StoreCoverage | null>(null);
  const [selected, setSelected] = useState<string[]>(['bge', 'mxbai', 'gpu']);
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState<ComparisonResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [useLlm, setUseLlm] = useState(false);
  const [fairMode, setFairMode] = useState(true);
  const [indexing, setIndexing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  const loadCoverage = useCallback(async () => {
    try {
      setCoverage(await getStoreCoverage());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not read embedding coverage.');
    }
  }, []);

  useEffect(() => {
    void loadCoverage();
  }, [loadCoverage]);

  // While anything is indexing, poll. Indexing a large model over a full pool takes
  // minutes, so the operator needs to watch it rather than guess when it is done.
  useEffect(() => {
    const active = coverage?.models.some(
      (m) => m.progress.state === 'running' || m.progress.state === 'queued',
    );
    if (!active) {
      if (pollRef.current) window.clearInterval(pollRef.current);
      pollRef.current = null;
      return;
    }
    if (pollRef.current) return;
    pollRef.current = window.setInterval(() => {
      void getIndexProgress()
        .then(() => loadCoverage())
        .catch(() => undefined);
    }, 1500);
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
      pollRef.current = null;
    };
  }, [coverage, loadCoverage]);

  const toggleModel = (name: string) =>
    setSelected((s) => (s.includes(name) ? s.filter((m) => m !== name) : [...s, name]));

  const onIndex = async (models: string[]) => {
    if (!models.length) return;
    setIndexing(true);
    setError(null);
    try {
      await startIndexing(models, true);
      await loadCoverage();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not start indexing.');
    } finally {
      setIndexing(false);
    }
  };

  const onAsk = async (e?: FormEvent) => {
    e?.preventDefault();
    const q = question.trim();
    if (!q || !selected.length || running) return;
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      setResult(await compareModels(q, selected, useLlm, fairMode));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Comparison failed.');
    } finally {
      setRunning(false);
    }
  };

  const engines = coverage?.engines ?? [];
  const aligned = coverage?.models_aligned ?? false;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Model Lab"
        description="Ask all three embedding models the same question, over the same candidates, and see where they disagree."
        icon={<FlaskConical className="h-5 w-5" />}
      />

      {error && (
        <div className="rounded-lg border border-rose-500/20 bg-rose-500/10 px-4 py-2.5 text-xs text-rose-300">
          {error}
        </div>
      )}

      {/* --- Coverage: one document set, three models indexing it -------------- */}
      <Card className="space-y-4 p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-white">
            <Database className="h-4 w-4 text-indigo-400" />
            Shared embedding store
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={() => void loadCoverage()}>
              <RefreshCw className="h-3.5 w-3.5" /> Refresh
            </Button>
            <Button
              size="sm"
              loading={indexing}
              disabled={!selected.length}
              onClick={() => void onIndex(selected)}
            >
              <Layers className="h-3.5 w-3.5" />
              Index selected ({selected.length})
            </Button>
          </div>
        </div>

        {coverage && (
          <>
            <p className="text-xs text-gray-400">
              <strong className="text-white">{coverage.documents.resumes}</strong> resumes parsed into{' '}
              <strong className="text-white">{coverage.documents.chunks}</strong> chunks. Every model
              indexes this same chunk set — that is what makes the comparison a comparison.
              {coverage.database_resumes !== null && !coverage.documents_in_sync && (
                <span className="ml-1 text-amber-400">
                  {coverage.database_resumes - coverage.documents.resumes} uploaded resume(s) are not
                  parsed yet.
                </span>
              )}
            </p>

            <div className="grid gap-3 sm:grid-cols-3">
              {engines.map((engine) => {
                const cov = coverage.models.find((m) => m.model === engine.name);
                const busy =
                  cov?.progress.state === 'running' || cov?.progress.state === 'queued';
                const isSelected = selected.includes(engine.name);
                return (
                  <button
                    key={engine.name}
                    type="button"
                    onClick={() => toggleModel(engine.name)}
                    className={`rounded-xl border p-3 text-left transition ${
                      isSelected
                        ? 'border-indigo-500/50 bg-indigo-500/10'
                        : 'border-white/10 bg-white/[0.02] hover:border-white/20'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-sm font-semibold text-white">{engine.name}</span>
                      {engine.available ? (
                        <Badge tone="success">online</Badge>
                      ) : (
                        <Badge tone="warning">unreachable</Badge>
                      )}
                    </div>
                    <p className="mt-0.5 text-[11px] text-gray-500">
                      {MODEL_BLURB[engine.name] || engine.model}
                    </p>

                    <div className="mt-2 text-[11px] text-gray-400">
                      <span className="text-white">{cov?.indexed_resumes ?? 0}</span> resumes ·{' '}
                      {cov?.indexed_chunks ?? 0} chunks
                    </div>

                    {busy ? (
                      <div className="mt-2">
                        <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/10">
                          <div
                            className="h-full bg-indigo-500 transition-all"
                            style={{ width: `${cov?.progress.percent ?? 0}%` }}
                          />
                        </div>
                        <p className="mt-1 text-[10px] text-indigo-300">
                          {cov?.progress.state === 'queued'
                            ? 'queued…'
                            : `indexing ${cov?.progress.done}/${cov?.progress.total} · ${Math.round(
                                cov?.progress.elapsed_seconds ?? 0,
                              )}s`}
                        </p>
                      </div>
                    ) : cov?.progress.state === 'error' ? (
                      <p className="mt-2 text-[10px] text-rose-400">{cov.progress.error}</p>
                    ) : cov?.in_sync ? (
                      <p className="mt-2 inline-flex items-center gap-1 text-[10px] text-emerald-400">
                        <CheckCircle2 className="h-3 w-3" /> up to date
                      </p>
                    ) : (
                      <p className="mt-2 inline-flex items-center gap-1 text-[10px] text-amber-400">
                        <AlertTriangle className="h-3 w-3" /> behind — index to catch up
                      </p>
                    )}
                  </button>
                );
              })}
            </div>

            <div
              className={`rounded-lg border px-3 py-2 text-xs ${
                aligned
                  ? 'border-emerald-500/20 bg-emerald-500/5 text-emerald-300'
                  : 'border-amber-500/20 bg-amber-500/5 text-amber-300'
              }`}
            >
              {aligned ? (
                <>
                  All models have indexed the same {coverage.comparable_resumes} resumes — a
                  comparison here is like-for-like.
                </>
              ) : (
                <>
                  The models have indexed different resumes. Only{' '}
                  <strong>{coverage.comparable_resumes}</strong> are common to all of them; fair mode
                  restricts the comparison to those so a model cannot win simply by having seen more.
                </>
              )}
            </div>
          </>
        )}
      </Card>

      {/* --- One question, every model ---------------------------------------- */}
      <Card className="space-y-3 p-4">
        <form onSubmit={onAsk} className="flex items-center gap-2">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask one question — every selected model answers it"
            aria-label="Question for all models"
            disabled={running}
            className="min-w-0 flex-1 rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-white placeholder:text-gray-600 focus:border-indigo-500 focus:outline-none"
          />
          <Button type="submit" loading={running} disabled={!question.trim() || !selected.length}>
            <Send className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Compare</span>
          </Button>
        </form>

        <div className="flex flex-wrap items-center gap-3 text-[11px] text-gray-400">
          <label className="inline-flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={fairMode}
              onChange={(e) => setFairMode(e.target.checked)}
              className="accent-indigo-500"
            />
            Fair mode — only resumes every model has indexed
          </label>
          <label className="inline-flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={useLlm}
              onChange={(e) => setUseLlm(e.target.checked)}
              className="accent-indigo-500"
            />
            Also write each model's answer with the LLM (slower, wording varies)
          </label>
        </div>

        <div className="flex flex-wrap gap-2">
          {SAMPLE_QUESTIONS.map((q) => (
            <button
              key={q}
              type="button"
              disabled={running}
              onClick={() => {
                setQuestion(q);
                void compareModels(q, selected, useLlm, fairMode)
                  .then(setResult)
                  .catch((e) => setError(e instanceof Error ? e.message : 'Comparison failed.'));
              }}
              className="rounded-lg border border-white/10 px-2.5 py-1 text-[11px] text-gray-300 transition hover:border-indigo-500/40 hover:bg-indigo-500/10 hover:text-white disabled:opacity-50"
            >
              {q}
            </button>
          ))}
        </div>
      </Card>

      {running && (
        <div className="flex items-center gap-2 pl-1 text-xs text-gray-400">
          <Spinner />
          <span aria-live="polite">Asking {selected.length} models the same question…</span>
        </div>
      )}

      {/* --- Agreement: the number that makes this a comparison ---------------- */}
      {result && (
        <>
          {result.warnings.map((w) => (
            <div
              key={w}
              className="flex items-start gap-2 rounded-lg border border-amber-500/20 bg-amber-500/5 px-3 py-2 text-xs text-amber-300"
            >
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              {w}
            </div>
          ))}

          <Card className="flex flex-wrap items-center gap-x-6 gap-y-2 p-3 text-xs">
            <span className="inline-flex items-center gap-1.5 font-semibold text-white">
              <Scale className="h-4 w-4 text-indigo-400" /> Agreement
            </span>
            <span className={result.agreement.top1_unanimous ? 'text-emerald-400' : 'text-amber-400'}>
              {result.agreement.top1_unanimous
                ? 'All models picked the same top candidate'
                : 'Models disagree on the top candidate'}
            </span>
            {Object.entries(result.agreement.overlap).map(([pair, score]) => (
              <span key={pair} className="text-gray-400">
                {pair.replace('|', ' ↔ ')}:{' '}
                <strong className="text-white">{Math.round(score * 100)}%</strong> overlap
              </span>
            ))}
            <span className="text-gray-500">
              over {result.compared_over_resumes} resumes
              {result.fair_mode ? ' (fair mode)' : ''}
            </span>
          </Card>

          {/* --- One column per model ------------------------------------------ */}
          <div className="grid gap-3 lg:grid-cols-3">
            {result.models.map((m) => (
              <ModelColumn key={m.model} answer={m} unanimous={result.agreement.top1_unanimous} />
            ))}
          </div>
        </>
      )}
    </div>
  );
};

const ModelColumn: FC<{ answer: ModelAnswer; unanimous: boolean }> = ({ answer, unanimous }) => (
  <Card className="flex flex-col gap-3 p-4">
    <div className="flex items-center justify-between gap-2 border-b border-white/5 pb-2">
      <div>
        <h3 className="text-sm font-semibold text-white">{answer.model}</h3>
        <p className="text-[10px] text-gray-500">
          {MODEL_BLURB[answer.model]} · {answer.indexed_resumes} indexed
        </p>
      </div>
      <div className="text-right">
        <span className="text-[11px] font-semibold text-indigo-300">{answer.elapsed_ms} ms</span>
        {!answer.available && <Badge tone="warning">offline</Badge>}
      </div>
    </div>

    {answer.error ? (
      <p className="text-xs text-amber-300">{answer.error}</p>
    ) : (
      <>
        <p className="text-xs leading-relaxed text-gray-300">{answer.answer}</p>

        {answer.llm_answer && (
          <div className="rounded-lg border border-violet-500/20 bg-violet-500/5 p-2.5">
            <span className="mb-1 inline-flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wide text-violet-300">
              <Sparkles className="h-3 w-3" /> LLM answer
            </span>
            <p className="text-[11px] leading-relaxed text-violet-100">{answer.llm_answer}</p>
          </div>
        )}

        <div className="space-y-2">
          {answer.candidates.length === 0 && (
            <p className="text-[11px] text-gray-500">No candidate matched.</p>
          )}
          {answer.candidates.map((c, i) => (
            <div
              key={c.resume_id}
              className={`rounded-lg border p-2.5 ${
                i === 0 && !unanimous
                  ? 'border-indigo-500/30 bg-indigo-500/5'
                  : 'border-white/10 bg-white/[0.02]'
              }`}
            >
              <div className="flex items-baseline justify-between gap-2">
                <span className="truncate text-xs font-semibold text-white">
                  {i + 1}. {c.name || `Resume ${c.resume_id}`}
                </span>
                <span className="shrink-0 text-[11px] text-gray-400">{c.match_percentage}%</span>
              </div>
              {c.matched_skills.length > 0 && (
                <p className="mt-0.5 text-[10px] text-emerald-400">
                  proven: {c.matched_skills.join(', ')}
                </p>
              )}
              {c.missing_skills.length > 0 && (
                <p className="text-[10px] text-gray-500">no evidence: {c.missing_skills.join(', ')}</p>
              )}
              {c.top_evidence && (
                <blockquote className="mt-1 border-l-2 border-white/10 pl-2 text-[10px] leading-relaxed text-gray-500">
                  <span className="mr-1 font-semibold uppercase text-gray-600">{c.section}</span>
                  {c.top_evidence.slice(0, 160)}
                  {c.top_evidence.length > 160 && '…'}
                </blockquote>
              )}
            </div>
          ))}
        </div>
      </>
    )}
  </Card>
);
