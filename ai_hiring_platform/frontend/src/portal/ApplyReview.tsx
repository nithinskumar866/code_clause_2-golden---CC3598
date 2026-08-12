import { useEffect, useState, type FC } from 'react';
import { AlertTriangle, Check, Send, Sparkles, X } from 'lucide-react';
import { applicationApi } from './api';
import { Working, SkeletonRows } from './Working';
import type { ApplicationPreview, ApplySubmitResult } from './types';

interface ApplyReviewProps {
  resumeId: number;
  /** Restrict to specific postings; omitted means "everything above the floor". */
  jobIds?: number[];
  minFitScore?: number;
  onClose: () => void;
  onSubmitted?: (result: ApplySubmitResult) => void;
}

/**
 * The review step, and the only way an application leaves this app.
 *
 * Bulk apply without this screen is a button that writes to strangers on the
 * candidate's behalf. So the expensive half runs first and stores nothing: every
 * posting is listed with the letter that would be sent, the fit it would be
 * judged on, and whether this CV has already been there. Each one can be
 * deselected or its letter edited, and the count on the button is the count that
 * gets sent.
 *
 * Postings already applied to are shown, greyed and unselectable, rather than
 * filtered out — "3 of these 5 already went" is something the candidate needs to
 * know, and a silently shorter list reads as a bug.
 */
export const ApplyReview: FC<ApplyReviewProps> = ({
  resumeId, jobIds, minFitScore, onClose, onSubmitted,
}) => {
  const [preview, setPreview] = useState<ApplicationPreview | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ApplySubmitResult | null>(null);

  /** Job ids the candidate still intends to send. */
  const [selected, setSelected] = useState<Set<number>>(new Set());
  /** Edited letters, keyed by job id. */
  const [letters, setLetters] = useState<Record<number, string>>({});
  const [editing, setEditing] = useState<number | null>(null);
  /** Job ids whose letter the model is currently writing. */
  const [drafting, setDrafting] = useState<Set<number>>(new Set());
  /** Job ids whose letter the model has written, so the badge can say so. */
  const [written, setWritten] = useState<Set<number>>(new Set());

  /**
   * Upgrades one composed letter to a written one.
   *
   * Per role rather than all at once: on a local model each letter takes tens of
   * seconds, and the panel has to stay usable while one is being written.
   */
  const draft = async (jobId: number) => {
    setDrafting(previous => new Set(previous).add(jobId));
    try {
      const result = await applicationApi.draftLetter(resumeId, jobId);
      setLetters(previous => ({ ...previous, [jobId]: result.coverLetter }));
      if (result.letterMode === 'llm') setWritten(previous => new Set(previous).add(jobId));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'That letter could not be written.');
    } finally {
      setDrafting(previous => {
        const next = new Set(previous);
        next.delete(jobId);
        return next;
      });
    }
  };

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    applicationApi.preview({ resumeId, jobIds, minFitScore })
      .then(data => {
        if (cancelled) return;
        setPreview(data);
        // Everything eligible starts selected; duplicates can never be.
        setSelected(new Set(data.items.filter(i => !i.alreadyApplied).map(i => i.job.id)));
        setLetters(Object.fromEntries(data.items.map(i => [i.job.id, i.coverLetter])));
      })
      .catch(e => { if (!cancelled) setError(e instanceof Error ? e.message : 'Could not prepare the applications.'); })
      .finally(() => { if (!cancelled) setLoading(false); });

    return () => { cancelled = true; };
  }, [resumeId, jobIds, minFitScore]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const toggle = (jobId: number) => setSelected(previous => {
    const next = new Set(previous);
    if (next.has(jobId)) next.delete(jobId); else next.add(jobId);
    return next;
  });

  const submit = async () => {
    if (!preview || selected.size === 0) return;
    setSubmitting(true);
    setError(null);
    try {
      const outcome = await applicationApi.submit(
        resumeId,
        [...selected].map(jobId => ({ jobId, coverLetter: letters[jobId] })),
      );
      setResult(outcome);
      onSubmitted?.(outcome);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'The applications could not be sent.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-[60] flex items-end justify-center bg-black/70 backdrop-blur-sm sm:items-center"
      onClick={onClose}
      role="presentation"
    >
      <section
        role="dialog"
        aria-modal="true"
        aria-label="Review applications"
        onClick={event => event.stopPropagation()}
        className="flex max-h-[92vh] w-full max-w-2xl flex-col rounded-t-2xl border border-white/10 bg-card shadow-2xl sm:rounded-2xl"
      >
        <header className="flex shrink-0 items-start gap-3 border-b border-border px-5 py-4">
          <div className="min-w-0 flex-1">
            <h2 className="text-base font-semibold text-white">
              {result ? 'Applications sent' : 'Review before applying'}
            </h2>
            <p className="mt-0.5 text-xs text-gray-400">
              {result
                ? 'Nothing further will be sent.'
                : 'Nothing is sent until you confirm. Edit or deselect anything below.'}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded-lg p-1.5 text-gray-400 transition hover:bg-white/5 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
          >
            <X className="h-4 w-4" />
          </button>
        </header>

        <div className="flex-1 space-y-3 overflow-y-auto px-5 py-4">
          {loading && (
            <>
              <Working
                label="Scoring your CV against these roles…"
                detail="Letters are ready immediately; you can have the model rewrite any of them once they appear."
              />
              <SkeletonRows rows={Math.min(jobIds?.length ?? 2, 3)} />
            </>
          )}

          {error && (
            <p className="rounded-lg border border-rose-500/20 bg-rose-500/10 p-3 text-xs text-rose-300">{error}</p>
          )}

          {result && (
            <div className="space-y-2">
              <p className="rounded-lg border border-emerald-500/20 bg-emerald-500/10 p-3 text-sm text-emerald-300">
                {result.submittedCount} application(s) sent.
              </p>
              {result.skipped.map(reason => (
                <p key={reason} className="rounded-lg border border-amber-500/20 bg-amber-500/10 p-2.5 text-xs text-amber-300">
                  {reason}
                </p>
              ))}
            </div>
          )}

          {!result && preview && (
            <>
              {preview.lettersAreDeterministic && (
                <p className="flex items-start gap-2 rounded-lg border border-amber-500/20 bg-amber-500/10 p-3 text-xs text-amber-300">
                  <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  <span>
                    No chat model is configured, so these letters were composed rather than written.
                    They are accurate but plain — worth a read before sending.
                  </span>
                </p>
              )}

              {!preview.semanticMatching && (
                <p className="rounded-lg border border-amber-500/20 bg-amber-500/10 p-3 text-xs text-amber-300">
                  These scores come from the fallback embedder, which compares wording rather than
                  meaning. Treat the percentages as rough.
                </p>
              )}

              {preview.items.length === 0 && (
                <p className="py-10 text-center text-sm text-gray-400">No roles matched.</p>
              )}

              {preview.items.map(item => {
                const chosen = selected.has(item.job.id);
                const isEditing = editing === item.job.id;

                return (
                  <article
                    key={item.job.id}
                    className={`rounded-xl border p-4 transition ${
                      item.alreadyApplied
                        ? 'border-white/5 bg-black/20 opacity-60'
                        : chosen
                          ? 'border-indigo-500/40 bg-indigo-500/5'
                          : 'border-white/5 bg-card'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <input
                        type="checkbox"
                        checked={chosen}
                        disabled={item.alreadyApplied}
                        onChange={() => toggle(item.job.id)}
                        aria-label={`Apply to ${item.job.title}`}
                        className="mt-1 h-4 w-4 shrink-0 accent-indigo-500 disabled:opacity-40"
                      />
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-semibold text-white">{item.job.title}</p>
                        <p className="text-xs text-gray-400">
                          {[item.job.company, item.job.location, item.job.workMode].filter(Boolean).join(' · ')}
                        </p>
                      </div>
                      <div className="shrink-0 text-right">
                        <p className="text-sm font-bold text-white">{item.fitScore.toFixed(1)}%</p>
                        <p className="text-[10px] uppercase tracking-wide text-gray-500">{item.fitBand}</p>
                      </div>
                    </div>

                    {item.alreadyApplied ? (
                      <p className="mt-3 flex items-center gap-1.5 text-xs text-gray-400">
                        <Check className="h-3.5 w-3.5" /> Already applied with this CV.
                      </p>
                    ) : (
                      <>
                        {item.gaps.length > 0 && (
                          <p className="mt-2 text-[11px] text-gray-500">
                            Not evidenced on your CV: {item.gaps.join(', ')}. The letter does not claim these.
                          </p>
                        )}

                        {isEditing ? (
                          <textarea
                            value={letters[item.job.id] ?? ''}
                            onChange={event =>
                              setLetters(previous => ({ ...previous, [item.job.id]: event.target.value }))}
                            onBlur={() => setEditing(null)}
                            rows={9}
                            autoFocus
                            aria-label={`Cover letter for ${item.job.title}`}
                            className="mt-3 w-full rounded-lg border border-white/10 bg-black/40 p-3 text-xs leading-relaxed text-gray-200 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                          />
                        ) : (
                          <>
                            <button
                              type="button"
                              onClick={() => setEditing(item.job.id)}
                              className="mt-3 w-full rounded-lg border border-white/5 bg-black/20 p-3 text-left text-xs leading-relaxed whitespace-pre-wrap text-gray-300 transition hover:border-white/15"
                            >
                              {letters[item.job.id]}
                              <span className="mt-2 block text-[10px] font-semibold uppercase tracking-wide text-indigo-400">
                                Click to edit
                              </span>
                            </button>

                            <div className="mt-2 flex items-center gap-2">
                              {written.has(item.job.id) ? (
                                <span className="inline-flex items-center gap-1.5 rounded-md border border-emerald-500/20 bg-emerald-500/10 px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-emerald-300">
                                  <Sparkles className="h-3 w-3" /> Written by the model
                                </span>
                              ) : (
                                <button
                                  type="button"
                                  onClick={() => void draft(item.job.id)}
                                  disabled={drafting.has(item.job.id)}
                                  className="inline-flex items-center gap-1.5 rounded-md border border-indigo-500/30 bg-indigo-500/10 px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-indigo-300 transition hover:bg-indigo-500/20 disabled:opacity-50"
                                >
                                  <Sparkles className={`h-3 w-3 ${drafting.has(item.job.id) ? 'animate-pulse' : ''}`} />
                                  {drafting.has(item.job.id) ? 'Writing…' : 'Write with AI'}
                                </button>
                              )}
                              <span className="text-[10px] text-gray-500">
                                {drafting.has(item.job.id)
                                  ? 'the local model takes a few seconds'
                                  : 'this letter was composed, not written'}
                              </span>
                            </div>
                          </>
                        )}
                      </>
                    )}
                  </article>
                );
              })}
            </>
          )}
        </div>

        <footer className="flex shrink-0 items-center gap-3 border-t border-border px-5 py-4">
          {!result && preview && (
            <p className="text-xs text-gray-400">
              {selected.size} selected
              {preview.alreadyAppliedCount > 0 && ` · ${preview.alreadyAppliedCount} already applied`}
            </p>
          )}
          <div className="ml-auto flex gap-2">
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg border border-white/10 px-4 py-2 text-sm font-semibold text-gray-200 transition hover:bg-white/5"
            >
              {result ? 'Done' : 'Cancel'}
            </button>
            {!result && (
              <button
                type="button"
                onClick={submit}
                disabled={submitting || selected.size === 0}
                className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <Send className="h-4 w-4" />
                {submitting ? 'Sending…' : `Send ${selected.size || ''}`.trim()}
              </button>
            )}
          </div>
        </footer>
      </section>
    </div>
  );
};
