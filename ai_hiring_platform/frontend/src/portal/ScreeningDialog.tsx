import { useCallback, useEffect, useState, type FC } from 'react';
import { ArrowLeft, Briefcase, Check, ExternalLink, MapPin, Sparkles, X } from 'lucide-react';
import { screeningApi } from './api';
import { Working } from './Working';
import type { ScreeningQuestion, ScreeningQuestionSet, ScreeningTopic } from './types';

interface ScreeningDialogProps {
  jobId: number;
  /** Called once every question has an answer and the server has recorded them. */
  onPassed: () => void;
  onClose: () => void;
}

/** A small visual cue for what each question is about. */
const TOPIC: Record<ScreeningTopic, { label: string; Icon: typeof MapPin }> = {
  location: { label: 'Location', Icon: MapPin },
  work_mode: { label: 'Work mode', Icon: MapPin },
  skill: { label: 'Skill', Icon: Sparkles },
  experience: { label: 'Experience', Icon: Briefcase },
  employment_type: { label: 'Contract', Icon: Briefcase },
};

/**
 * The questions between wanting a role and applying for it, asked one at a time.
 *
 * One per screen rather than a form of four. A list invites skimming and picking
 * the answers that look like they unlock the button; a single question with
 * nothing else on screen gets read. That is the entire point — these ask about
 * things a CV cannot say, so an unconsidered answer is worth nothing to anyone.
 *
 * ANSWERING is what is required. Yes and no both advance. Someone who reads
 * "Berlin, onsite", says they will not relocate, and applies anyway has made an
 * informed decision that is theirs to make — and the recruiter now knows something
 * no CV would have told them. The noes travel with the application rather than
 * blocking it.
 */
export const ScreeningDialog: FC<ScreeningDialogProps> = ({ jobId, onPassed, onClose }) => {
  const [set, setSet] = useState<ScreeningQuestionSet | null>(null);
  const [step, setStep] = useState(0);
  const [answers, setAnswers] = useState<Record<number, boolean>>({});
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setError(null);

    screeningApi.questions(jobId)
      .then(result => { if (!cancelled) setSet(result); })
      .catch(e => {
        if (cancelled) return;
        // Failing open would be worse than failing shut: a step that silently
        // disappears when the service is down is not a step.
        setError(e instanceof Error ? e.message : 'The questions could not be loaded.');
      });

    return () => { cancelled = true; };
  }, [jobId]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const finish = useCallback(async (final: Record<number, boolean>) => {
    if (!set) return;

    setSubmitting(true);
    setError(null);
    try {
      // Recorded server-side rather than counted here. The answers are the point —
      // they go to the employer — so they have to reach it before the review opens.
      const result = await screeningApi.submit(
        jobId, set.questions.map(q => ({ questionId: q.id, yes: final[q.id] })));

      // Recorded. Now the handover card — the apply review opens from there, not
      // from here, so the contacts are seen rather than flashed past.
      if (result.passed) setStep(set.questions.length);
      else setError(result.message);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Your answers could not be saved.');
    } finally {
      setSubmitting(false);
    }
  }, [jobId, set]);

  const answer = useCallback((yes: boolean) => {
    if (!set || submitting) return;

    const question = set.questions[step];
    const next = { ...answers, [question.id]: yes };
    setAnswers(next);

    // The last answer records them and moves to the handover card rather than
    // closing. Ending on "you're done" is a dead stop; ending on "here is who to
    // ask" is the one moment the applicant is certain to be paying attention.
    if (step + 1 < set.questions.length) setStep(step + 1);
    else void finish(next);
  }, [answers, finish, set, step, submitting]);

  const total = set?.questions.length ?? 0;
  const onHandover = set !== null && step >= total;
  const question: ScreeningQuestion | undefined = onHandover ? undefined : set?.questions[step];

  return (
    <div
      className="fixed inset-0 z-[60] flex items-end justify-center bg-black/70 backdrop-blur-sm sm:items-center"
      onClick={onClose}
      role="presentation"
    >
      <section
        role="dialog"
        aria-modal="true"
        aria-label="A few questions before you apply"
        onClick={event => event.stopPropagation()}
        className="flex w-full max-w-lg flex-col rounded-t-2xl border border-white/10 bg-card shadow-2xl sm:rounded-2xl"
      >
        <header className="flex shrink-0 items-start gap-3 border-b border-border px-5 py-4">
          <div className="min-w-0 flex-1">
            <h2 className="text-base font-semibold text-white">
              {!set ? 'A few questions first' : onHandover ? 'Before you apply' : `Question ${step + 1} of ${total}`}
            </h2>
            <p className="mt-0.5 truncate text-xs text-gray-400">
              {set ? set.jobTitle : 'Reading the posting…'}
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

        {/* Progress. Segments rather than a bar, so "two more" is countable at a
            glance instead of estimated from a width. */}
        {set && (
          <div className="flex shrink-0 gap-1 px-5 pt-3">
            {set.questions.map((q, i) => (
              <span
                key={q.id}
                className={`h-1 flex-1 rounded-full transition ${
                  i < step ? "bg-indigo-500" : i === step ? "bg-indigo-400" : "bg-white/10"
                }`}
              />
            ))}
          </div>
        )}

        <div className="flex-1 px-5 py-5">
          {error && (
            <p className="mb-3 rounded-lg border border-rose-500/20 bg-rose-500/10 p-2.5 text-xs text-rose-300">
              {error}
            </p>
          )}

          {!set && !error && (
            <Working
              label="Reading the posting…"
              detail="The questions come from this job's own requirements, so the first one takes a moment."
            />
          )}

          {question && (
            <div key={question.id} className="animate-fadeIn">
              <div className="mb-3 flex items-center gap-1.5">
                {(() => {
                  const topic = TOPIC[question.topic] ?? TOPIC.skill;
                  return (
                    <>
                      <topic.Icon className="h-3.5 w-3.5 shrink-0 text-indigo-400" />
                      <span className="text-[10px] font-semibold uppercase tracking-wider text-gray-500">
                        {topic.label}
                      </span>
                    </>
                  );
                })()}
              </div>

              <p className="text-lg font-medium leading-snug text-white">{question.question}</p>

              {/* Why this is being asked, in the posting's own words. A screening
                  question with no visible basis reads as an arbitrary hurdle. */}
              {question.basedOn && (
                <p className="mt-2 border-l-2 border-white/10 pl-2.5 text-[11px] italic leading-snug text-gray-500">
                  the posting says: {question.basedOn}
                </p>
              )}

              <div className="mt-5 flex gap-2.5">
                {[true, false].map(value => (
                  <button
                    key={String(value)}
                    type="button"
                    disabled={submitting}
                    onClick={() => answer(value)}
                    className={`inline-flex flex-1 items-center justify-center gap-1.5 rounded-xl border px-4 py-3 text-sm font-semibold transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:opacity-50 ${
                      answers[question.id] === value
                        ? 'border-indigo-500/50 bg-indigo-500/20 text-indigo-200'
                        : 'border-white/10 bg-white/5 text-gray-200 hover:border-indigo-500/40 hover:bg-indigo-500/10'
                    }`}
                  >
                    {answers[question.id] === value && <Check className="h-4 w-4" />}
                    {value ? 'Yes' : 'No'}
                  </button>
                ))}
              </div>

              {/* Said plainly, because the shape of the UI implies the opposite: two
                  buttons and a progress bar look like a test with a right answer. */}
              <p className="mt-3 text-center text-[11px] text-gray-600">
                {step + 1 === total
                  ? 'Either answer finishes this step. Your answers go to the employer.'
                  : 'Either answer moves you on — this is not a test.'}
              </p>
            </div>
          )}

          {/* -- the handover --
              The last screen, and the only one anybody is guaranteed to read: they
              have just finished and are waiting to be let through. So it carries the
              thing worth carrying — who to ask. */}
          {onHandover && set && (
            <div className="animate-fadeIn">
              <div className="mb-4 flex items-center gap-2">
                <span className="flex h-7 w-7 items-center justify-center rounded-full bg-emerald-500/15">
                  <Check className="h-4 w-4 text-emerald-400" />
                </span>
                <div>
                  <p className="text-sm font-semibold text-white">All four answered</p>
                  <p className="text-[11px] text-gray-500">
                    They go to the employer with your application.
                  </p>
                </div>
              </div>

              {set.contacts.length > 0 && (
                <>
                  <p className="mb-2 text-xs text-gray-400">
                    Anything you want to ask before applying?
                  </p>

                  <div className="space-y-2">
                    {set.contacts.map(contact => (
                      <a
                        key={contact.url}
                        href={contact.url}
                        target="_blank"
                        rel="noreferrer noopener"
                        className={`flex items-center gap-3 rounded-xl border px-3 py-2.5 transition ${
                          contact.relevant
                            ? 'border-indigo-500/30 bg-indigo-500/10 hover:bg-indigo-500/15'
                            : 'border-white/5 bg-black/25 hover:bg-white/5'
                        }`}
                      >
                        <span
                          className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-xs font-bold ${
                            contact.relevant
                              ? 'bg-indigo-500/25 text-indigo-200'
                              : 'bg-white/5 text-gray-400'
                          }`}
                          aria-hidden="true"
                        >
                          {contact.name.slice(0, 1).toUpperCase()}
                        </span>

                        <span className="min-w-0 flex-1">
                          <span className="block truncate text-sm font-semibold text-white">
                            {contact.name}
                          </span>
                          <span className="block truncate text-[11px] text-gray-400">
                            {/* Why this person is at the top, when they are. */}
                            {contact.relevant
                              ? `${contact.handles} — that's this one`
                              : contact.handles}
                          </span>
                        </span>

                        <ExternalLink className="h-3.5 w-3.5 shrink-0 text-gray-500" />
                      </a>
                    ))}
                  </div>
                </>
              )}

              <button
                type="button"
                onClick={onPassed}
                className="mt-5 w-full rounded-xl bg-indigo-600 px-4 py-3 text-sm font-semibold text-white transition hover:bg-indigo-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
              >
                Continue to apply
              </button>
            </div>
          )}
        </div>

        <footer className="flex shrink-0 items-center justify-between gap-3 border-t border-border px-5 py-3">
          <button
            type="button"
            onClick={() => setStep(s => Math.max(0, s - 1))}
            disabled={step === 0 || submitting}
            className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-xs font-semibold text-gray-400 transition hover:text-white disabled:opacity-30"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Back
          </button>

          <p className="text-[11px] text-gray-600">
            {submitting ? 'Saving your answers…' : 'Nothing is sent until you confirm the application.'}
          </p>
        </footer>
      </section>
    </div>
  );
};
