import { useCallback, useEffect, useState, type FC } from 'react';
import { Check, Download, Inbox, X } from 'lucide-react';
import { applicationApi } from './api';
import { usePortalState } from './portal-context';
import type { Application } from './types';
import { PageHeader } from '../components/ui/PageHeader';
import { Banner } from '../components/ui/Banner';
import { EmptyState } from '../components/ui/EmptyState';
import { Select } from '../components/ui/Select';
import { Spinner } from '../components/ui/Spinner';

const STATUS_TONE: Record<string, string> = {
  Submitted: 'border-indigo-500/20 bg-indigo-500/10 text-indigo-300',
  Accepted: 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300',
  Declined: 'border-rose-500/20 bg-rose-500/10 text-rose-300',
};

const fitTone = (score: number) =>
  score >= 75 ? 'text-emerald-400' : score >= 50 ? 'text-amber-400' : 'text-rose-400';

/**
 * Who applied, to what, and on what evidence.
 *
 * The scores here are the ones frozen when the application was made, not a fresh
 * evaluation. That is deliberate: a candidate must be judged on what they were
 * shown when they applied, and re-scoring on read would silently move the bar as
 * models and thresholds change.
 */
export const Applicants: FC = () => {
  const { engage, reachable } = usePortalState();

  const [applications, setApplications] = useState<Application[]>([]);
  const [status, setStatus] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<number | null>(null);
  const [expanded, setExpanded] = useState<number | null>(null);

  useEffect(engage, [engage]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setApplications(await applicationApi.list({ status: status || undefined }));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Applications could not be loaded.');
    } finally {
      setLoading(false);
    }
  }, [status]);

  useEffect(() => { void load(); }, [load]);

  const decide = async (application: Application, next: 'Accepted' | 'Declined') => {
    setBusyId(application.id);
    setError(null);
    try {
      // Accepting records the decision on the portal side. Bringing the candidate
      // into the recruiter pipeline is a separate, Python-side write — this app
      // never writes the recruiter platform's tables from here.
      const updated = await applicationApi.setStatus(application.id, next);
      setApplications(previous => previous.map(a => (a.id === updated.id ? updated : a)));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'That decision could not be saved.');
    } finally {
      setBusyId(null);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader
        icon={<Inbox className="h-5 w-5" />}
        title="Applicants"
        description="People who applied through the candidate-facing board, with the fit score captured at the moment they applied."
        actions={
          <Select
            label="Status"
            srLabel
            value={status}
            onChange={event => setStatus(event.target.value)}
            options={[
              { value: '', label: 'All statuses' },
              { value: 'Submitted', label: 'Submitted' },
              { value: 'Accepted', label: 'Accepted' },
              { value: 'Declined', label: 'Declined' },
            ]}
          />
        }
      />

      {reachable === false && (
        <Banner variant="error" title="Job portal API unreachable">
          Start it with <code>dotnet run</code> in <code>job_portal/backend/JobPortal.Api</code>.
        </Banner>
      )}
      {error && <Banner variant="error" onDismiss={() => setError(null)}>{error}</Banner>}

      {/* Said plainly rather than left to be discovered: accepting records the
          decision here, but does not yet create the analysis that would put this
          person into History, Ranking and Analytics. */}
      <Banner variant="info" title="Accepting records the decision only">
        Bringing an applicant into the hiring pipeline is a write the Python backend has to make,
        and that endpoint is not built yet. Until it is, accept and decline track your shortlist
        here; evaluate an accepted candidate from <strong>AI Analysis</strong> in the meantime.
      </Banner>

      {loading && applications.length === 0 && (
        <div className="flex justify-center py-16"><Spinner className="h-6 w-6" /></div>
      )}

      {!loading && applications.length === 0 && (
        <EmptyState
          icon={<Inbox className="h-8 w-8" />}
          title="No applications yet"
          description="When a candidate applies from the board or the assistant, they appear here."
        />
      )}

      <div className="space-y-3">
        {applications.map(application => {
          const open = expanded === application.id;

          return (
            <article key={application.id} className="rounded-xl border border-white/5 bg-card p-4">
              <div className="flex flex-wrap items-start gap-3">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="truncate text-sm font-semibold text-white">
                      {application.candidate.candidateName || application.candidate.filename}
                    </h3>
                    <span
                      className={`rounded-md border px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${
                        STATUS_TONE[application.status] ?? 'border-white/10 bg-white/5 text-gray-300'
                      }`}
                    >
                      {application.status}
                    </span>
                  </div>
                  <p className="mt-0.5 text-xs text-gray-400">
                    applied to <span className="text-gray-300">{application.job.title}</span>
                    {application.job.company && ` · ${application.job.company}`}
                    {' · '}
                    {new Date(application.createdAt).toLocaleDateString()}
                  </p>
                </div>

                <div className="shrink-0 text-right">
                  <p className={`text-lg font-bold ${fitTone(application.fitScore)}`}>
                    {application.fitScore.toFixed(1)}%
                  </p>
                  <p className="text-[10px] uppercase tracking-wide text-gray-500">
                    {application.fitBand}
                  </p>
                </div>
              </div>

              {/* A score from the fallback embedder means something different from a
                  real one, so it says which it was rather than showing a bare number. */}
              {!application.semanticMatching && (
                <p className="mt-2 text-[11px] text-amber-400">
                  Scored without semantic matching — compares wording, not meaning.
                </p>
              )}

              <div className="mt-3 flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => setExpanded(open ? null : application.id)}
                  className="rounded-lg border border-white/10 px-2.5 py-1.5 text-xs font-semibold text-gray-200 transition hover:bg-white/5"
                >
                  {open ? 'Hide details' : 'View letter & evidence'}
                </button>

                {application.hasResumeFile && (
                  <a
                    href={applicationApi.resumeUrl(application.id)}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 px-2.5 py-1.5 text-xs font-semibold text-gray-200 transition hover:bg-white/5"
                  >
                    <Download className="h-3.5 w-3.5" /> CV
                  </a>
                )}

                {application.status !== 'Accepted' && (
                  <button
                    type="button"
                    disabled={busyId === application.id}
                    onClick={() => decide(application, 'Accepted')}
                    className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600/90 px-2.5 py-1.5 text-xs font-semibold text-white transition hover:bg-emerald-500 disabled:opacity-50"
                  >
                    <Check className="h-3.5 w-3.5" /> Accept
                  </button>
                )}

                {application.status !== 'Declined' && (
                  <button
                    type="button"
                    disabled={busyId === application.id}
                    onClick={() => decide(application, 'Declined')}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 px-2.5 py-1.5 text-xs font-semibold text-gray-300 transition hover:bg-rose-500/10 hover:text-rose-300 disabled:opacity-50"
                  >
                    <X className="h-3.5 w-3.5" /> Decline
                  </button>
                )}
              </div>

              {open && (
                <div className="mt-4 space-y-4 border-t border-white/5 pt-4">
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                    {[
                      { label: 'relevance', value: application.semanticScore },
                      { label: 'skills', value: application.skillScore },
                      { label: 'title', value: application.titleScore },
                      { label: 'experience', value: application.experienceScore },
                    ].map(part => (
                      <div key={part.label} className="rounded-lg border border-white/5 bg-black/20 p-2 text-center">
                        <p className="text-sm font-semibold text-white">{part.value.toFixed(0)}%</p>
                        <p className="text-[9px] uppercase tracking-wide text-gray-500">{part.label}</p>
                      </div>
                    ))}
                  </div>

                  {application.skills.length > 0 && (
                    <div>
                      <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-gray-500">
                        Skill verdicts when they applied
                      </p>
                      <div className="flex flex-wrap gap-1.5">
                        {application.skills.map(skill => (
                          <span
                            key={skill.skill}
                            title={skill.evidenceSkill ? `Matched by ${skill.evidenceSkill}` : 'Not evidenced'}
                            className={`rounded-md border px-1.5 py-0.5 text-[10px] font-medium ${
                              skill.status === 'Have' ? 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300'
                              : skill.status === 'Transferable' ? 'border-amber-500/20 bg-amber-500/10 text-amber-300'
                              : 'border-rose-500/20 bg-rose-500/10 text-rose-300'
                            }`}
                          >
                            {skill.skill}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  <div>
                    <p className="mb-1.5 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wider text-gray-500">
                      Cover letter
                      <span className="rounded border border-white/10 bg-white/5 px-1 py-0.5 normal-case tracking-normal text-gray-400">
                        {application.letterMode === 'reviewed' ? 'edited by the candidate'
                          : application.letterMode === 'llm' ? 'written by the model, grounded'
                          : 'composed'}
                      </span>
                    </p>
                    <p className="whitespace-pre-wrap rounded-lg border border-white/5 bg-black/20 p-3 text-xs leading-relaxed text-gray-300">
                      {application.coverLetter}
                    </p>
                  </div>

                  {application.recruiterNote && (
                    <p className="text-xs leading-relaxed text-gray-400">{application.recruiterNote}</p>
                  )}

                  <div className="flex flex-wrap gap-4 text-[11px] text-gray-500">
                    {application.candidate.email && <span>{application.candidate.email}</span>}
                    {application.candidate.phone && <span>{application.candidate.phone}</span>}
                    {application.acceptedAnalysisId && (
                      <span className="text-emerald-400">
                        In the pipeline as analysis #{application.acceptedAnalysisId}
                      </span>
                    )}
                  </div>
                </div>
              )}
            </article>
          );
        })}
      </div>
    </div>
  );
};
