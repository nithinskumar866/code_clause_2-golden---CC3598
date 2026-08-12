import type { FC } from 'react';
import { ExternalLink, Lock } from 'lucide-react';
import type { JobSummary } from './types';

interface SuggestionCardProps {
  job: JobSummary;
  /** Prompts for a CV instead of applying, since applying needs one. */
  onNeedResume: () => void;
}

/**
 * A posting shown to someone who has not uploaded a CV.
 *
 * Carries no percentage anywhere, on purpose. A fit score is a claim about a
 * person, and the whole point of this card is that there is no person on file
 * yet — printing "0%" or dressing a text similarity up as a fit would invent
 * one. What it shows instead is what the role wants, and what uploading a CV
 * would tell you about it.
 */
export const SuggestionCard: FC<SuggestionCardProps> = ({ job, onNeedResume }) => (
  <article className="flex flex-col gap-2 rounded-xl border border-white/5 bg-card p-3">
    <div className="flex items-start justify-between gap-2">
      <h4 className="min-w-0 flex-1 text-sm font-semibold text-white">{job.title}</h4>
      {job.applyUrl && (
        <a
          href={job.applyUrl}
          target="_blank"
          rel="noreferrer"
          aria-label={`Open ${job.title} on the employer's site`}
          className="shrink-0 text-gray-500 transition hover:text-indigo-400"
        >
          <ExternalLink className="h-3.5 w-3.5" />
        </a>
      )}
    </div>

    <p className="text-[11px] text-gray-400">
      {[job.company, job.location, job.workMode,
        job.seniorityLevel !== 'Unspecified' ? job.seniorityLevel : null]
        .filter(Boolean).join(' · ')}
    </p>

    {job.requiredSkills.length > 0 && (
      <div className="flex flex-wrap gap-1">
        {job.requiredSkills.slice(0, 6).map(skill => (
          <span
            key={skill}
            className="rounded border border-white/10 bg-white/5 px-1.5 py-0.5 text-[10px] text-gray-300"
          >
            {skill}
          </span>
        ))}
      </div>
    )}

    {/* Applying needs a CV — there is nothing to send without one — so the button
        asks for it rather than being disabled with no explanation. */}
    <button
      type="button"
      onClick={onNeedResume}
      className="mt-1 inline-flex items-center justify-center gap-1.5 rounded-lg border border-indigo-500/30 bg-indigo-500/10 px-2.5 py-1.5 text-[11px] font-semibold text-indigo-300 transition hover:bg-indigo-500/20"
    >
      <Lock className="h-3 w-3" />
      Add your CV to score &amp; apply
    </button>
  </article>
);
