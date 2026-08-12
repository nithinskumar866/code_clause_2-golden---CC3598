import type { FC } from 'react';
import { ExternalLink } from 'lucide-react';
import type { JobMatch, SkillStatus } from './types';

const SKILL_TONE: Record<SkillStatus, string> = {
  Have: 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300',
  Transferable: 'border-amber-500/20 bg-amber-500/10 text-amber-300',
  Missing: 'border-rose-500/20 bg-rose-500/10 text-rose-300',
};

/** Same score bands the recruiter side uses, so one number reads the same everywhere. */
const fitTone = (score: number) =>
  score >= 75 ? 'text-emerald-400' : score >= 50 ? 'text-amber-400' : 'text-rose-400';

const barTone = (score: number) =>
  score >= 75 ? 'bg-emerald-500' : score >= 50 ? 'bg-amber-500' : 'bg-rose-500';

interface MatchCardProps {
  match: JobMatch;
  /** Trims the card down for the chat popup, where the column is ~380px wide. */
  compact?: boolean;
  /** Offers to apply to this one role. Opens the review; never sends directly. */
  onApply?: (jobId: number) => void;
}

/**
 * One scored match.
 *
 * Every skill is shown with its verdict, not just the gaps. A candidate looking
 * at a 65% wants to know what the other 35% was, and a card that lists only what
 * is missing reads as a rejection rather than as advice.
 */
export const MatchCard: FC<MatchCardProps> = ({ match, compact = false, onApply }) => {
  const { job } = match;

  return (
    <article className="flex flex-col gap-3 rounded-xl border border-white/5 bg-card p-4">
      <div className="flex items-baseline justify-between gap-3">
        <h3 className={`min-w-0 truncate font-semibold text-white ${compact ? 'text-sm' : 'text-base'}`}>
          {job.title}
        </h3>
        <div className="shrink-0 text-right">
          <span className={`font-bold ${fitTone(match.fitScore)} ${compact ? 'text-base' : 'text-lg'}`}>
            {match.fitScore.toFixed(1)}%
          </span>
          <span className="ml-1.5 text-[10px] font-semibold uppercase tracking-wide text-gray-500">
            {match.fitBand}
          </span>
        </div>
      </div>

      <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/5">
        <div
          className={`h-full rounded-full ${barTone(match.fitScore)}`}
          style={{ width: `${Math.min(100, Math.max(0, match.fitScore))}%` }}
        />
      </div>

      <p className="text-xs text-gray-400">
        {[
          job.company,
          job.location,
          job.workMode,
          job.seniorityLevel !== 'Unspecified' ? job.seniorityLevel : null,
        ].filter(Boolean).join(' · ')}
      </p>

      <div className="grid grid-cols-4 gap-2 rounded-lg border border-white/5 bg-black/20 p-2 text-center">
        {[
          { label: 'relevance', value: match.semanticScore },
          { label: 'skills', value: match.skillScore },
          { label: 'title', value: match.titleScore },
          { label: 'experience', value: match.experienceScore },
        ].map(part => (
          <div key={part.label}>
            <p className="text-sm font-semibold text-white">{part.value.toFixed(0)}%</p>
            <p className="text-[9px] uppercase tracking-wide text-gray-500">{part.label}</p>
          </div>
        ))}
      </div>

      {match.skills.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {match.skills.map(skill => (
            <span
              key={skill.skill}
              className={`rounded-md border px-1.5 py-0.5 text-[10px] font-medium ${SKILL_TONE[skill.status] ?? 'border-white/10 bg-white/5 text-gray-300'}`}
              /* The evidence belongs on the chip: "Transferable" on its own
                 invites "transferable from what?", and the answer is already known. */
              title={
                skill.status === 'Missing'
                  ? 'Not evidenced on the resume'
                  : `Matched by ${skill.evidenceSkill} (${(skill.similarity * 100).toFixed(0)}%)`
              }
            >
              {skill.skill}
            </span>
          ))}
        </div>
      )}

      {match.recruiterNote && (
        <p className="border-t border-white/5 pt-3 text-xs leading-relaxed text-gray-400">
          {match.recruiterNote}
        </p>
      )}

      <div className="flex flex-wrap items-center gap-2">
        {onApply && (
          <button
            type="button"
            onClick={() => onApply(job.id)}
            className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-2.5 py-1.5 text-xs font-semibold text-white transition hover:bg-indigo-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
          >
            Apply on this board
          </button>
        )}

        {/* The posting's own link, when it has one. Kept separate from the button
            above so it is never mistaken for applying here. */}
        {job.applyUrl && (
          <a
            href={job.applyUrl}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 text-xs font-semibold text-indigo-400 transition hover:text-indigo-300"
          >
            Apply on the employer's site <ExternalLink className="h-3 w-3" />
          </a>
        )}
      </div>
    </article>
  );
};
