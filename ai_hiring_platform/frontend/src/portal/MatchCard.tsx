import type { FC } from 'react';
import { ExternalLink } from 'lucide-react';
import type { JobMatch, SkillStatus } from './types';
import { EvaluationCard } from './EvaluationCard';
import { CardSection, ModeBadge, ScoreRing } from './MatchVisuals';
import { scoreTone } from './match-tone';

const SKILL_TONE: Record<SkillStatus, string> = {
  Have: 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300',
  Transferable: 'border-amber-500/20 bg-amber-500/10 text-amber-300',
  Missing: 'border-rose-500/20 bg-rose-500/10 text-rose-300',
};

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
 * Dispatches on how the score was produced, so every caller in the app gets the
 * right card without knowing the modes exist. A reasoned match is a different
 * kind of object — it carries an argument, not four dimensions — and squeezing it
 * into this layout would throw away the part worth reading.
 *
 * Every skill is shown with its verdict, not just the gaps. A candidate looking
 * at a 65% wants to know what the other 35% was, and a card that lists only what
 * is missing reads as a rejection rather than as advice.
 */
export const MatchCard: FC<MatchCardProps> = ({ match, compact = false, onApply }) => {
  if (match.scoringMode === 'reasoned' || match.scoringMode === 'rag') {
    return <EvaluationCard match={match} compact={compact} onApply={onApply} />;
  }

  const { job } = match;
  const tone = scoreTone(match.fitScore);

  return (
    <article className="flex flex-col gap-3 rounded-xl border border-white/5 bg-card p-4 shadow-lg shadow-black/20">
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="mb-1 flex flex-wrap items-center gap-1.5">
            <ModeBadge mode="computed" />
            <span className={`text-[10px] font-semibold uppercase tracking-wide ${tone.text}`}>
              {match.fitBand}
            </span>
          </div>
          <h3 className={`truncate font-semibold text-white ${compact ? 'text-sm' : 'text-base'}`}>
            {job.title}
          </h3>
          <p className="truncate text-[11px] text-gray-400">
            {[
              job.company,
              job.location,
              job.workMode,
              job.seniorityLevel !== 'Unspecified' ? job.seniorityLevel : null,
            ].filter(Boolean).join(' · ')}
          </p>
        </div>
        <ScoreRing score={match.fitScore} size={compact ? 52 : 60} label="fit" />
      </div>

      <CardSection title="What the number is made of">
        <div className="grid grid-cols-4 gap-2 rounded-lg border border-white/5 bg-black/25 p-2 text-center">
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
      </CardSection>

      {match.skills.length > 0 && (
        <CardSection title="Skills this posting asks for">
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
        </CardSection>
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
