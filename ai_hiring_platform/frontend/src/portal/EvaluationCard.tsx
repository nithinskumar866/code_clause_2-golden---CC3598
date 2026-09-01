import { useState, type FC } from 'react';
import {
  CheckCircle2, ChevronDown, CircleSlash, ExternalLink, MinusCircle, Scale,
  ShieldAlert, Sigma, Signpost,
} from 'lucide-react';
import type { JobMatch, EvaluationRequirement, MatchLevel, RequirementKind } from './types';
import { CardSection, ModeBadge, ScoreRing } from './MatchVisuals';
import { scoreTone } from './match-tone';

interface EvaluationCardProps {
  match: JobMatch;
  compact?: boolean;
  onApply?: (jobId: number) => void;
}

/** The three verdict groups, in the order a person reads them: proven, partial, absent. */
const GROUPS: { level: MatchLevel; label: string; Icon: typeof CheckCircle2; tone: string; dot: string }[] = [
  {
    level: 'STRONG',
    label: 'Demonstrated in work history',
    Icon: CheckCircle2,
    tone: 'text-emerald-400',
    dot: 'bg-emerald-400',
  },
  {
    level: 'WEAK',
    label: 'Claimed, not demonstrated',
    Icon: MinusCircle,
    tone: 'text-amber-400',
    dot: 'bg-amber-400',
  },
  {
    level: 'MISSING',
    label: 'No evidence found',
    Icon: CircleSlash,
    tone: 'text-rose-400',
    dot: 'bg-rose-400',
  },
];

/**
 * How each kind of requirement is labelled and toned.
 *
 * The tone carries real information: a missing `must_have` is a dealbreaker and a
 * missing `responsibility` is usually nothing at all, so they must not look alike
 * in a list the eye scans for red.
 */
const KIND: Record<RequirementKind, { label: string; tone: string }> = {
  must_have: { label: 'must-have', tone: 'border-rose-500/25 bg-rose-500/10 text-rose-300' },
  core_skill: { label: 'core skill', tone: 'border-indigo-500/25 bg-indigo-500/10 text-indigo-300' },
  experience: { label: 'experience', tone: 'border-sky-500/25 bg-sky-500/10 text-sky-300' },
  responsibility: { label: 'duty of the role', tone: 'border-white/10 bg-white/5 text-gray-400' },
  education: { label: 'education', tone: 'border-white/10 bg-white/5 text-gray-400' },
  nice_to_have: { label: 'nice to have', tone: 'border-white/10 bg-white/5 text-gray-500' },
};

/** What the decision means, said plainly rather than as a status word. */
const DECISION_TONE: Record<string, string> = {
  'Proceed to Interview': 'border-emerald-500/25 bg-emerald-500/10 text-emerald-300',
  'Consider Alternate Role': 'border-amber-500/25 bg-amber-500/10 text-amber-300',
  'Consider for Alternate Role': 'border-amber-500/25 bg-amber-500/10 text-amber-300',
  Disqualify: 'border-rose-500/25 bg-rose-500/10 text-rose-300',
};

/**
 * One role, as judged rather than computed.
 *
 * The organising idea is that a percentage is a conclusion and this card shows
 * the argument. The score sits at the top; underneath it are the weights the
 * evaluator chose FOR THIS POSTING, then every requirement sorted into what the
 * CV proves, what it merely claims, and what it does not mention at all — each
 * with the line from the document that decided it.
 *
 * The requirement-level detail is collapsed by default and the summary is not.
 * A candidate scanning a shortlist wants the verdict; a candidate who disagrees
 * with the verdict wants the receipts, and giving both at once is how the card
 * became an unreadable wall in the first place.
 */
export const EvaluationCard: FC<EvaluationCardProps> = ({ match, compact = false, onApply }) => {
  const [open, setOpen] = useState(false);
  const [showWeights, setShowWeights] = useState(false);

  const { job, evaluation } = match;
  const tone = scoreTone(match.fitScore);

  // A reasoned match with no evaluation is one the model could not be reached
  // for. It keeps its computed score, and saying so is better than a card that
  // silently looks like every other.
  if (!evaluation) {
    return (
      <article className="flex flex-col gap-2 rounded-xl border border-white/5 bg-card p-4">
        <div className="flex items-start justify-between gap-3">
          <h3 className="min-w-0 flex-1 truncate text-sm font-semibold text-white">{job.title}</h3>
          <ScoreRing score={match.fitScore} size={44} />
        </div>
        <p className="text-[11px] text-amber-300/80">
          The evaluator could not be reached for this role, so this is the computed score.
        </p>
        <p className="text-xs leading-relaxed text-gray-400">{match.recruiterNote}</p>
      </article>
    );
  }

  // The per-requirement verdicts are the source of truth — they are what the score
  // is counted from. The older grouped lists are the fallback for an evaluation
  // stored before requirements were recorded, so an old conversation still renders.
  const requirements: EvaluationRequirement[] = evaluation.requirements?.length
    ? evaluation.requirements
    : [
        ...evaluation.highConfidence.map(m => ({
          requirement: m.requirement, quote: m.evidence, where: m.where, reasoning: '',
          matchLevel: 'STRONG' as const, kind: 'core_skill' as const,
        })),
        ...evaluation.partial.map(m => ({
          requirement: m.requirement, quote: m.evidence, where: m.where, reasoning: '',
          matchLevel: 'WEAK' as const, kind: 'core_skill' as const,
        })),
        ...evaluation.gaps.map(g => ({
          requirement: g.requirement, quote: '', where: '', reasoning: g.why,
          matchLevel: 'MISSING' as const, kind: 'core_skill' as const,
        })),
      ];

  const byLevel = (level: MatchLevel) => requirements.filter(r => r.matchLevel === level);

  return (
    <article className="flex flex-col gap-3 rounded-xl border border-white/5 bg-card p-4 shadow-lg shadow-black/20">
      {/* -- verdict -- */}
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="mb-1 flex flex-wrap items-center gap-1.5">
            <ModeBadge mode={match.scoringMode} />
            <span className={`text-[10px] font-semibold uppercase tracking-wide ${tone.text}`}>
              {evaluation.category}
            </span>
          </div>
          <h3 className={`truncate font-semibold text-white ${compact ? 'text-sm' : 'text-base'}`}>
            {job.title}
          </h3>
          <p className="truncate text-[11px] text-gray-400">
            {[job.company, job.location, job.workMode].filter(Boolean).join(' · ')}
          </p>
        </div>
        <ScoreRing score={evaluation.overallMatch} size={compact ? 52 : 60} label="match" />
      </div>

      {/* -- a dealbreaker, if one fired --
          Above the summary, because it is the only thing on the card that
          overrides everything else: the score is capped at 35 regardless of what
          the evidence below it says. */}
      {evaluation.knockout?.reasons?.length &&
        (evaluation.knockout.missingMandatorySkills || evaluation.knockout.missingYearsOfExperience) ? (
        <div className="rounded-lg border border-rose-500/25 bg-rose-500/10 px-2.5 py-2">
          <p className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-rose-300">
            <ShieldAlert className="h-3.5 w-3.5" />
            Dealbreaker — score capped
          </p>
          <ul className="mt-1 list-disc pl-4 text-[11px] leading-snug text-rose-200/80">
            {evaluation.knockout.reasons.slice(0, 3).map(reason => (
              <li key={reason}>{reason}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {evaluation.executiveSummary && (
        <p className="text-xs leading-relaxed text-gray-300">{evaluation.executiveSummary}</p>
      )}

      {/* -- the two readings of the same evidence --
          The ring is computed from the model's own per-requirement verdicts. This
          is what the model said the total was. They are shown together only when
          they disagree materially: agreement is the normal case and needs no
          commentary, while a wide gap means one of the two is wrong about this
          candidate — worth seeing rather than hiding behind whichever number we
          happened to pick. */}
      {evaluation.modelMatch > 0 && Math.abs(evaluation.overallMatch - evaluation.modelMatch) >= 10 && (
        <p className="flex items-center gap-1.5 rounded-lg border border-white/5 bg-black/25 px-2.5 py-1.5 text-[10px] text-gray-500">
          <Sigma className="h-3 w-3 shrink-0 text-gray-600" />
          <span>
            Evidence totals <span className="font-semibold text-gray-300">{evaluation.overallMatch}%</span>;
            the model called it <span className="font-semibold text-gray-400">{evaluation.modelMatch}%</span>.
            The requirement verdicts below are what the score is counted from.
          </span>
        </p>
      )}

      {/* -- the three verdict groups, as a single glanceable row --
          Counts before contents: how many requirements are met, half-met and
          missing is the whole shape of the answer, and it fits on one line. */}
      <div className="grid grid-cols-3 gap-1.5">
        {GROUPS.map(group => (
          <div
            key={group.level}
            title={group.label}
            className="flex items-center justify-center gap-1.5 rounded-lg border border-white/5 bg-black/25 py-1.5"
          >
            <group.Icon className={`h-3.5 w-3.5 ${group.tone}`} />
            <span className="text-sm font-semibold text-white">{byLevel(group.level).length}</span>
          </div>
        ))}
      </div>

      <button
        type="button"
        onClick={() => setOpen(value => !value)}
        aria-expanded={open}
        className="flex items-center justify-between rounded-lg border border-white/5 bg-white/[0.03] px-2.5 py-1.5 text-[11px] font-semibold text-gray-300 transition hover:bg-white/[0.06] hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
      >
        {open ? 'Hide the evidence' : 'Show requirement-by-requirement evidence'}
        <ChevronDown className={`h-3.5 w-3.5 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div className="flex flex-col gap-3">
          {GROUPS.map(group => (
            <EvidenceGroup key={group.level} group={group} items={byLevel(group.level)} />
          ))}
        </div>
      )}

      {/* -- the verdict, in words --
          Outside the expander on purpose. It lived inside it, after every
          requirement, which meant the one paragraph explaining the number was
          the last thing on a card most people never scrolled to the bottom of —
          the score looked asserted rather than argued. It is the answer to the
          only question a percentage provokes, so it sits under the evidence
          whether or not the evidence is open. */}
      {evaluation.reasoning && (
        <CardSection title="Why this score">
          <p className="text-xs leading-relaxed text-gray-300">{evaluation.reasoning}</p>
        </CardSection>
      )}

      {/* -- the weighting this posting got --
          Shown because it is the answer to "why did missing one thing cost so
          much?" — and because the weights are chosen per posting, so they are a
          real finding about this role rather than a constant. */}
      {evaluation.weights.length > 0 && (
        <CardSection
          title="How this role was weighted"
          aside={
            <button
              type="button"
              onClick={() => setShowWeights(value => !value)}
              aria-expanded={showWeights}
              className="inline-flex items-center gap-1 text-[10px] font-semibold text-gray-500 transition hover:text-gray-300"
            >
              <Scale className="h-3 w-3" />
              {showWeights ? 'hide' : 'show'}
            </button>
          }
        >
          {showWeights && (
            <ul className="flex flex-col gap-1.5">
              {evaluation.weights.map(weight => (
                <li key={weight.criterion} className="flex items-center gap-2">
                  <span className="w-[46%] shrink-0 truncate text-[11px] text-gray-400">
                    {/* The backend sends a kind slug; the chips elsewhere on this
                        card use the same vocabulary, so they must read alike. */}
                    {KIND[weight.criterion as RequirementKind]?.label ?? weight.criterion.replace(/_/g, ' ')}
                  </span>
                  <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-white/5">
                    <span
                      className="block h-full rounded-full bg-indigo-500/70"
                      style={{ width: `${Math.min(100, Math.max(0, weight.weight))}%` }}
                    />
                  </span>
                  <span className="w-8 shrink-0 text-right text-[10px] font-semibold text-gray-400">
                    {weight.weight}%
                  </span>
                </li>
              ))}
            </ul>
          )}
        </CardSection>
      )}

      {/* -- what to do about it -- */}
      {(evaluation.decision || evaluation.alternateRole) && (
        <CardSection title="Recommendation">
          <div className="flex flex-wrap items-center gap-2">
            {evaluation.decision && (
              <span
                className={`rounded-md border px-2 py-0.5 text-[10px] font-semibold ${
                  DECISION_TONE[evaluation.decision] ?? 'border-white/10 bg-white/5 text-gray-300'
                }`}
              >
                {evaluation.decision}
              </span>
            )}
            {evaluation.alternateRole && (
              <span className="inline-flex items-center gap-1 text-[11px] text-gray-400">
                <Signpost className="h-3 w-3 shrink-0 text-indigo-400" />
                Better fit: <span className="text-gray-300">{evaluation.alternateRole}</span>
              </span>
            )}
          </div>
        </CardSection>
      )}

      {/* -- the measured numbers, kept --
          The reasoned score replaced the computed one, it did not disprove it.
          These four are real measurements and stay available as a cross-check on
          a judgement that is, by design, one model's opinion. */}
      <details className="group">
        <summary className="cursor-pointer list-none text-[10px] font-semibold uppercase tracking-wider text-gray-600 transition hover:text-gray-400">
          Measured signals
        </summary>
        <div className="mt-2 grid grid-cols-4 gap-2 rounded-lg border border-white/5 bg-black/25 p-2 text-center">
          {[
            { label: 'relevance', value: match.semanticScore },
            { label: 'skills', value: match.skillScore },
            { label: 'title', value: match.titleScore },
            { label: 'experience', value: match.experienceScore },
          ].map(part => (
            <div key={part.label}>
              <p className="text-xs font-semibold text-gray-300">{part.value.toFixed(0)}%</p>
              <p className="text-[9px] uppercase tracking-wide text-gray-500">{part.label}</p>
            </div>
          ))}
        </div>
      </details>

      <div className="flex flex-wrap items-center gap-2 border-t border-white/5 pt-3">
        {onApply && (
          <button
            type="button"
            onClick={() => onApply(job.id)}
            className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-2.5 py-1.5 text-xs font-semibold text-white transition hover:bg-indigo-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
          >
            Apply on this board
          </button>
        )}
        {job.applyUrl && (
          <a
            href={job.applyUrl}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 text-xs font-semibold text-indigo-400 transition hover:text-indigo-300"
          >
            Employer’s site <ExternalLink className="h-3 w-3" />
          </a>
        )}
      </div>
    </article>
  );
};

type Group = (typeof GROUPS)[number];

const GroupHeading: FC<{ group: Group; count: number }> = ({ group, count }) => (
  <div className="mb-1.5 flex items-center gap-1.5">
    <span className={`h-1.5 w-1.5 rounded-full ${group.dot}`} aria-hidden="true" />
    <span className="text-[10px] font-semibold uppercase tracking-wider text-gray-500">
      {group.label} ({count})
    </span>
  </div>
);

/**
 * One verdict group with its quotes.
 *
 * The quote is the point of the whole card: it has been checked against the CV
 * before reaching here, so it is the candidate's own wording rather than the
 * model's account of it. `where` says which section it came from, because that is
 * what separates a skill someone has used from one they have listed.
 *
 * The KIND chip matters just as much in the missing group. "No evidence found"
 * reads as a failure, and for a must-have it is one — but for a duty of the role
 * it usually is not, and a candidate staring at a red list deserves to see which
 * of the two they are looking at.
 */
const EvidenceGroup: FC<{ group: Group; items: EvaluationRequirement[] }> = ({ group, items }) => {
  if (items.length === 0) return null;

  const missing = group.level === 'MISSING';

  return (
    <div>
      <GroupHeading group={group} count={items.length} />
      <ul className="flex flex-col gap-1.5">
        {items.map(item => {
          const kind = KIND[item.kind] ?? KIND.core_skill;
          return (
            <li
              key={item.requirement}
              className={`rounded-lg border px-2.5 py-1.5 ${
                missing ? 'border-rose-500/10 bg-rose-500/[0.06]' : 'border-white/5 bg-black/25'
              }`}
            >
              <div className="flex items-baseline justify-between gap-2">
                <p className={`min-w-0 flex-1 text-[11px] font-semibold ${missing ? 'text-rose-200' : 'text-gray-200'}`}>
                  {item.requirement}
                </p>
                <span className={`shrink-0 rounded border px-1 py-px text-[9px] uppercase tracking-wide ${kind.tone}`}>
                  {kind.label}
                </span>
              </div>

              {item.quote && (
                <p className="mt-1 border-l-2 border-white/10 pl-2 text-[11px] italic leading-snug text-gray-400">
                  “{item.quote}”
                  {item.where && <span className="not-italic text-gray-600"> — {item.where}</span>}
                </p>
              )}

              {/* The model's own sentence about THIS row. It can only be read
                  against the verdict and quote directly above it, which is what
                  makes it safe to show where a free-floating summary was not. */}
              {item.reasoning ? (
                <p className="mt-1 text-[11px] leading-snug text-gray-500">{item.reasoning}</p>
              ) : !item.quote && missing ? (
                <p className="mt-0.5 text-[11px] leading-snug text-gray-500">
                  {item.kind === 'responsibility'
                    ? 'A duty of the role, not a stated qualification — worth little against the score.'
                    : item.kind === 'must_have'
                      ? 'Mandatory and not evidenced anywhere in the CV.'
                      : 'Nothing in the CV supports this.'}
                </p>
              ) : null}
            </li>
          );
        })}
      </ul>
    </div>
  );
};
