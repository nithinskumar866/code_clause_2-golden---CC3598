import { useState, type FC } from 'react';
import { AlertTriangle, ChevronDown, ChevronRight, Eye, EyeOff, CheckCircle2, AlertCircle, XCircle } from 'lucide-react';
import type { RequirementFit } from '../../types';
import { StatusBadge } from './StatusBadge';

interface RequirementListProps {
  requirements: RequirementFit[];
  /** Skills the authenticity check flagged as listed-but-not-demonstrated. */
  overClaimedSkills?: string[];
}

/** Must-have vs nice-to-have pill (F2). Absent importance renders nothing. */
const ImportanceBadge: FC<{ importance?: 'must' | 'nice' | null; weight?: number | null }> = ({
  importance,
  weight,
}) => {
  if (!importance) return null;
  const isMust = importance === 'must';
  const style = isMust
    ? 'text-indigo-300 border-indigo-500/30 bg-indigo-500/10'
    : 'text-gray-400 border-white/10 bg-white/5';
  return (
    <span
      className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${style}`}
      title={weight != null ? `Scoring weight ${weight}` : undefined}
    >
      {isMust ? 'Must-have' : 'Nice-to-have'}
    </span>
  );
};

/** Color accent for the left border of each card based on status */
const statusBorderColor = (status: string) => {
  switch (status) {
    case 'Matched': return 'border-l-emerald-500';
    case 'Partial': return 'border-l-amber-500';
    case 'Missing': return 'border-l-rose-500';
    default: return 'border-l-gray-500';
  }
};

/** A small circular progress ring for confidence */
const ConfidenceRing: FC<{ value: number }> = ({ value }) => {
  const size = 28;
  const stroke = 2.5;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (Math.max(0, Math.min(100, value)) / 100) * circumference;
  const color = value >= 75 ? '#34d399' : value >= 50 ? '#fbbf24' : '#f87171';

  return (
    <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="transform -rotate-90">
        <circle cx={size/2} cy={size/2} r={radius} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth={stroke} />
        <circle
          cx={size/2} cy={size/2} r={radius}
          fill="none" stroke={color} strokeWidth={stroke}
          strokeDasharray={circumference} strokeDashoffset={offset}
          strokeLinecap="round"
          className="transition-all duration-500"
        />
      </svg>
      <span className="absolute text-[8px] font-bold" style={{ color }}>{value}%</span>
    </div>
  );
};

/** Single collapsible requirement card */
const RequirementCard: FC<{
  req: RequirementFit;
  isOverClaimed: boolean;
  isExpanded: boolean;
  onToggle: () => void;
}> = ({ req, isOverClaimed, isExpanded, onToggle }) => {
  const borderColor = statusBorderColor(req.status);

  return (
    <div
      className={`rounded-lg border border-white/5 border-l-[3px] ${borderColor} bg-[#141422] overflow-hidden transition-all duration-200 hover:border-white/10`}
    >
      {/* Card Header — always visible, clickable to toggle */}
      <button
        onClick={onToggle}
        className="w-full flex flex-wrap items-center gap-x-3 gap-y-1 px-4 py-3 text-left group cursor-pointer"
      >
        {/* Expand/collapse chevron */}
        <div className="text-gray-500 group-hover:text-gray-300 transition shrink-0">
          {isExpanded
            ? <ChevronDown className="h-3.5 w-3.5" />
            : <ChevronRight className="h-3.5 w-3.5" />
          }
        </div>

        {/* Skill name */}
        <span className="min-w-0 flex-1 truncate text-sm font-semibold text-gray-100">
          {req.requirement}
        </span>

        {/* Badges row */}
        <div className="flex items-center gap-2 shrink-0 flex-wrap justify-end">
          <span className="inline-flex items-center rounded bg-white/5 border border-white/10 px-1.5 py-0.5 text-[9px] font-medium text-gray-500">
            {req.category}
          </span>
          <ImportanceBadge importance={req.importance} weight={req.weight} />
          {isOverClaimed && (
            <span
              className="inline-flex items-center gap-1 rounded border border-rose-500/30 bg-rose-500/10 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-rose-300"
              title="Listed but not demonstrated in Experience/Projects"
            >
              <AlertTriangle className="h-2.5 w-2.5" /> Listed only
            </span>
          )}
        </div>

        {/* Confidence ring + Status badge */}
        <div className="flex items-center gap-2.5 shrink-0 ml-1">
          <ConfidenceRing value={req.confidence} />
          <StatusBadge status={req.status} />
        </div>

        {/* The plain-language verdict gets its OWN full-width row, so it is never
            squeezed to "Listed in skills — n…" by the badges beside it. This line is
            the whole point of the card: it answers "how good is this?" at a glance. */}
        {req.evidence_summary && (
          <span className="w-full basis-full pl-6 text-[11px] font-normal leading-relaxed text-gray-400">
            {req.evidence_summary}
          </span>
        )}
      </button>

      {/* Card Body — expanded details */}
      {isExpanded && (
        <div className="px-4 pb-4 pt-0 space-y-3 animate-fadeIn border-t border-white/5">
          {/* Relevance */}
          <div className="pt-3">
            <div className="flex items-start gap-2">
              <span className="text-indigo-400 text-[10px] font-bold uppercase tracking-wider shrink-0 mt-0.5">
                Relevance
              </span>
              <p className="text-xs text-gray-300 leading-relaxed">
                {req.explanation}
              </p>
            </div>
          </div>

          {/* Limitations */}
          {req.limitations && req.limitations !== 'None' && (
            <div className="flex items-start gap-2">
              <span className="text-rose-400 text-[10px] font-bold uppercase tracking-wider shrink-0 mt-0.5">
                Limitation
              </span>
              <p className="text-xs text-gray-400 leading-relaxed">
                {req.limitations}
              </p>
            </div>
          )}

          {/* Evidence quote */}
          {req.matched_evidence && (
            <div className="mt-2">
              <span className="text-[10px] text-gray-500 font-semibold uppercase tracking-wider block mb-1.5">
                Supporting Evidence
              </span>
              <div className="rounded-md bg-black/50 border border-indigo-500/10 p-3 relative">
                <div className="absolute top-0 left-0 w-0.5 h-full bg-indigo-500/40 rounded-l" />
                <p className="text-xs text-indigo-300/90 leading-relaxed pl-2 italic">
                  "{req.matched_evidence}"
                </p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

/**
 * Redesigned requirement-by-requirement evidence cards with:
 * - Summary status bar at top (Matched/Partial/Missing counts)
 * - Collapsible cards with color-coded left borders
 * - Progressive disclosure: header-only by default, expand for details
 * - Grouped by importance (must-haves first)
 */
export const RequirementList: FC<RequirementListProps> = ({ requirements, overClaimedSkills = [] }) => {
  const overClaimed = new Set(overClaimedSkills.map((s) => s.toLowerCase()));

  // Track which cards are expanded. Start with all collapsed.
  const [expandedSet, setExpandedSet] = useState<Set<number>>(new Set());
  const allExpanded = expandedSet.size === requirements.length && requirements.length > 0;

  const toggleCard = (idx: number) => {
    setExpandedSet((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  };

  const toggleAll = () => {
    if (allExpanded) {
      setExpandedSet(new Set());
    } else {
      setExpandedSet(new Set(requirements.map((_, i) => i)));
    }
  };

  // Stable sort: must-haves before nice-to-haves, otherwise preserve original order.
  const ordered = requirements
    .map((req, idx) => ({ req, idx }))
    .sort((a, b) => {
      const rank = (r: RequirementFit) => (r.importance === 'nice' ? 1 : 0);
      return rank(a.req) - rank(b.req) || a.idx - b.idx;
    });

  // Status counts for summary bar
  const matchedCount = requirements.filter((r) => r.status === 'Matched').length;
  const partialCount = requirements.filter((r) => r.status === 'Partial').length;
  const missingCount = requirements.filter((r) => r.status !== 'Matched' && r.status !== 'Partial').length;

  return (
    <div className="rounded-xl border border-white/5 bg-card p-6 space-y-4">
      {/* Header with expand/collapse toggle */}
      <div className="flex items-center justify-between border-b border-white/5 pb-3">
        <h3 className="text-sm font-semibold text-white">Skill Evidence Analysis</h3>
        <button
          onClick={toggleAll}
          className="flex items-center gap-1.5 text-[11px] text-gray-400 hover:text-gray-200 transition px-2 py-1 rounded-md hover:bg-white/5"
        >
          {allExpanded ? <EyeOff className="h-3 w-3" /> : <Eye className="h-3 w-3" />}
          {allExpanded ? 'Collapse All' : 'Expand All'}
        </button>
      </div>

      {/* Summary status bar */}
      <div className="flex items-center gap-3 flex-wrap">
        <div className="flex items-center gap-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 px-3 py-1">
          <CheckCircle2 className="h-3 w-3 text-emerald-400" />
          <span className="text-xs font-semibold text-emerald-400">{matchedCount}</span>
          <span className="text-[10px] text-emerald-400/70">Matched</span>
        </div>
        <div className="flex items-center gap-1.5 rounded-full bg-amber-500/10 border border-amber-500/20 px-3 py-1">
          <AlertCircle className="h-3 w-3 text-amber-400" />
          <span className="text-xs font-semibold text-amber-400">{partialCount}</span>
          <span className="text-[10px] text-amber-400/70">Partial</span>
        </div>
        <div className="flex items-center gap-1.5 rounded-full bg-rose-500/10 border border-rose-500/20 px-3 py-1">
          <XCircle className="h-3 w-3 text-rose-400" />
          <span className="text-xs font-semibold text-rose-400">{missingCount}</span>
          <span className="text-[10px] text-rose-400/70">Missing</span>
        </div>
        <span className="text-[10px] text-gray-500 ml-auto">
          {requirements.length} requirements total
        </span>
      </div>

      {/* Requirement cards */}
      <div className="space-y-2">
        {ordered.map(({ req, idx }) => (
          <RequirementCard
            key={idx}
            req={req}
            isOverClaimed={overClaimed.has(req.requirement.toLowerCase())}
            isExpanded={expandedSet.has(idx)}
            onToggle={() => toggleCard(idx)}
          />
        ))}
      </div>
    </div>
  );
};
