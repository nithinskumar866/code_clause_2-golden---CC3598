import { useState, type FC } from 'react';
import { ChevronDown, Mail, Phone, MapPin, FileText, BadgeCheck, AlertTriangle } from 'lucide-react';
import { ResumeViewer } from './ResumeViewer';
import type { ChatCandidate } from '../../types';
import { Badge } from '../ui/Badge';
import { getScoreBarBg, getScoreColor } from '../analysis/scoreColors';

interface CandidateCardProps {
  candidate: ChatCandidate;
  rank: number;
}

/**
 * One ranked candidate in a chat answer.
 *
 * Everything shown is traceable: the percentage is decomposed into the three
 * deterministic sub-scores, and the verbatim resume excerpts that produced it are one
 * click away. A recruiter can always ask "why?" and get an answer from the resume
 * rather than from the model.
 */
export const CandidateCard: FC<CandidateCardProps> = ({ candidate: c, rank }) => {
  const [open, setOpen] = useState(false);
  const [viewing, setViewing] = useState(false);
  const name = c.name || c.filename || `Candidate ${c.resume_id}`;
  // "Tell me about X" returns an overview, not a ranking: no requested skills, so the
  // score bars would all read 0% and mean nothing.
  const isProfileAnswer = c.match_percentage === 0 && c.matched_skills.length === 0
    && c.missing_skills.length === 0;

  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.03] transition hover:border-white/20">
      <div className="flex items-start gap-3 p-3 sm:p-4">
        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-white/5 text-xs font-bold text-gray-300">
          {rank}
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <h4 className="truncate text-sm font-semibold text-white">{name}</h4>
            {c.title && <span className="truncate text-xs text-gray-400">· {c.title}</span>}
            {c.seniority_level && <Badge tone="info">{c.seniority_level}</Badge>}
          </div>

          {/* A card must never look like a match on a constraint it failed. When the
              recruiter asked for a location this candidate is not in, that is the
              first thing they need to see — not something buried in the reasoning. */}
          {!c.location_match && c.location_note && (
            <div className="mt-1 inline-flex items-center gap-1.5 rounded-md border border-amber-500/25 bg-amber-500/10 px-2 py-0.5 text-[11px] text-amber-300">
              <AlertTriangle className="h-3 w-3" /> {c.location_note}
            </div>
          )}

          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-gray-400">
            {c.total_years !== null && <span>{c.total_years} yrs experience</span>}
            {c.location && (
              <span className="inline-flex items-center gap-1">
                <MapPin className="h-3 w-3" /> {c.location}
              </span>
            )}
            {c.email && (
              <a href={`mailto:${c.email}`} className="inline-flex items-center gap-1 hover:text-indigo-300">
                <Mail className="h-3 w-3" /> {c.email}
              </a>
            )}
            {c.phone && (
              <a href={`tel:${c.phone.replace(/\s/g, '')}`} className="inline-flex items-center gap-1 hover:text-indigo-300">
                <Phone className="h-3 w-3" /> {c.phone}
              </a>
            )}
          </div>
        </div>

        {/* A profile answer has nothing to match against, so no score is shown —
            inventing one would imply a judgement the recruiter never asked for. */}
        {!isProfileAnswer && (
          <div
            className={`shrink-0 rounded-lg border px-2.5 py-1 text-sm font-bold ${getScoreColor(c.match_percentage)}`}
            title="Deterministic match score"
          >
            {c.match_percentage}%
          </div>
        )}
      </div>

      {/* VERDICT — the headline judgement, so a shortlist can be skimmed without
          reading a paragraph per person. */}
      {c.verdict && (
        <p className="px-3 pb-1.5 text-xs font-semibold text-white sm:px-4">{c.verdict}</p>
      )}

      {/* Why this candidate — generated from retrieved evidence, never invented. */}
      <p className="px-3 pb-2 text-xs leading-relaxed text-gray-300 sm:px-4">{c.reasoning}</p>

      {/* The body as POINTS, not prose. Each is one claim traceable to a section — a
          recruiter skims a shortlist, and a paragraph forces them to parse a sentence
          to find the one fact they care about. */}
      {c.highlights?.length > 0 && (
        <ul className="space-y-1 px-3 pb-3 sm:px-4">
          {c.highlights.map((point, i) => (
            <li key={i} className="flex gap-1.5 text-[11px] leading-relaxed text-gray-400">
              <span className="mt-[3px] h-1 w-1 shrink-0 rounded-full bg-indigo-400/60" />
              <span>{point}</span>
            </li>
          ))}
        </ul>
      )}

      {/* Read the resume the search actually matched on, or download the original. */}
      <div className="px-3 pb-3 sm:px-4">
        <button
          type="button"
          onClick={() => setViewing(true)}
          className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 px-2.5 py-1 text-[11px] text-gray-300 transition hover:border-indigo-500/40 hover:bg-indigo-500/10 hover:text-white"
        >
          <FileText className="h-3 w-3" /> View resume
        </button>
      </div>

      {viewing && (
        <ResumeViewer
          resumeId={c.resume_id}
          name={name}
          highlight={[...c.demonstrated_skills, ...c.listed_only_skills, ...c.matched_skills]}
          onClose={() => setViewing(false)}
        />
      )}

      {/* Score decomposition */}
      <div className={`grid grid-cols-3 gap-2 px-3 pb-3 sm:px-4 ${isProfileAnswer ? 'hidden' : ''}`}>
        {(
          [
            ['Skill coverage', c.breakdown.skill_coverage],
            ['Evidence strength', c.breakdown.evidence_strength],
            ['Experience fit', c.breakdown.experience_fit],
          ] as const
        ).map(([label, value]) => (
          <div key={label}>
            <div className="flex justify-between text-[10px] text-gray-500">
              <span className="truncate">{label}</span>
              <span className="font-semibold text-gray-400">{value}%</span>
            </div>
            <div className="mt-1 h-1 overflow-hidden rounded-full bg-white/5">
              <div className={`h-full rounded-full ${getScoreBarBg(value)}`} style={{ width: `${value}%` }} />
            </div>
          </div>
        ))}
      </div>

      {/* Skill chips: each carries how deeply it is proven and where. */}
      <div className="flex flex-wrap gap-1.5 px-3 pb-3 sm:px-4">
        {c.demonstrated_skills.map((s) => (
          <span
            key={`d-${s}`}
            className="inline-flex items-center gap-1 rounded-md border border-emerald-500/20 bg-emerald-500/10 px-1.5 py-0.5 text-[10px] font-medium text-emerald-400"
            title={`Proven in ${(c.skill_sections?.[s] || []).join(', ') || 'the resume'}`}
          >
            <BadgeCheck className="h-3 w-3" /> {s}
            {c.skill_depth?.[s] !== undefined && (
              <span className="text-emerald-300/70">{c.skill_depth[s]}%</span>
            )}
          </span>
        ))}
        {c.listed_only_skills.map((s) => (
          <span
            key={`l-${s}`}
            className="inline-flex items-center gap-1 rounded-md border border-amber-500/20 bg-amber-500/10 px-1.5 py-0.5 text-[10px] font-medium text-amber-400"
            title={`Only appears in ${(c.skill_sections?.[s] || []).join(', ') || 'a skills list'} — claimed, not demonstrated`}
          >
            <AlertTriangle className="h-3 w-3" /> {s}
            {c.skill_depth?.[s] !== undefined && (
              <span className="text-amber-300/70">{c.skill_depth[s]}%</span>
            )}
          </span>
        ))}
        {c.missing_skills.map((s) => (
          <span
            key={`m-${s}`}
            className="rounded-md border border-rose-500/20 bg-rose-500/10 px-1.5 py-0.5 text-[10px] font-medium text-rose-400"
            title="No evidence found in this resume"
          >
            no {s}
          </span>
        ))}
      </div>

      {/* A profile answer deliberately shows NO raw skill-token list. Dumping every
          extracted term produced "5M, BI, client, ETL, hubs, week" — noise harvested
          from bullet prose. The written summary above says what the resume shows, and
          the excerpts below prove it. */}

      {/* Verbatim evidence — the audit trail */}
      {c.evidence.length > 0 && (
        <div className="border-t border-white/5">
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            className="flex w-full items-center justify-between px-3 py-2 text-[11px] font-medium text-gray-400 transition hover:text-white sm:px-4"
          >
            <span className="inline-flex items-center gap-1.5">
              <FileText className="h-3 w-3" />
              {c.evidence.length} supporting excerpt{c.evidence.length > 1 ? 's' : ''} from the resume
            </span>
            <ChevronDown className={`h-3.5 w-3.5 transition-transform ${open ? 'rotate-180' : ''}`} />
          </button>

          {open && (
            <div className="space-y-2 px-3 pb-3 sm:px-4">
              {c.evidence.map((e, i) => (
                <blockquote
                  key={i}
                  className="rounded-lg border-l-2 border-indigo-500/40 bg-black/20 px-3 py-2 text-[11px] leading-relaxed text-gray-300"
                >
                  <div className="mb-1 flex flex-wrap items-center gap-2 text-[10px] text-gray-500">
                    <span className="font-semibold uppercase tracking-wide text-indigo-300/80">{e.skill}</span>
                    <span>· {e.section}</span>
                    <span>· page {e.page}</span>
                    <span>· similarity {e.similarity}</span>
                  </div>
                  {e.text}
                </blockquote>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
