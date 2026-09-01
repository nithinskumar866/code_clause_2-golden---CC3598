import { useState, type FC } from 'react';
import { Building2, ChevronDown, Users } from 'lucide-react';
import type { CompanyResult } from '../../types';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';

/**
 * One company behind an answer, with the retrieved text that put it there.
 *
 * The matched fields are shown expanded and the rest of the record is one click away.
 * That ordering is the point: the recruiter sees WHY this company was returned before
 * they see everything else about it, which is what makes the answer auditable rather
 * than merely plausible. Every line here came out of the stored record — nothing on
 * this card was written by a language model.
 */
interface CompanyCardProps {
  company: CompanyResult;
  rank: number;
}

export const CompanyCard: FC<CompanyCardProps> = ({ company, rank }) => {
  const [open, setOpen] = useState(false);

  const matchedFields = new Set(company.matches.map((m) => m.field));
  const otherFields = Object.entries(company.fields ?? {}).filter(
    ([key]) => !matchedFields.has(key),
  );

  return (
    <Card className="overflow-hidden p-0">
      <div className="flex items-start justify-between gap-3 border-b border-white/5 p-3">
        <div className="flex min-w-0 items-start gap-2.5">
          <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-emerald-500/15 text-[11px] font-bold text-emerald-300">
            {rank}
          </span>
          <div className="min-w-0">
            <h4 className="flex items-center gap-1.5 truncate text-sm font-semibold text-white">
              <Building2 className="h-3.5 w-3.5 shrink-0 text-emerald-400" />
              {company.company_name}
            </h4>
            {company.industries.length > 0 && (
              <div className="mt-1.5 flex flex-wrap gap-1">
                {company.industries.slice(0, 5).map((industry) => (
                  <span
                    key={industry}
                    className="rounded border border-white/10 px-1.5 py-0.5 text-[10px] text-gray-400"
                  >
                    {industry}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
        {/* Relevance, not a percentage: it is a cosine score plus a discounted
            contribution from the company's other matching fields, so presenting it as
            a percentage would imply a precision it does not have. */}
        <Badge tone="success">{company.relevance.toFixed(2)}</Badge>
      </div>

      <div className="space-y-2 p-3">
        {company.matches.map((match) => (
          <div key={match.field}>
            <p className="text-[10px] font-semibold uppercase tracking-wide text-emerald-400/80">
              {match.label} · {match.score.toFixed(3)}
            </p>
            <p className="mt-0.5 text-xs leading-relaxed text-gray-300">{match.text}</p>
          </div>
        ))}

        {company.people.length > 0 && (
          <div>
            <p className="flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wide text-emerald-400/80">
              <Users className="h-3 w-3" /> People on record
            </p>
            <ul className="mt-0.5 space-y-0.5">
              {company.people.map((person, i) => (
                <li key={i} className="text-xs text-gray-300">
                  {Object.entries(person)
                    .map(([key, value]) => `${key}: ${String(value)}`)
                    .join(' · ')}
                </li>
              ))}
            </ul>
          </div>
        )}

        {otherFields.length > 0 && (
          <>
            <button
              type="button"
              onClick={() => setOpen((v) => !v)}
              aria-expanded={open}
              className="inline-flex items-center gap-1 text-[11px] text-gray-500 transition hover:text-gray-300"
            >
              <ChevronDown className={`h-3 w-3 transition ${open ? 'rotate-180' : ''}`} />
              {open ? 'Hide' : `Show ${otherFields.length} more field(s) on record`}
            </button>
            {open && (
              <div className="space-y-2 border-t border-white/5 pt-2">
                {otherFields.map(([key, value]) => (
                  <div key={key}>
                    <p className="text-[10px] font-semibold uppercase tracking-wide text-gray-600">
                      {key.replace(/_/g, ' ')}
                    </p>
                    <p className="mt-0.5 text-xs leading-relaxed text-gray-400">{value}</p>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </Card>
  );
};
