import type { FC } from 'react';
import { User, Briefcase, Clock, TrendingUp } from 'lucide-react';
import type { CandidateProfile } from '../../types';

interface CandidateProfileCardProps {
  profile: CandidateProfile;
}

/** Seniority-fit chip palette — Below=red, Meets=green, Exceeds=blue, Unknown=grey. */
const FIT_STYLE: Record<string, string> = {
  Below: 'text-rose-400 border-rose-500/20 bg-rose-500/10',
  Meets: 'text-emerald-400 border-emerald-500/20 bg-emerald-500/10',
  Exceeds: 'text-sky-400 border-sky-500/20 bg-sky-500/10',
  Unknown: 'text-gray-400 border-white/10 bg-white/5',
};

const Field: FC<{ icon: React.ReactNode; label: string; value: string }> = ({ icon, label, value }) => (
  <div className="flex items-start gap-2.5">
    <span className="mt-0.5 text-gray-500">{icon}</span>
    <div className="min-w-0">
      <dt className="text-[10px] font-semibold uppercase tracking-wider text-gray-500">{label}</dt>
      <dd className="truncate text-sm font-medium text-white">{value}</dd>
    </div>
  </div>
);

/**
 * Deterministic candidate identity + seniority fit (F3). Answers "who is this and
 * are they senior enough for THIS role" — orthogonal to pure skill coverage. Every
 * field is nullable; only present fields render.
 */
export const CandidateProfileCard: FC<CandidateProfileCardProps> = ({ profile }) => {
  const { name, title, total_years, seniority_level, required_years, seniority_fit, explanation } = profile;

  const fit = seniority_fit ?? 'Unknown';
  const yearsValue =
    total_years != null ? `${total_years}${seniority_level ? ` yrs · ${seniority_level}` : ' yrs'}` : null;

  return (
    <div className="rounded-xl border border-white/5 bg-card p-6 space-y-4">
      <div className="flex items-center justify-between border-b border-white/5 pb-2">
        <h3 className="text-sm font-semibold text-white">Candidate Profile &amp; Seniority</h3>
        <span
          className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-0.5 text-xs font-semibold ${
            FIT_STYLE[fit] ?? FIT_STYLE.Unknown
          }`}
          title="Estimated experience vs the JD's stated requirement"
        >
          <TrendingUp className="h-3.5 w-3.5" /> Seniority fit: {fit}
        </span>
      </div>

      <dl className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {name && <Field icon={<User className="h-4 w-4" />} label="Name" value={name} />}
        {title && <Field icon={<Briefcase className="h-4 w-4" />} label="Headline Role" value={title} />}
        {yearsValue && <Field icon={<Clock className="h-4 w-4" />} label="Experience" value={yearsValue} />}
        {required_years != null && (
          <Field icon={<TrendingUp className="h-4 w-4" />} label="JD Requires" value={`${required_years}+ yrs`} />
        )}
      </dl>

      {explanation && <p className="text-xs text-gray-400 leading-relaxed">{explanation}</p>}
    </div>
  );
};
