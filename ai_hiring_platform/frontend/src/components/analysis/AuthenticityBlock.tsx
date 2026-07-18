import type { FC } from 'react';
import { ShieldCheck, ShieldAlert, ShieldX } from 'lucide-react';
import type { AuthenticityAssessment } from '../../types';

interface AuthenticityBlockProps {
  authenticity: AuthenticityAssessment;
}

/** Risk chip styling — Low=green, Medium=amber, High=red (mirrors score palette). */
const RISK_STYLE: Record<AuthenticityAssessment['keyword_stuffing_risk'], string> = {
  Low: 'text-emerald-400 border-emerald-500/20 bg-emerald-500/10',
  Medium: 'text-amber-400 border-amber-500/20 bg-amber-500/10',
  High: 'text-rose-400 border-rose-500/20 bg-rose-500/10',
};

const RISK_ICON = {
  Low: ShieldCheck,
  Medium: ShieldAlert,
  High: ShieldX,
} as const;

const credibilityBar = (score: number): string =>
  score >= 75 ? 'bg-emerald-500' : score >= 50 ? 'bg-amber-500' : 'bg-rose-500';

/**
 * Deterministic credibility / keyword-stuffing signal (F1). Surfaces how well the
 * candidate's claimed skills are corroborated by concrete Experience/Project
 * evidence, plus any skills listed but never demonstrated.
 */
export const AuthenticityBlock: FC<AuthenticityBlockProps> = ({ authenticity }) => {
  const { credibility_score, keyword_stuffing_risk, over_claimed_skills, corroboration_ratio, explanation } =
    authenticity;
  const RiskIcon = RISK_ICON[keyword_stuffing_risk] ?? ShieldAlert;

  return (
    <div className="rounded-xl border border-white/5 bg-card p-6 space-y-4">
      <div className="flex items-center justify-between border-b border-white/5 pb-2">
        <h3 className="text-sm font-semibold text-white">Authenticity &amp; Credibility</h3>
        <span
          className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-0.5 text-xs font-semibold ${RISK_STYLE[keyword_stuffing_risk]}`}
          title="Fraction of claimed skills listed but never demonstrated in Experience/Projects"
        >
          <RiskIcon className="h-3.5 w-3.5" /> Keyword-stuffing risk: {keyword_stuffing_risk}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Credibility score */}
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs text-gray-300 font-medium">
            <span>Credibility score</span>
            <span className="font-semibold">{credibility_score}%</span>
          </div>
          <div className="w-full bg-white/5 rounded-full h-1.5 overflow-hidden">
            <div
              className={`h-full ${credibilityBar(credibility_score)}`}
              style={{ width: `${Math.max(0, Math.min(100, credibility_score))}%` }}
            />
          </div>
          <p className="text-[10px] text-gray-500">
            Corroboration ratio {corroboration_ratio.toFixed(2)} — demonstrated ÷ claimed skills
          </p>
        </div>

        {/* Explanation */}
        <div className="md:col-span-2 text-xs text-gray-300 leading-relaxed">{explanation}</div>
      </div>

      {over_claimed_skills.length > 0 && (
        <div className="space-y-1.5 pt-1">
          <span className="text-[10px] text-gray-500 font-semibold uppercase tracking-wider">
            Listed but not demonstrated
          </span>
          <div className="flex flex-wrap gap-2">
            {over_claimed_skills.map((s) => (
              <span
                key={s}
                className="inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs font-medium uppercase text-rose-300 border border-rose-500/30 bg-rose-500/10"
                title="Appears only in listing sections — no Experience/Project evidence"
              >
                <ShieldAlert className="h-3 w-3" /> {s}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
