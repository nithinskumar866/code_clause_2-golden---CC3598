import type { FC } from 'react';
import type { AnalysisReport } from '../../types';
import { getScoreBarBg, getScoreLabel, REQUIREMENT_STATUS_HEX } from './scoreColors';
import { MatchScoreBreakdown } from './MatchScoreBreakdown';
import { ScoreRing } from '../charts/ScoreRing';
import { SubScoreRadar } from '../charts/SubScoreRadar';
import { RequirementDonut } from '../charts/RequirementDonut';
import type { DonutSegment } from '../charts/RequirementDonut';

interface ScoreOverviewProps {
  report: AnalysisReport;
}

interface SubScore {
  label: string;
  value: number;
}

/** Overall score ring, weighted sub-score bars, radar and requirement donut. */
export const ScoreOverview: FC<ScoreOverviewProps> = ({ report }) => {
  // Evidence sub-scores. These no longer decide the headline number — the Match
  // Score does — but they are what the strengths, weaknesses, learning roadmap and
  // interview questions are reasoned from, so they stay visible and unweighted.
  const subScores: SubScore[] = [
    { label: 'Requirement Coverage', value: report.coverage_score },
    { label: 'Experience Alignment', value: report.experience_score },
    { label: 'Project Relevance', value: report.project_score },
    { label: 'Evidence Confidence', value: report.confidence_score },
    { label: 'Resume Quality', value: report.quality_score },
  ];

  const radarData = subScores.map((s) => ({ label: s.label.split(' ')[0], value: s.value }));

  const statusCounts = report.requirements.reduce(
    (acc, r) => {
      const key = r.status === 'Matched' || r.status === 'Partial' ? r.status : 'Missing';
      acc[key] += 1;
      return acc;
    },
    { Matched: 0, Partial: 0, Missing: 0 } as Record<string, number>,
  );

  const donutSegments: DonutSegment[] = [
    { label: 'Matched', value: statusCounts.Matched, color: REQUIREMENT_STATUS_HEX.Matched },
    { label: 'Partial', value: statusCounts.Partial, color: REQUIREMENT_STATUS_HEX.Partial },
    { label: 'Missing', value: statusCounts.Missing, color: REQUIREMENT_STATUS_HEX.Missing },
  ];

  return (
    <div className="space-y-6">
      {/* Match Score: the ring, then the nine parameters that produced it. */}
      <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
        <div className="flex flex-col items-center justify-center rounded-xl border border-white/5 bg-card p-5 text-center">
          <span className="mb-3 text-[10px] font-semibold uppercase tracking-wider text-gray-500">
            Match Score
          </span>
          <ScoreRing score={report.overall_score} />
          <span className="mt-3 text-xs font-semibold uppercase tracking-wide text-gray-400">
            {report.match_score?.band ?? getScoreLabel(report.overall_score)}
          </span>
          <span className="mt-1 text-[10px] text-gray-600">
            {report.overall_score.toFixed(1)} / 100
          </span>
        </div>

        <div className="md:col-span-2">
          {report.match_score ? (
            <MatchScoreBreakdown match={report.match_score} />
          ) : (
            /* Reports evaluated before Match Score existed carry no parameters.
               Saying so is better than rendering nine empty rows. */
            <div className="h-full rounded-xl border border-white/5 bg-card p-5">
              <h3 className="border-b border-white/5 pb-2 text-xs font-semibold uppercase tracking-wider text-white">
                Match Score Breakdown
              </h3>
              <p className="pt-3 text-xs leading-relaxed text-gray-400">
                This analysis was produced before the nine-parameter Match Score existed, so only
                the overall number is available. Re-run the evaluation to get the full breakdown.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Evidence sub-scores — what the written report reasons from. */}
      <div className="rounded-xl border border-white/5 bg-card p-5 space-y-3.5">
        <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-white/5 pb-2">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-white">
            Evidence Sub-scores
          </h3>
          <span className="text-[10px] text-gray-500">
            Behind the written report · not part of the Match Score
          </span>
        </div>
        <div className="grid grid-cols-1 gap-x-6 gap-y-2.5 text-xs text-gray-300 sm:grid-cols-2">
          {subScores.map((s) => (
            <div key={s.label} className="space-y-1">
              <div className="flex justify-between font-medium">
                <span>{s.label}</span>
                <span className="font-semibold tabular-nums">{s.value}%</span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/5">
                <div
                  className={`h-full ${getScoreBarBg(s.value)}`}
                  style={{ width: `${Math.max(0, Math.min(100, s.value))}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Visualizations: radar profile + requirement outcome donut */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="rounded-xl border border-white/5 bg-card p-5 space-y-3">
          <h3 className="text-xs font-semibold text-white uppercase tracking-wider border-b border-white/5 pb-2">
            Sub-score Profile
          </h3>
          <div className="flex items-center justify-center px-6 pt-2 pb-4">
            <SubScoreRadar data={radarData} />
          </div>
        </div>

        <div className="rounded-xl border border-white/5 bg-card p-5 space-y-3">
          <h3 className="text-xs font-semibold text-white uppercase tracking-wider border-b border-white/5 pb-2">
            Requirement Outcomes
          </h3>
          <div className="flex items-center justify-center py-4">
            <RequirementDonut segments={donutSegments} centerLabel="Reqs" />
          </div>
        </div>
      </div>
    </div>
  );
};
