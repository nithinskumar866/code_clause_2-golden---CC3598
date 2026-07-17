import type { FC } from 'react';
import type { AnalysisReport } from '../../types';
import { getScoreBarBg, getScoreLabel, REQUIREMENT_STATUS_HEX } from './scoreColors';
import { ScoreRing } from '../charts/ScoreRing';
import { SubScoreRadar } from '../charts/SubScoreRadar';
import { RequirementDonut } from '../charts/RequirementDonut';
import type { DonutSegment } from '../charts/RequirementDonut';

interface ScoreOverviewProps {
  report: AnalysisReport;
}

interface SubScore {
  label: string;
  weight: string;
  value: number;
}

/** Overall score ring, weighted sub-score bars, radar and requirement donut. */
export const ScoreOverview: FC<ScoreOverviewProps> = ({ report }) => {
  const subScores: SubScore[] = [
    { label: 'Requirement Coverage', weight: '35%', value: report.coverage_score },
    { label: 'Experience Alignment', weight: '25%', value: report.experience_score },
    { label: 'Project Relevance', weight: '20%', value: report.project_score },
    { label: 'Evidence Confidence', weight: '15%', value: report.confidence_score },
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
      {/* Overall score + weighted sub-score bars */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="rounded-xl border border-white/5 bg-card p-5 flex flex-col items-center justify-center text-center">
          <span className="text-[10px] text-gray-500 font-semibold uppercase tracking-wider mb-3">Overall Score</span>
          <ScoreRing score={report.overall_score} />
          <span className="text-xs text-gray-400 mt-3 font-semibold uppercase tracking-wide">
            {getScoreLabel(report.overall_score)}
          </span>
        </div>

        <div className="md:col-span-2 rounded-xl border border-white/5 bg-card p-5 space-y-3.5">
          <h3 className="text-xs font-semibold text-white uppercase tracking-wider border-b border-white/5 pb-2">
            Explainable Metric Breakdown
          </h3>
          <div className="space-y-2.5 text-xs text-gray-300">
            {subScores.map((s) => (
              <div key={s.label} className="space-y-1">
                <div className="flex justify-between font-medium">
                  <span>
                    {s.label} (Weight {s.weight}):
                  </span>
                  <span className="font-semibold">{s.value}%</span>
                </div>
                <div className="w-full bg-white/5 rounded-full h-1.5 overflow-hidden">
                  <div
                    className={`h-full ${getScoreBarBg(s.value)}`}
                    style={{ width: `${Math.max(0, Math.min(100, s.value))}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
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
