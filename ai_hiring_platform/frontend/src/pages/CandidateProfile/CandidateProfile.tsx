import { useCallback, useEffect, useState, type FC, type ReactNode } from 'react';
import { ArrowLeft, FileText, Briefcase, Clock, Star, ShieldCheck, Download, Sparkles } from 'lucide-react';
import type { AnalysisReport, HistoryRecord } from '../../types';
import { fetchHistoryReport } from '../../api/history';
import { downloadReportJSON, downloadReportPDF } from '../../lib/report';
import { WorkflowStatusControl } from '../../components/workflow/WorkflowStatusControl';
import { RecruiterNotes } from '../../components/notes/RecruiterNotes';
import { useToast } from '../../components/ui/toast-context';
import { getScoreColor, getScoreBarBg, getScoreLabel, classifyFit } from '../../components/analysis/scoreColors';
import type { FitCategory } from '../../components/analysis/scoreColors';
import { ScoreRing } from '../../components/charts/ScoreRing';
import { StrengthsWeaknesses } from '../../components/analysis/StrengthsWeaknesses';
import { InterviewQuestions } from '../../components/analysis/InterviewQuestions';
import { MissingSkills } from '../../components/analysis/MissingSkills';
import { CandidateProfileCard } from '../../components/analysis/CandidateProfileCard';
import { AuthenticityBlock } from '../../components/analysis/AuthenticityBlock';
import { AiRecruiterPanel } from '../../components/analysis/AiRecruiterPanel';
import { Card } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import type { BadgeTone } from '../../components/ui/Badge';
import { Button } from '../../components/ui/Button';
import { ProgressBar } from '../../components/ui/ProgressBar';
import { ErrorState } from '../../components/ui/ErrorState';
import { EmptyState } from '../../components/ui/EmptyState';
import { Skeleton } from '../../components/common/Skeleton';

interface CandidateProfileProps {
  record: HistoryRecord;
  onBack: () => void;
}

const CATEGORY_TONE: Record<FitCategory, BadgeTone> = {
  Selected: 'success',
  Borderline: 'warning',
  Rejected: 'danger',
};

const formatTimestamp = (iso: string): string => {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? 'Unknown date' : date.toLocaleString();
};

const hiringRisk = (score: number): { label: string; tone: BadgeTone } =>
  score >= 75
    ? { label: 'Low', tone: 'success' }
    : score >= 60
      ? { label: 'Medium', tone: 'warning' }
      : { label: 'High', tone: 'danger' };

const StarRating: FC<{ score: number }> = ({ score }) => {
  const filled = Math.max(0, Math.min(5, Math.round(score / 20)));
  return (
    <div className="flex gap-0.5" aria-label={`${filled} out of 5 stars`}>
      {[0, 1, 2, 3, 4].map((i) => (
        <Star key={i} className={`h-4 w-4 ${i < filled ? 'fill-amber-400 text-amber-400' : 'text-gray-600'}`} />
      ))}
    </div>
  );
};

const MetaRow: FC<{ icon: ReactNode; label: string; value: string }> = ({ icon, label, value }) => (
  <div className="flex items-start gap-2.5">
    <span className="mt-0.5 text-gray-500">{icon}</span>
    <div className="min-w-0">
      <dt className="text-[10px] font-semibold uppercase tracking-wider text-gray-500">{label}</dt>
      <dd className="truncate text-sm font-medium text-white">{value}</dd>
    </div>
  </div>
);

const ProfileSkeleton: FC = () => (
  <div className="space-y-6">
    <Skeleton className="h-40 w-full" />
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
      <Skeleton className="h-56 w-full" />
      <Skeleton className="h-56 w-full" />
    </div>
    <Skeleton className="h-48 w-full" />
  </div>
);

export const CandidateProfile: FC<CandidateProfileProps> = ({ record, onBack }) => {
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloadingPdf, setDownloadingPdf] = useState(false);
  const toast = useToast();

  const exportPdf = async () => {
    setDownloadingPdf(true);
    try {
      await downloadReportPDF(record.id);
    } catch (err: any) {
      toast.error(err?.message || 'Failed to export PDF');
    } finally {
      setDownloadingPdf(false);
    }
  };

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchHistoryReport(record.id);
      setReport(data);
    } catch (err: any) {
      console.error(err);
      setError('Failed to load this candidate report. Ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  }, [record.id]);

  useEffect(() => {
    load();
  }, [load]);

  const backButton = (
    <Button variant="secondary" size="sm" leftIcon={<ArrowLeft className="h-4 w-4" />} onClick={onBack}>
      Back to History
    </Button>
  );

  return (
    <div className="space-y-6 animate-fadeIn">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          {backButton}
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white">Candidate Profile</h1>
            <p className="text-xs text-gray-500">Analysis #{record.id}</p>
          </div>
        </div>
        {report && (
          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              leftIcon={<Download className="h-4 w-4" />}
              onClick={() => downloadReportJSON(report)}
            >
              Export JSON
            </Button>
            <Button
              variant="secondary"
              size="sm"
              loading={downloadingPdf}
              leftIcon={<Download className="h-4 w-4" />}
              onClick={exportPdf}
            >
              Export PDF
            </Button>
          </div>
        )}
      </div>

      {loading ? (
        <ProfileSkeleton />
      ) : error ? (
        <ErrorState title="Couldn't load candidate" message={error} onRetry={load} />
      ) : !report ? (
        <EmptyState title="No report available" description="This analysis has no stored report." />
      ) : (
        <ProfileBody report={report} record={record} />
      )}
    </div>
  );
};

const ProfileBody: FC<{ report: AnalysisReport; record: HistoryRecord }> = ({ report, record }) => {
  const category = classifyFit(report.overall_score);
  const risk = hiringRisk(report.overall_score);

  const requirements = report.requirements ?? [];
  const matched = requirements.filter((r) => r.status === 'Matched').map((r) => r.requirement);
  const partial = requirements.filter((r) => r.status === 'Partial').map((r) => r.requirement);

  const counts = requirements.reduce(
    (acc, r) => {
      const key = r.status === 'Matched' || r.status === 'Partial' ? r.status : 'Missing';
      acc[key] += 1;
      return acc;
    },
    { Matched: 0, Partial: 0, Missing: 0 } as Record<string, number>,
  );
  const totalReq = requirements.length || 1;

  const breakdown = [
    { label: 'Overall', value: report.overall_score },
    { label: 'Experience', value: report.experience_score },
    { label: 'Projects', value: report.project_score },
    { label: 'Quality', value: report.quality_score },
    { label: 'Coverage', value: report.coverage_score },
  ];

  const coverageBars = [
    { label: 'Matched', value: counts.Matched, bar: 'bg-emerald-500' },
    { label: 'Partial', value: counts.Partial, bar: 'bg-amber-500' },
    { label: 'Missing', value: counts.Missing, bar: 'bg-rose-500' },
  ];

  return (
    <div className="space-y-6">
      {/* Candidate summary hero */}
      <Card className="p-6">
        <div className="flex flex-col gap-6 md:flex-row md:items-center">
          <div className="flex flex-col items-center gap-3">
            <ScoreRing score={report.overall_score} />
            <StarRating score={report.overall_score} />
          </div>

          <div className="flex-1 space-y-4">
            <div className="flex flex-wrap items-center gap-2">
              <Badge tone={CATEGORY_TONE[category]}>{getScoreLabel(report.overall_score)}</Badge>
              <span className="inline-flex items-center gap-1.5 rounded-md border border-indigo-500/20 bg-indigo-500/10 px-2.5 py-1 text-xs font-semibold text-indigo-300">
                <Sparkles className="h-3.5 w-3.5" /> {report.recruiter_recommendation}
              </span>
              <span className="inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-semibold text-gray-300">
                <ShieldCheck className="h-3.5 w-3.5" /> Hiring risk:
                <Badge tone={risk.tone}>{risk.label}</Badge>
              </span>
            </div>

            <dl className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <MetaRow icon={<FileText className="h-4 w-4" />} label="Resume" value={record.resume_filename} />
              <MetaRow icon={<Briefcase className="h-4 w-4" />} label="Job Description" value={record.jd_filename} />
              <MetaRow icon={<Clock className="h-4 w-4" />} label="Analyzed" value={formatTimestamp(record.created_at)} />
              <MetaRow
                icon={<Star className="h-4 w-4" />}
                label="Overall Score"
                value={`${report.overall_score}%`}
              />
            </dl>
          </div>
        </div>
      </Card>

      {/* Editable recruiter workflow stage (persisted) */}
      <WorkflowStatusControl analysisId={record.id} />

      {/* Deterministic identity + seniority fit (F3) */}
      {report.candidate_profile && <CandidateProfileCard profile={report.candidate_profile} />}

      {/* Credibility / keyword-stuffing signal (F1) */}
      {report.authenticity && <AuthenticityBlock authenticity={report.authenticity} />}

      {/* Skill analysis + Score breakdown */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Skill Analysis */}
        <Card className="space-y-5 p-6">
          <h2 className="border-b border-white/5 pb-2 text-sm font-semibold text-white">Skill Analysis</h2>

          <div>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-emerald-400">Matched Skills</h3>
            {matched.length === 0 && partial.length === 0 ? (
              <p className="text-xs italic text-gray-500">No matched skills detected for this role.</p>
            ) : (
              <div className="flex flex-wrap gap-2">
                {matched.map((s) => (
                  <span
                    key={`m-${s}`}
                    className="rounded px-2.5 py-1 text-xs font-semibold uppercase text-emerald-400 border border-emerald-500/30 bg-emerald-500/10"
                  >
                    {s}
                  </span>
                ))}
                {partial.map((s) => (
                  <span
                    key={`p-${s}`}
                    className="rounded px-2.5 py-1 text-xs font-semibold uppercase text-amber-400 border border-amber-500/30 bg-amber-500/10"
                  >
                    {s} · partial
                  </span>
                ))}
              </div>
            )}
          </div>

          <div>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-gray-400">Skill Coverage</h3>
            <div className="space-y-2.5">
              {coverageBars.map((c) => (
                <div key={c.label} className="space-y-1">
                  <div className="flex justify-between text-xs text-gray-300">
                    <span>{c.label}</span>
                    <span className="font-semibold">{c.value}</span>
                  </div>
                  <ProgressBar
                    value={(c.value / totalReq) * 100}
                    barClass={c.bar}
                    label={`${c.label} requirements`}
                  />
                </div>
              ))}
            </div>
          </div>

          <MissingSkills skills={report.missing_skills ?? []} />
        </Card>

        {/* Score Breakdown */}
        <Card className="space-y-4 p-6">
          <h2 className="border-b border-white/5 pb-2 text-sm font-semibold text-white">Score Breakdown</h2>
          <div className="space-y-3.5">
            {breakdown.map((s) => (
              <div key={s.label} className="space-y-1.5">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-300">{s.label}</span>
                  <span
                    className={`rounded-md border px-2 py-0.5 text-xs font-bold ${getScoreColor(s.value)}`}
                  >
                    {s.value}%
                  </span>
                </div>
                <ProgressBar value={s.value} barClass={getScoreBarBg(s.value)} label={`${s.label} score`} />
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* AI Summary */}
      <Card className="space-y-2 p-6">
        <h2 className="flex items-center gap-1.5 text-sm font-semibold text-white">
          <Sparkles className="h-4 w-4 text-indigo-400" /> AI Summary
        </h2>
        <p className="text-sm leading-relaxed text-gray-300">{report.summary}</p>
      </Card>

      {/* Strengths & Weaknesses (reused) */}
      <StrengthsWeaknesses strengths={report.strengths ?? []} weaknesses={report.weaknesses ?? []} />

      {/* Recommended Interview Focus (reused) */}
      <InterviewQuestions questions={report.interview_questions ?? []} />

      {/* AI Recruiter — self-interview from resume evidence (on demand) */}
      <AiRecruiterPanel analysisId={record.id} />

      {/* Recruiter notes CRUD (persisted) */}
      <RecruiterNotes analysisId={record.id} />
    </div>
  );
};
