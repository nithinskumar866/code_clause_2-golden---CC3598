import type { FC } from 'react';
import type { AnalysisReport } from '../../types';
import { ScoreOverview } from './ScoreOverview';
import { RecruiterSummary } from './RecruiterSummary';
import { StrengthsWeaknesses } from './StrengthsWeaknesses';
import { SkillRelationships } from './SkillRelationships';
import { RequirementList } from './RequirementList';
import { MissingSkills } from './MissingSkills';
import { LearningRoadmap } from './LearningRoadmap';
import { InterviewQuestions } from './InterviewQuestions';
import { RejectionEmail } from './RejectionEmail';

interface HiringReportViewProps {
  report: AnalysisReport;
}

/** The full recruiter-facing hiring report (report view mode). */
export const HiringReportView: FC<HiringReportViewProps> = ({ report }) => (
  <div className="space-y-6">
    <ScoreOverview report={report} />

    <RecruiterSummary recommendation={report.recruiter_recommendation} summary={report.summary} />

    <StrengthsWeaknesses strengths={report.strengths} weaknesses={report.weaknesses} />

    <SkillRelationships relationships={report.skill_relationships} />

    <RequirementList requirements={report.requirements} />

    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <MissingSkills skills={report.missing_skills} />
      <LearningRoadmap items={report.learning_roadmap} />
    </div>

    <InterviewQuestions questions={report.interview_questions} />

    {report.rejection_email && <RejectionEmail email={report.rejection_email} />}
  </div>
);
