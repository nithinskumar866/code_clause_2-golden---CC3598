import { useState, type FC } from 'react';
import { UserCheck, Sparkles, MessageSquare, Quote, AlertCircle, HelpCircle, Gavel } from 'lucide-react';
import type { InterviewQA, InterviewSimulation } from '../../types';
import { simulateInterview } from '../../api/interview';
import { Button } from '../ui/Button';
import { useToast } from '../ui/toast-context';
import { getScoreColor } from './scoreColors';

interface AiRecruiterPanelProps {
  analysisId: number;
}

const QaCard: FC<{ qa: InterviewQA; index: number }> = ({ qa, index }) => (
  <div className="rounded-xl border border-white/5 bg-black/20 p-5 space-y-3">
    <div className="flex items-start justify-between gap-3">
      <h4 className="flex items-start gap-2 text-sm font-semibold text-white">
        <MessageSquare className="mt-0.5 h-4 w-4 shrink-0 text-indigo-400" />
        <span>
          <span className="text-gray-500">Q{index + 1}.</span> {qa.question}
        </span>
      </h4>
      <span
        className={`shrink-0 rounded-md border px-2 py-0.5 text-[10px] font-bold ${getScoreColor(qa.confidence)}`}
        title="How well the resume supports a solid answer"
      >
        {qa.confidence}%
      </span>
    </div>

    <div className="space-y-1">
      <span className="text-[10px] font-semibold uppercase tracking-wider text-emerald-400">Ideal answer (from resume)</span>
      <p className="text-sm leading-relaxed text-gray-200">{qa.ideal_answer}</p>
    </div>

    {qa.evidence && (
      <div className="flex gap-2 rounded-lg bg-black/40 border border-white/5 p-3">
        <Quote className="h-3.5 w-3.5 shrink-0 text-indigo-300" />
        <p className="text-xs font-mono leading-relaxed text-indigo-300">{qa.evidence}</p>
      </div>
    )}

    {qa.missing_information && (
      <p className="flex items-start gap-1.5 text-xs text-amber-300/90">
        <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
        <span>
          <span className="font-semibold">Missing:</span> {qa.missing_information}
        </span>
      </p>
    )}

    {qa.follow_up_questions.length > 0 && (
      <div className="space-y-1">
        <span className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-gray-400">
          <HelpCircle className="h-3.5 w-3.5" /> Follow-up probes
        </span>
        <ul className="list-disc space-y-0.5 pl-6 text-xs text-gray-300">
          {qa.follow_up_questions.map((f, i) => (
            <li key={i}>{f}</li>
          ))}
        </ul>
      </div>
    )}

    <div className="flex items-start gap-1.5 border-t border-white/5 pt-2.5 text-xs text-gray-300">
      <Gavel className="mt-0.5 h-3.5 w-3.5 shrink-0 text-indigo-400" />
      <span>
        <span className="font-semibold text-indigo-300">Recruiter verdict:</span> {qa.recruiter_evaluation}
      </span>
    </div>
  </div>
);

/**
 * AI Recruiter: on demand, the AI both asks and answers each interview question
 * using only the candidate's resume evidence — so the recruiter can see whether
 * the resume can genuinely justify each skill.
 */
export const AiRecruiterPanel: FC<AiRecruiterPanelProps> = ({ analysisId }) => {
  const [sim, setSim] = useState<InterviewSimulation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const toast = useToast();

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await simulateInterview(analysisId);
      setSim(data);
      toast.success('AI interview ready', `${data.items.length} questions answered.`);
    } catch (err: any) {
      const msg = err?.response?.data?.message || err?.message || 'Failed to run the AI interview';
      setError(msg);
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rounded-xl border border-white/5 bg-card p-6 space-y-4">
      <div className="flex flex-col gap-3 border-b border-white/5 pb-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h3 className="flex items-center gap-2 text-sm font-semibold text-white">
            <UserCheck className="h-4 w-4 text-indigo-400" /> AI Recruiter — Interview Simulation
          </h3>
          <p className="mt-1 text-xs text-gray-500">
            The AI answers each interview question from the candidate's own resume, so you can judge whether they can
            back up their skills.
          </p>
        </div>
        <Button size="sm" leftIcon={<Sparkles className="h-4 w-4" />} loading={loading} onClick={run}>
          {sim ? 'Re-run' : 'Run AI interview'}
        </Button>
      </div>

      {error ? (
        <p className="text-xs text-rose-400">{error}</p>
      ) : loading ? (
        <p className="py-6 text-center text-xs text-gray-500">The AI recruiter is reviewing the resume…</p>
      ) : !sim ? (
        <p className="py-6 text-center text-xs italic text-gray-500">
          Click “Run AI interview” to have the AI question and answer for this candidate.
        </p>
      ) : (
        <div className="space-y-4">
          <span className="inline-flex items-center gap-1.5 rounded-md border border-white/10 bg-white/5 px-2 py-0.5 text-[10px] font-medium text-gray-400">
            Reasoned by: {sim.generated_by === 'llm' ? 'LLM' : 'deterministic engine'}
          </span>
          {sim.items.map((qa, i) => (
            <QaCard key={i} qa={qa} index={i} />
          ))}
        </div>
      )}
    </div>
  );
};
