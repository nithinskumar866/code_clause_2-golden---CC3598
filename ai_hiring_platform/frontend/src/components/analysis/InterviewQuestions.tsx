import type { FC } from 'react';

interface InterviewQuestionsProps {
  questions: string[];
}

/** Recruiter interview guide focused on low-confidence claims. */
export const InterviewQuestions: FC<InterviewQuestionsProps> = ({ questions }) => (
  <div className="rounded-xl border border-white/5 bg-card p-5 space-y-4">
    <h3 className="text-sm font-semibold text-white border-b border-white/5 pb-2">
      Interview Questions (Depth vs Verification)
    </h3>
    <p className="text-xs text-gray-400">
      Recruiter guide to validate candidate's credentials specifically on low-confidence or weakly supported claims:
    </p>
    <div className="space-y-3">
      {questions.map((question, idx) => (
        <div
          key={idx}
          className="flex gap-3 bg-black/20 p-3 rounded-lg border border-white/5 text-xs text-gray-300"
        >
          <span className="flex items-center justify-center shrink-0 w-5 h-5 rounded-full bg-indigo-600 text-white font-semibold text-[10px]">
            {idx + 1}
          </span>
          <p className="leading-relaxed font-sans font-normal">{question}</p>
        </div>
      ))}
    </div>
  </div>
);
