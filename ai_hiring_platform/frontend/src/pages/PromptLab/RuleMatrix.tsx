import { useState, type FC } from 'react';
import { Check, X, Minus, ChevronDown, ChevronRight, Clock } from 'lucide-react';
import type { PromptCaseResult, RuleStatus, VariantResult } from '../../types';
import { Badge } from '../../components/ui/Badge';
import { Card } from '../../components/ui/Card';
import { pct, scoreTone } from './promptLabHelpers';

const STATUS_ICON: Record<RuleStatus, FC<{ className?: string }>> = {
  pass: Check,
  fail: X,
  na: Minus,
};

const STATUS_CLASS: Record<RuleStatus, string> = {
  pass: 'text-emerald-400',
  fail: 'text-rose-400',
  na: 'text-gray-600',
};

const STATUS_LABEL: Record<RuleStatus, string> = {
  pass: 'obeyed',
  fail: 'broken',
  na: 'not applicable',
};

/**
 * Where a prompt actually breaks, rule by rule.
 *
 * The headline pass rate says a prompt is 78% compliant; it never says which clause to
 * rewrite. This does: one row per rule, so a link policy that fails on every follow-up
 * is visibly a different problem from one case that fell over completely.
 *
 * A rule with nothing to say about a turn is shown as `—`, never as a pass. Counting
 * silence as success is how a thin case set flatters a bad prompt.
 */
export const RuleMatrix: FC<{ variant: VariantResult }> = ({ variant }) => (
  <Card className="overflow-hidden">
    <div className="flex items-center justify-between gap-3 border-b border-white/5 px-4 py-3">
      <h3 className="text-sm font-semibold text-white">Rule compliance</h3>
      <Badge tone={scoreTone(variant.score)}>{pct(variant.score)} of rules obeyed</Badge>
    </div>
    <div className="overflow-x-auto">
      <table className="w-full min-w-[520px] text-left text-sm">
        <thead>
          <tr className="border-b border-white/5 text-[10px] uppercase tracking-wider text-gray-500">
            <th className="px-4 py-2 font-semibold">Rule</th>
            <th className="px-3 py-2 text-right font-semibold">Obeyed</th>
            <th className="px-3 py-2 text-right font-semibold">Broken</th>
            <th className="px-3 py-2 text-right font-semibold">N/A</th>
            <th className="px-4 py-2 text-right font-semibold">Score</th>
          </tr>
        </thead>
        <tbody>
          {variant.by_rule.map((rule) => (
            <tr key={rule.rule} className="border-b border-white/5 last:border-0">
              <td className="px-4 py-2 text-gray-200">{rule.title}</td>
              <td className="px-3 py-2 text-right text-emerald-400">{rule.passed}</td>
              <td className="px-3 py-2 text-right text-rose-400">{rule.failed || ''}</td>
              <td className="px-3 py-2 text-right text-gray-600">{rule.not_applicable || ''}</td>
              <td className="px-4 py-2 text-right">
                <Badge tone={scoreTone(rule.score)}>{pct(rule.score)}</Badge>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </Card>
);

/** One case: what came back, and every rule's verdict on it with the reason. */
const CaseRow: FC<{ result: PromptCaseResult }> = ({ result }) => {
  const [open, setOpen] = useState(result.failed > 0);
  const Chevron = open ? ChevronDown : ChevronRight;

  return (
    <div className="border-b border-white/5 last:border-0">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center gap-3 px-4 py-3 text-left transition hover:bg-white/5"
      >
        <Chevron className="h-4 w-4 shrink-0 text-gray-500" />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-medium text-white">{result.name}</span>
          <span className="block truncate text-xs text-gray-500">{result.question}</span>
        </span>
        {result.latency_ms > 0 && (
          <span className="hidden items-center gap-1 text-[11px] text-gray-500 sm:flex">
            <Clock className="h-3 w-3" />
            {result.latency_ms} ms
          </span>
        )}
        <Badge tone={result.failed ? 'danger' : 'success'}>
          {result.failed ? `${result.failed} broken` : 'clean'}
        </Badge>
      </button>

      {open && (
        <div className="space-y-3 px-4 pb-4 pl-11">
          {result.error ? (
            <p className="rounded-lg border border-rose-500/20 bg-rose-500/10 px-3 py-2 text-xs text-rose-300">
              {result.error}
            </p>
          ) : (
            <pre className="whitespace-pre-wrap rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-xs text-gray-200">
              {result.answer || '(empty answer)'}
            </pre>
          )}

          <ul className="space-y-1.5">
            {result.rules.map((verdict) => {
              const Icon = STATUS_ICON[verdict.status];
              return (
                <li key={verdict.rule} className="flex items-start gap-2 text-xs">
                  <Icon className={`mt-0.5 h-3.5 w-3.5 shrink-0 ${STATUS_CLASS[verdict.status]}`} />
                  <span className="sr-only">{STATUS_LABEL[verdict.status]}</span>
                  <span className={verdict.status === 'na' ? 'text-gray-600' : 'text-gray-300'}>
                    <span className="font-medium text-gray-200">{verdict.title}</span> — {verdict.reason}
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
};

export const CaseResults: FC<{ variant: VariantResult }> = ({ variant }) => (
  <Card className="overflow-hidden">
    <div className="flex items-center justify-between gap-3 border-b border-white/5 px-4 py-3">
      <h3 className="text-sm font-semibold text-white">Answers under test</h3>
      <span className="text-xs text-gray-500">
        {variant.clean_cases}/{variant.total_cases} cases fully compliant
      </span>
    </div>
    {variant.cases.map((result) => (
      <CaseRow key={result.case_id} result={result} />
    ))}
  </Card>
);
