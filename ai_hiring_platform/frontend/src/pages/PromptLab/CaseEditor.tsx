import { useState, type FC } from 'react';
import { ChevronDown, ChevronRight, Trash2, Plus } from 'lucide-react';
import type { CaseExpectations, PromptCase } from '../../types';
import { Button } from '../../components/ui/Button';
import { Card } from '../../components/ui/Card';

/**
 * The switches that describe THIS TURN — not the prompt.
 *
 * Keeping them on the case is what lets a rewritten prompt be graded by the same suite:
 * whether a link is permissible on a follow-up is a fact about the conversation, and it
 * does not change because someone reworded the instruction that governs it.
 */
const TOGGLES: { key: keyof CaseExpectations; label: string; hint: string }[] = [
  { key: 'link_allowed', label: 'Link permitted', hint: 'A profile URL is acceptable on this turn.' },
  { key: 'link_required', label: 'Link required', hint: 'The user explicitly asked for the link.' },
  { key: 'years_known', label: 'Context states years', hint: 'The answer must give the exact figure.' },
  { key: 'skills_requested', label: 'Skills question', hint: 'Context skills and the skills URL are expected.' },
  { key: 'jobs_requested', label: 'Jobs asked for', hint: 'The user asked to search or list jobs.' },
  { key: 'list_allowed', label: 'List acceptable', hint: 'An enumerated answer suits this question.' },
];

const labelClass = 'mb-1.5 block text-[10px] font-semibold uppercase tracking-wider text-gray-400';
const fieldClass =
  'w-full rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-white placeholder:text-gray-600 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500';

interface Props {
  index: number;
  value: PromptCase;
  onChange: (next: PromptCase) => void;
  onRemove: () => void;
}

export const CaseEditor: FC<Props> = ({ index, value, onChange, onRemove }) => {
  const [open, setOpen] = useState(false);
  const Chevron = open ? ChevronDown : ChevronRight;
  const exp = value.expectations;

  const setExp = (patch: Partial<CaseExpectations>) =>
    onChange({ ...value, expectations: { ...exp, ...patch } });

  const historyText = value.history
    .map((turn) => `${turn.role}: ${turn.content}`)
    .join('\n');

  /** `role: text` per line — the shortest honest way to type a thread into a form. */
  const parseHistory = (text: string) =>
    text
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => {
        const at = line.indexOf(':');
        const role = at > 0 ? line.slice(0, at).trim().toLowerCase() : 'user';
        return {
          role: role === 'assistant' || role === 'bot' ? 'assistant' : 'user',
          content: at > 0 ? line.slice(at + 1).trim() : line,
        };
      });

  return (
    <Card className="overflow-hidden">
      <div className="flex items-center gap-2 px-3 py-2">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          className="flex min-w-0 flex-1 items-center gap-2 text-left"
        >
          <Chevron className="h-4 w-4 shrink-0 text-gray-500" />
          <span className="min-w-0">
            <span className="block truncate text-sm font-medium text-white">
              {value.name || `Case ${index + 1}`}
            </span>
            <span className="block truncate text-xs text-gray-500">
              {value.question || 'No question yet'}
            </span>
          </span>
        </button>
        <Button
          variant="ghost"
          size="sm"
          onClick={onRemove}
          aria-label={`Remove ${value.name || `case ${index + 1}`}`}
        >
          <Trash2 className="h-4 w-4" />
        </Button>
      </div>

      {open && (
        <div className="space-y-3 border-t border-white/5 px-3 py-3">
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <label className={labelClass} htmlFor={`case-name-${index}`}>Name</label>
              <input
                id={`case-name-${index}`}
                className={fieldClass}
                value={value.name || ''}
                onChange={(e) => onChange({ ...value, name: e.target.value })}
                placeholder="What this case is testing"
              />
            </div>
            <div>
              <label className={labelClass} htmlFor={`case-question-${index}`}>Question</label>
              <input
                id={`case-question-${index}`}
                className={fieldClass}
                value={value.question}
                onChange={(e) => onChange({ ...value, question: e.target.value })}
                placeholder="What the user asks"
              />
            </div>
          </div>

          <div>
            <label className={labelClass} htmlFor={`case-context-${index}`}>
              Retrieved context — the only facts the answer may use
            </label>
            <textarea
              id={`case-context-${index}`}
              rows={4}
              className={`${fieldClass} font-mono text-xs`}
              value={value.context}
              onChange={(e) => onChange({ ...value, context: e.target.value })}
              placeholder="PEOPLE DIRECTORY:&#10;Name=... | Role=... | LinkedIn=... | Skills=..."
            />
          </div>

          <div>
            <label className={labelClass} htmlFor={`case-history-${index}`}>
              Conversation so far — one `role: message` per line
            </label>
            <textarea
              id={`case-history-${index}`}
              rows={3}
              className={`${fieldClass} font-mono text-xs`}
              value={historyText}
              onChange={(e) => onChange({ ...value, history: parseHistory(e.target.value) })}
              placeholder="user: Who is Priya Raman?&#10;assistant: Priya Raman is a Senior Data Engineer. https://..."
            />
          </div>

          <div className="grid gap-3 sm:grid-cols-3">
            <div>
              <label className={labelClass} htmlFor={`case-lines-${index}`}>Max lines</label>
              <input
                id={`case-lines-${index}`}
                type="number"
                min={1}
                className={fieldClass}
                value={exp.max_lines ?? ''}
                onChange={(e) => setExp({ max_lines: e.target.value ? Number(e.target.value) : null })}
                placeholder="off"
              />
            </div>
            <div>
              <label className={labelClass} htmlFor={`case-chars-${index}`}>Max characters</label>
              <input
                id={`case-chars-${index}`}
                type="number"
                min={1}
                className={fieldClass}
                value={exp.max_chars ?? ''}
                onChange={(e) => setExp({ max_chars: e.target.value ? Number(e.target.value) : null })}
                placeholder="off"
              />
            </div>
            <div>
              <label className={labelClass} htmlFor={`case-years-${index}`}>Exact years in context</label>
              <input
                id={`case-years-${index}`}
                className={fieldClass}
                value={exp.expected_years ?? ''}
                onChange={(e) => setExp({ expected_years: e.target.value || null })}
                placeholder="e.g. 9"
              />
            </div>
          </div>

          <fieldset className="flex flex-wrap gap-x-4 gap-y-2">
            <legend className={labelClass}>What this turn permits</legend>
            {TOGGLES.map((toggle) => (
              <label
                key={toggle.key}
                title={toggle.hint}
                className="flex items-center gap-2 text-xs text-gray-300"
              >
                <input
                  type="checkbox"
                  className="h-3.5 w-3.5 rounded border-white/20 bg-black/40 accent-indigo-500"
                  checked={Boolean(exp[toggle.key])}
                  onChange={(e) => setExp({ [toggle.key]: e.target.checked } as Partial<CaseExpectations>)}
                />
                {toggle.label}
              </label>
            ))}
          </fieldset>

          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <label className={labelClass} htmlFor={`case-must-${index}`}>Must contain (comma separated)</label>
              <input
                id={`case-must-${index}`}
                className={fieldClass}
                value={exp.must_contain.join(', ')}
                onChange={(e) =>
                  setExp({ must_contain: e.target.value.split(',').map((s) => s.trim()).filter(Boolean) })
                }
              />
            </div>
            <div>
              <label className={labelClass} htmlFor={`case-mustnot-${index}`}>Must not contain</label>
              <input
                id={`case-mustnot-${index}`}
                className={fieldClass}
                value={exp.must_not_contain.join(', ')}
                onChange={(e) =>
                  setExp({ must_not_contain: e.target.value.split(',').map((s) => s.trim()).filter(Boolean) })
                }
              />
            </div>
          </div>
        </div>
      )}
    </Card>
  );
};

export const AddCaseButton: FC<{ onAdd: () => void }> = ({ onAdd }) => (
  <Button variant="secondary" size="sm" leftIcon={<Plus className="h-4 w-4" />} onClick={onAdd}>
    Add case
  </Button>
);
