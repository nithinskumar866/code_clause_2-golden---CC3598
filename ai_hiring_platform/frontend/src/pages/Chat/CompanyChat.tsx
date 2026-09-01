import { useCallback, useEffect, useRef, useState, type FC, type FormEvent } from 'react';
import { Building2, Database, RotateCcw, Send, ShieldAlert, Sparkles } from 'lucide-react';
import type { CompanyStoreStatus, CompanyTurn } from '../../types';
import { askCompanyStreaming, getCompanyStatus, resetCompanyChat } from '../../api/companyChat';
import { CompanyCard } from '../../components/chat/CompanyCard';
import { ChatModeToggle, type ChatMode } from '../../components/chat/ChatModeToggle';
import { PageHeader } from '../../components/ui/PageHeader';
import { Button } from '../../components/ui/Button';
import { Card } from '../../components/ui/Card';
import { Spinner } from '../../components/ui/Spinner';
import { Badge } from '../../components/ui/Badge';

/**
 * Company mode — ask about employers instead of candidates.
 *
 * A sibling of `Chat.tsx` rather than a branch inside it, deliberately: the two modes
 * share a look and nothing else, and keeping them apart means the candidate assistant's
 * code path is not touched at all by this module. Every fact on this screen comes from
 * the company records retrieved for the question; the model only phrases them.
 */
const SUGGESTIONS = [
  'Which companies have a strong learning and training culture?',
  'Who works in banking and financial services?',
  'Which companies do cloud and cybersecurity work?',
  'What products does TCS build?',
];

const newId = () => `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

interface CompanyChatProps {
  /** Switching back to candidate mode is done from the same toggle. */
  onModeChange: (mode: ChatMode) => void;
}

export const CompanyChat: FC<CompanyChatProps> = ({ onModeChange }) => {
  const [turns, setTurns] = useState<CompanyTurn[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<CompanyStoreStatus | null>(null);
  const sessionId = useRef<string>(`co-${newId()}`);
  const endRef = useRef<HTMLDivElement>(null);

  const loadStatus = useCallback(() => {
    getCompanyStatus()
      .then(setStatus)
      .catch(() => setStatus(null));
  }, []);

  useEffect(loadStatus, [loadStatus]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [turns, busy]);

  const send = async (message: string) => {
    const text = message.trim();
    if (!text || busy) return;

    setError(null);
    setInput('');
    setTurns((t) => [...t, { id: newId(), role: 'user', text }]);
    setBusy(true);
    setStage('Starting…');

    try {
      const res = await askCompanyStreaming(text, sessionId.current, (_stage, detail) =>
        setStage(detail ? `${detail}…` : 'Working…'),
      );
      setTurns((t) => [
        ...t,
        {
          id: newId(),
          role: 'assistant',
          text: res.answer,
          companies: res.companies,
          mode: res.mode,
          engine: res.engine,
          refused: res.refused,
          isFollowup: res.is_followup,
          elapsedMs: res.elapsed_ms,
        },
      ]);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'The company assistant could not be reached.');
    } finally {
      setStage('');
      setBusy(false);
    }
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void send(input);
  };

  const onReset = async () => {
    await resetCompanyChat(sessionId.current).catch(() => undefined);
    sessionId.current = `co-${newId()}`;
    setTurns([]);
    setError(null);
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Company Intelligence"
        description="Ask about the employers in your company database. Every answer shows the stored record behind it."
        icon={<Building2 className="h-5 w-5" />}
      />

      {/* Store coverage — makes it obvious what the assistant can see. */}
      <Card className="flex flex-wrap items-center justify-between gap-3 p-3">
        <div className="flex flex-wrap items-center gap-3 text-xs text-gray-400">
          <span className="inline-flex items-center gap-1.5">
            <Database className="h-3.5 w-3.5 text-emerald-400" />
            {status ? (
              <>
                <strong className="text-white">{status.companies}</strong> companies searchable
                <span className="text-gray-600">·</span>
                {status.points} indexed fields
              </>
            ) : (
              'Checking the company database…'
            )}
          </span>
          {status && !status.reachable && (
            <Badge tone="warning">{status.detail || 'The company database is unreachable'}</Badge>
          )}
          {status && status.reachable && status.companies === 0 && (
            <Badge tone="warning">
              No companies loaded yet — run <code>scripts/load_companies.py</code>
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-2">
          <ChatModeToggle mode="companies" onChange={onModeChange} />
          <Button
            variant="secondary"
            size="sm"
            leftIcon={<RotateCcw className="h-3.5 w-3.5" />}
            onClick={onReset}
          >
            New chat
          </Button>
        </div>
      </Card>

      {/* Transcript */}
      <div className="space-y-5">
        {turns.length === 0 && (
          <Card className="p-6 text-center">
            <Sparkles className="mx-auto h-7 w-7 text-emerald-400" />
            <h3 className="mt-3 text-sm font-semibold text-white">Ask about your companies</h3>
            <p className="mx-auto mt-1 max-w-lg text-xs leading-relaxed text-gray-400">
              I search every company record — what they do, who they sell to, what they build,
              their culture and their leadership — and show the exact stored text behind each
              answer. I answer only from those records.
            </p>
            <div className="mt-4 flex flex-wrap justify-center gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => void send(s)}
                  className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-gray-300 transition hover:border-emerald-500/40 hover:bg-emerald-500/10 hover:text-white"
                >
                  {s}
                </button>
              ))}
            </div>
          </Card>
        )}

        {turns.map((turn) =>
          turn.role === 'user' ? (
            <div key={turn.id} className="flex justify-end">
              <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-emerald-600 px-4 py-2.5 text-sm text-white shadow-lg shadow-emerald-600/10">
                {turn.text}
              </div>
            </div>
          ) : (
            <div key={turn.id} className="space-y-3">
              <div
                className={`max-w-[92%] whitespace-pre-wrap rounded-2xl rounded-bl-sm border px-4 py-3 text-sm leading-relaxed ${
                  turn.refused
                    ? 'border-amber-500/25 bg-amber-500/10 text-amber-200'
                    : 'border-white/10 bg-card text-gray-200'
                }`}
              >
                {turn.refused && (
                  <div className="mb-1.5 inline-flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-amber-400">
                    <ShieldAlert className="h-3.5 w-3.5" /> Guardrail
                  </div>
                )}
                {turn.text}
                {!turn.refused && (
                  <div className="mt-2 flex flex-wrap gap-2 text-[10px] text-gray-600">
                    {/* Which path answered, so a recruiter can tell a targeted lookup
                        from a pool-wide search — and deterministic prose from phrased. */}
                    <span>{turn.mode === 'company' ? 'named company' : 'searched all companies'}</span>
                    <span>·</span>
                    <span>{turn.engine === 'llm' ? 'phrased by the model' : 'deterministic'}</span>
                    {turn.isFollowup && (
                      <>
                        <span>·</span>
                        <span>follow-up</span>
                      </>
                    )}
                    {turn.elapsedMs !== undefined && (
                      <>
                        <span>·</span>
                        <span>{turn.elapsedMs} ms</span>
                      </>
                    )}
                  </div>
                )}
              </div>

              {turn.companies && turn.companies.length > 0 && (
                <div className="space-y-2.5">
                  {turn.companies.map((c, i) => (
                    <CompanyCard key={`${turn.id}-${c.company_id}`} company={c} rank={i + 1} />
                  ))}
                </div>
              )}
            </div>
          ),
        )}

        {busy && (
          <div className="flex items-center gap-2 pl-1 text-xs text-gray-400">
            <Spinner />
            <span aria-live="polite">{stage || 'Searching the company database…'}</span>
          </div>
        )}

        {error && (
          <div className="rounded-lg border border-rose-500/20 bg-rose-500/10 px-4 py-2.5 text-xs text-rose-300">
            {error}
          </div>
        )}

        <div ref={endRef} />
      </div>

      {/* Composer */}
      <form
        onSubmit={onSubmit}
        className="sticky bottom-0 -mx-1 bg-background/80 px-1 pb-1 pt-2 backdrop-blur"
      >
        <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-card p-2 focus-within:border-emerald-500/50">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="e.g. which companies build banking products?"
            aria-label="Ask about your companies"
            disabled={busy}
            className="min-w-0 flex-1 bg-transparent px-2 text-sm text-white placeholder:text-gray-600 focus:outline-none disabled:opacity-60"
          />
          <Button type="submit" size="sm" loading={busy} disabled={!input.trim()}>
            <Send className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Ask</span>
          </Button>
        </div>
        <p className="mt-1.5 px-2 text-[10px] text-gray-600">
          Answers come only from your loaded company records. This mode does not search resumes.
        </p>
      </form>
    </div>
  );
};
