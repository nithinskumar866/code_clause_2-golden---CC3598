import { useCallback, useEffect, useRef, useState, type FC, type FormEvent } from 'react';
import {
  Send, ShieldAlert, RotateCcw, Database, Sparkles, Info, MessagesSquare,
  HelpCircle, ArrowRight, SpellCheck, Scale, Layers, MapPin,
} from 'lucide-react';
import type { ChatSuggestion, ChatTurn, CorpusStatus } from '../../types';
import { askChatStreaming, getCorpusStatus, resetChat, syncCorpus } from '../../api/chat';
import { CandidateCard } from '../../components/chat/CandidateCard';
import { ChatModeToggle, type ChatMode } from '../../components/chat/ChatModeToggle';
import { CompanyChat } from './CompanyChat';
import { WebChat } from './WebChat';
import { getCompanyStatus } from '../../api/companyChat';
import { getWebHealth } from '../../api/webChat';
import { PageHeader } from '../../components/ui/PageHeader';
import { Button } from '../../components/ui/Button';
import { Card } from '../../components/ui/Card';
import { Spinner } from '../../components/ui/Spinner';
import { Badge } from '../../components/ui/Badge';

/** Starter questions that demonstrate what the assistant can do. */
const SUGGESTIONS = [
  'Need a candidate with 10 years of experience in Java',
  'Who has the strongest Python and AWS background?',
  'Find me a React developer with 5+ years',
  'Which candidates have Kubernetes in real projects?',
];

/**
 * The three embedding models the platform indexes with.
 *
 * All three are selectable here, not just two: whichever model answers a question is
 * the model whose index is searched, and a recruiter comparing answers in Model Lab
 * needs to be able to reproduce them here. They read the SAME shared store — the same
 * `documents.db` and `vectors-<model>.faiss` that upload and AI Analysis write — so
 * switching model switches which vectors are searched, never which resumes exist.
 */
const ENGINE_CHOICES = [
  { value: 'bge' as const, label: 'Fast', title: 'BGE-large · 1024d · in-process, ~30ms, always available' },
  { value: 'mxbai' as const, label: 'Deep', title: 'mxbai-large · 1024d · in-process, heavier, sharper separation' },
  { value: 'gpu' as const, label: 'GPU', title: 'nomic-embed-text · 768d · your GPU endpoint; falls back to BGE if unreachable' },
];

type EngineChoice = (typeof ENGINE_CHOICES)[number]['value'];

const newId = () => `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

/**
 * Recruiter chat — ask questions about the whole candidate pool.
 *
 * The assistant answers only from indexed resumes: every candidate card carries the
 * verbatim resume excerpts behind its score, and questions outside hiring (or that
 * would screen on personal attributes) are declined by backend guardrails.
 */
interface ChatProps {
  /** Hand a candidate to the AI Analysis page so JD evaluation reuses that flow. */
  onEvaluateCandidate?: (resumeId: number) => void;
}

export const Chat: FC<ChatProps> = ({ onEvaluateCandidate }) => {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState<string>('');
  // Which embedding model answers this question — and therefore which model's vectors
  // are searched. All three read the same shared store, so switching model changes how
  // the pool is ranked, never which resumes are in it.
  const [engine, setEngine] = useState<EngineChoice>('bge');
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<CorpusStatus | null>(null);
  const [syncing, setSyncing] = useState(false);
  // Which knowledge base answers. A hard switch: the company and web modes are separate
  // corpora with their own conversations, so nothing below this line runs while either
  // is selected.
  const [mode, setMode] = useState<ChatMode>('candidates');
  // Offered only when the company store is actually configured, reachable and loaded —
  // a toggle that leads to an assistant with nothing to say is worse than no toggle.
  const [companiesReady, setCompaniesReady] = useState<{ ok: boolean; reason: string }>({
    ok: false,
    reason: 'Checking the company database…',
  });
  // Web mode is served by the standalone `company_intel` crawler, a different process
  // on a different port. Probed the same way and for the same reason as the company
  // store: a toggle that leads to an assistant with nothing to say is worse than none.
  const [webReady, setWebReady] = useState<{ ok: boolean; reason: string }>({
    ok: false,
    reason: 'Checking the company crawler…',
  });
  const sessionId = useRef<string>(`hr-${newId()}`);
  const endRef = useRef<HTMLDivElement>(null);

  // Coverage is per model, so it is re-read whenever the selected model changes.
  const loadStatus = useCallback(() => {
    getCorpusStatus(engine)
      .then(setStatus)
      .catch(() => setStatus(null));
  }, [engine]);

  useEffect(loadStatus, [loadStatus]);

  // Probed once. A crawler that is not running is a normal state — every other mode
  // works exactly as before without it — so a failure here only disables the toggle.
  useEffect(() => {
    getWebHealth()
      .then((h) =>
        setWebReady({
          ok: h.qdrant.reachable && h.companies > 0,
          reason: !h.qdrant.reachable
            ? h.qdrant.detail || 'The crawler cannot reach its vector store.'
            : h.companies === 0
              ? 'No company websites crawled yet.'
              : '',
        }),
      )
      .catch(() =>
        setWebReady({
          ok: false,
          reason: 'The company crawler is not running (start company_intel).',
        }),
      );
  }, []);

  // Probed once. A missing or empty company store is a normal state — the platform
  // works exactly as before without it — so a failure here only disables the toggle.
  useEffect(() => {
    getCompanyStatus()
      .then((s) =>
        setCompaniesReady({
          ok: s.configured && s.reachable && s.companies > 0,
          reason: !s.configured
            ? 'The company database is not configured (QDRANT_URL / QDRANT_API_KEY).'
            : !s.reachable
              ? s.detail || 'The company database is unreachable.'
              : s.companies === 0
                ? 'No companies loaded yet — run scripts/load_companies.py.'
                : `Search ${s.companies} companies`,
        }),
      )
      .catch(() =>
        setCompaniesReady({ ok: false, reason: 'The company database is unavailable.' }),
      );
  }, []);

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

    // Real progress: each line below is pushed by the backend as that step actually
    // begins, so the message reflects what is happening rather than a timer's guess.
    setStage('Starting…');

    try {
      const res = await askChatStreaming(
        text,
        sessionId.current,
        (_stage, detail) => setStage(detail ? `${detail}…` : 'Working…'),
        5,
        true,
        engine,
      );
      setTurns((t) => [
        ...t,
        {
          id: newId(),
          role: 'assistant',
          text: res.answer,
          answerType: res.answer_type,
          fact: res.fact,
          clarification: res.clarification,
          candidates: res.candidates,
          intent: res.intent,
          suggestions: res.suggestions,
          diagnostics: res.diagnostics,
          refused: res.refused,
          needsClarification: res.needs_clarification,
          elapsedMs: res.elapsed_ms,
        },
      ]);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'The assistant could not be reached.');
    } finally {
      setStage('');
      setBusy(false);
    }
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void send(input);
  };

  /**
   * A suggestion either continues the conversation or hands the candidate to the
   * AI Analysis page — JD evaluation already exists there, so the chip carries the
   * selection across rather than duplicating that flow inside the chat.
   */
  const onSuggestion = (s: ChatSuggestion) => {
    if (s.action === 'navigate' && s.resume_id !== null) {
      onEvaluateCandidate?.(s.resume_id);
      return;
    }
    void send(s.query || s.label);
  };

  const onReset = async () => {
    await resetChat(sessionId.current).catch(() => undefined);
    sessionId.current = `hr-${newId()}`;
    setTurns([]);
    setError(null);
  };

  const onSync = async () => {
    setSyncing(true);
    try {
      await syncCorpus(false, engine);
      loadStatus();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Corpus sync failed.');
    } finally {
      setSyncing(false);
    }
  };

  // Each non-candidate mode is a whole separate screen against a separate corpus.
  // Returning early is what guarantees the candidate pipeline below is not merely
  // unused but unreached.
  if (mode === 'companies') {
    return <CompanyChat onModeChange={setMode} />;
  }
  if (mode === 'web') {
    return <WebChat onModeChange={setMode} />;
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Recruiter Assistant"
        description="Ask questions about your whole candidate pool. Every answer is backed by resume evidence."
        icon={<MessagesSquare className="h-5 w-5" />}
      />

      {/* Pool coverage — makes it obvious what the assistant can see. */}
      <Card className="flex flex-wrap items-center justify-between gap-3 p-3">
        <div className="flex flex-wrap items-center gap-3 text-xs text-gray-400">
          <span className="inline-flex items-center gap-1.5">
            <Database className="h-3.5 w-3.5 text-indigo-400" />
            {status ? (
              <>
                <strong className="text-white">{status.indexed_resumes}</strong> resumes searchable
                <span className="text-gray-600">·</span>
                {status.indexed_chunks} indexed passages
              </>
            ) : (
              'Checking the candidate pool…'
            )}
          </span>
          {/* Uploads index themselves in the background, so a gap here means indexing
              is still running — not that the recruiter forgot to press something. */}
          {status && !status.in_sync && (
            <Badge tone="warning">
              indexing {(status.database_resumes ?? 0) - status.indexed_resumes} more resume(s) in
              the background — searchable as they finish
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-2">
          <ChatModeToggle
            mode="candidates"
            onChange={setMode}
            companiesDisabled={!companiesReady.ok}
            companiesTitle={companiesReady.reason}
            webDisabled={!webReady.ok}
            webTitle={webReady.reason}
          />

          {/* Choose the search model per question. Both run on your own
              infrastructure — the local one in this process, the GPU one on your
              endpoint. If the GPU is unreachable the answer falls back to local and
              the meta line under it says which actually served. */}
          <div
            className="flex items-center rounded-lg border border-white/10 p-0.5"
            role="group"
            aria-label="Search model"
          >
            {ENGINE_CHOICES.map(({ value, label, title }) => (
              <button
                key={value}
                type="button"
                title={title}
                aria-pressed={engine === value}
                onClick={() => setEngine(value)}
                className={`rounded-md px-2.5 py-1 text-[11px] font-semibold transition ${
                  engine === value
                    ? 'bg-indigo-600 text-white'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {/* A manual catch-up, not the normal path — useful when a model's endpoint
              was down while its resumes arrived. */}
          <Button
            variant="ghost"
            size="sm"
            onClick={onSync}
            loading={syncing}
            title="Indexing runs automatically on upload. Use this to catch up a model that was unavailable."
          >
            Catch up this model
          </Button>
          <Button variant="secondary" size="sm" leftIcon={<RotateCcw className="h-3.5 w-3.5" />} onClick={onReset}>
            New chat
          </Button>
        </div>
      </Card>

      {/* Transcript */}
      <div className="space-y-5">
        {turns.length === 0 && (
          <Card className="p-6 text-center">
            <Sparkles className="mx-auto h-7 w-7 text-indigo-400" />
            <h3 className="mt-3 text-sm font-semibold text-white">Ask about your candidates</h3>
            <p className="mx-auto mt-1 max-w-lg text-xs leading-relaxed text-gray-400">
              I search every uploaded resume, rank the people who fit, and show the exact resume text
              behind each score. I answer only from those resumes — nothing else.
            </p>
            <div className="mt-4 flex flex-wrap justify-center gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => void send(s)}
                  className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-gray-300 transition hover:border-indigo-500/40 hover:bg-indigo-500/10 hover:text-white"
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
              <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-indigo-600 px-4 py-2.5 text-sm text-white shadow-lg shadow-indigo-600/10">
                {turn.text}
              </div>
            </div>
          ) : (
            <div key={turn.id} className="space-y-3">
              <div
                className={`max-w-[92%] rounded-2xl rounded-bl-sm border px-4 py-3 text-sm leading-relaxed ${
                  turn.refused
                    ? 'border-amber-500/25 bg-amber-500/10 text-amber-200'
                    : turn.needsClarification
                      ? 'border-sky-500/25 bg-sky-500/10 text-sky-100'
                      : 'border-white/10 bg-card text-gray-200'
                }`}
              >
                {turn.refused && (
                  <div className="mb-1.5 inline-flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-amber-400">
                    <ShieldAlert className="h-3.5 w-3.5" /> Guardrail
                  </div>
                )}
                {turn.needsClarification && (
                  <div className="mb-1.5 inline-flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-sky-400">
                    <HelpCircle className="h-3.5 w-3.5" /> Just to be sure
                  </div>
                )}
                {turn.answerType === 'explanation' && (
                  <div className="mb-1.5 inline-flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-violet-400">
                    <Scale className="h-3.5 w-3.5" /> Why I said that
                  </div>
                )}
                <p>{turn.text}</p>
              </div>

              {/* A question back, with the choices that answer it. Every option is a
                  real, measured outcome — the recruiter picks between facts. */}
              {turn.clarification && turn.clarification.options.length > 0 && (
                <div className="space-y-2 rounded-xl border border-sky-500/20 bg-sky-500/5 p-3">
                  <div className="flex flex-wrap gap-2">
                    {turn.clarification.options.map((o) => (
                      <button
                        key={`${o.label}-${o.resume_id ?? 'x'}`}
                        type="button"
                        disabled={busy}
                        onClick={() => onSuggestion(o)}
                        className="inline-flex items-center gap-1.5 rounded-lg border border-sky-500/30 bg-sky-500/10 px-3 py-1.5 text-xs text-sky-100 transition hover:bg-sky-500/20 disabled:opacity-50"
                      >
                        {o.label}
                      </button>
                    ))}
                  </div>
                  {turn.clarification.allow_all && (
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => void send(turn.clarification?.all_label || "I'm not sure, show me all of them")}
                      className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 px-3 py-1.5 text-xs text-gray-300 transition hover:border-sky-500/40 hover:text-white disabled:opacity-50"
                    >
                      <Layers className="h-3 w-3" />
                      {turn.clarification.all_label || 'Not sure — summarise all of them'}
                    </button>
                  )}
                </div>
              )}

              {/* A single fact about a single person, with the passage it came from.
                  A resume that simply does not say is shown as exactly that. */}
              {turn.fact && (
                <div
                  className={`space-y-2 rounded-xl border p-3 ${
                    turn.fact.found
                      ? 'border-emerald-500/20 bg-emerald-500/5'
                      : 'border-white/10 bg-card'
                  }`}
                >
                  <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
                    <span className="text-[10px] font-semibold uppercase tracking-wide text-gray-500">
                      {turn.fact.attribute}
                    </span>
                    <span className="text-sm font-semibold text-white">
                      {turn.fact.value ?? (turn.fact.found ? 'see the resume extract below' : 'not stated in the resume')}
                    </span>
                    <span className="text-[11px] text-gray-500">· {turn.fact.name}</span>
                  </div>
                  {turn.fact.evidence.slice(0, 2).map((e, i) => (
                    <blockquote
                      key={`${turn.id}-fact-${i}`}
                      className="border-l-2 border-white/10 pl-2.5 text-[11px] leading-relaxed text-gray-400"
                    >
                      <span className="mr-1.5 font-semibold uppercase tracking-wide text-gray-600">
                        {e.section}
                      </span>
                      {e.text.slice(0, 260)}
                      {e.text.length > 260 && '…'}
                    </blockquote>
                  ))}
                </div>
              )}

              {/* Spelling assumptions are stated, never applied silently. */}
              {turn.intent?.corrections?.length ? (
                <div className="inline-flex items-center gap-1.5 rounded-lg border border-indigo-500/20 bg-indigo-500/5 px-2.5 py-1 text-[11px] text-indigo-300">
                  <SpellCheck className="h-3 w-3" />
                  {turn.intent.corrections.map(([typed, fixed]) => (
                    <span key={typed}>
                      searched <strong className="text-indigo-200">{fixed}</strong> for “{typed}”
                    </span>
                  ))}
                </div>
              ) : null}

              {/* How the question was understood + what the search cost. */}
              {!turn.refused && turn.intent && (
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1 pl-1 text-[10px] text-gray-500">
                  <span className="inline-flex items-center gap-1">
                    <Info className="h-3 w-3" />
                    understood as:
                  </span>
                  {turn.intent.skills.length > 0 && <span>skills [{turn.intent.skills.join(', ')}]</span>}
                  {turn.intent.min_years !== null && <span>· min {turn.intent.min_years} yrs</span>}
                  {turn.intent.places?.length > 0 && (
                    <span className="inline-flex items-center gap-1">
                      · <MapPin className="h-3 w-3" />
                      {turn.intent.places.join(', ')}
                    </span>
                  )}
                  {turn.intent.attribute && <span>· asking for {turn.intent.attribute}</span>}
                  {turn.intent.named_candidates.length > 0 && (
                    <span>· about {turn.intent.named_candidates.join(', ')}</span>
                  )}
                  {turn.intent.is_followup && <span>· follow-up</span>}
                  {typeof turn.diagnostics?.pool_size === 'number' && (
                    <span>
                      · searched {String(turn.diagnostics.pool_size)} resumes
                      {typeof turn.diagnostics?.after_prefilter === 'number' &&
                        `, ${String(turn.diagnostics.after_prefilter)} passed filters`}
                    </span>
                  )}
                  {typeof turn.diagnostics?.embedding_engine === 'string' && (
                    <span>
                      · {turn.diagnostics.embedding_engine === 'gpu' ? 'GPU' : 'local'} search
                    </span>
                  )}
                  {turn.elapsedMs !== undefined && <span>· {turn.elapsedMs} ms</span>}
                  {typeof turn.diagnostics?.reasoning_engine === 'string' && (
                    <span>· {String(turn.diagnostics.reasoning_engine)} reasoning</span>
                  )}
                </div>
              )}

              {turn.candidates && turn.candidates.length > 0 && (
                <div className="space-y-2.5">
                  {turn.candidates.map((c, i) => (
                    <CandidateCard key={`${turn.id}-${c.resume_id}`} candidate={c} rank={i + 1} />
                  ))}
                </div>
              )}

              {/* Where to go next — keeps the recruiter moving without re-typing. */}
              {turn.suggestions && turn.suggestions.length > 0 && (
                <div className="space-y-1.5">
                  <p className="pl-1 text-[10px] font-semibold uppercase tracking-wide text-gray-600">
                    Suggested next steps
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {turn.suggestions.map((s) => (
                      <button
                        key={s.label}
                        type="button"
                        disabled={busy}
                        onClick={() => onSuggestion(s)}
                        className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs transition disabled:opacity-50 ${
                          s.action === 'navigate'
                            ? 'border-indigo-500/40 bg-indigo-500/10 text-indigo-200 hover:bg-indigo-500/20'
                            : 'border-white/10 text-gray-300 hover:border-indigo-500/40 hover:bg-indigo-500/10 hover:text-white'
                        }`}
                      >
                        {s.label}
                        {s.action === 'navigate' && <ArrowRight className="h-3 w-3" />}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ),
        )}

        {busy && (
          <div className="flex items-center gap-2 pl-1 text-xs text-gray-400">
            <Spinner />
            <span aria-live="polite">{stage || 'Searching the candidate pool…'}</span>
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
      <form onSubmit={onSubmit} className="sticky bottom-0 -mx-1 bg-background/80 px-1 pb-1 pt-2 backdrop-blur">
        <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-card p-2 focus-within:border-indigo-500/50">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="e.g. need a candidate with 10 years of experience in Java"
            aria-label="Ask about your candidates"
            disabled={busy}
            className="min-w-0 flex-1 bg-transparent px-2 text-sm text-white placeholder:text-gray-600 focus:outline-none disabled:opacity-60"
          />
          <Button type="submit" size="sm" loading={busy} disabled={!input.trim()}>
            <Send className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Ask</span>
          </Button>
        </div>
        <p className="mt-1.5 px-2 text-[10px] text-gray-600">
          Answers come only from indexed resumes. Screening on personal attributes such as age, gender or
          religion is declined.
        </p>
      </form>
    </div>
  );
};
