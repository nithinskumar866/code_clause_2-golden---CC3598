import { useCallback, useEffect, useRef, useState, type FC, type FormEvent } from 'react';
import {
  Globe, Database, RotateCcw, Send, ShieldAlert, Sparkles, ExternalLink, ChevronRight,
} from 'lucide-react';
import type { WebHealth, WebTurn } from '../../types';
import { askWebStreaming, getWebHealth, WEB_BASE } from '../../api/webChat';
import { WebCompanyCard } from '../../components/chat/WebCompanyCard';
import { ChatModeToggle, type ChatMode } from '../../components/chat/ChatModeToggle';
import { WebCompanyManager } from './WebCompanyManager';
import { PageHeader } from '../../components/ui/PageHeader';
import { Button } from '../../components/ui/Button';
import { Card } from '../../components/ui/Card';
import { Spinner } from '../../components/ui/Spinner';
import { Badge } from '../../components/ui/Badge';

/**
 * Web mode — ask about what companies publish on their own websites.
 *
 * A sibling of `Chat.tsx` and `CompanyChat.tsx` rather than a branch inside either:
 * the three modes share a look and nothing else, and keeping them apart means neither
 * existing assistant's code path is touched by this one. This one also talks to a
 * different process entirely — the standalone `company_intel` crawler — so its failure
 * modes ("the crawler isn't running") are its own.
 *
 * Answering never crawls. Every claim here comes from a page that was crawled earlier
 * and stored, and each one links back to the URL it came from.
 */
const SUGGESTIONS = [
  'What does this company sell?',
  'Which of these companies talks about remote work?',
  'What does their careers page say about culture?',
  'Who are their customers?',
];

const newId = () => `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

interface WebChatProps {
  /** Switching to another knowledge base is done from the same toggle. */
  onModeChange: (mode: ChatMode) => void;
}

export const WebChat: FC<WebChatProps> = ({ onModeChange }) => {
  const [turns, setTurns] = useState<WebTurn[]>([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<WebHealth | null>(null);
  const [unreachable, setUnreachable] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  const loadHealth = useCallback(() => {
    getWebHealth()
      .then((h) => {
        setHealth(h);
        setUnreachable(false);
      })
      .catch(() => {
        setHealth(null);
        setUnreachable(true);
      });
  }, []);

  useEffect(loadHealth, [loadHealth]);

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
      const res = await askWebStreaming(text, (_stage, detail) =>
        setStage(detail ? `${detail}…` : 'Working…'),
      );
      setTurns((t) => [
        ...t,
        {
          id: newId(),
          role: 'assistant',
          text: res.answer,
          companies: res.companies,
          citations: res.citations,
          scope: res.scope,
          refused: res.refused,
          llmUsed: res.llm_used,
          evidenceCount: res.evidence_count,
          durationSeconds: res.duration_seconds,
        },
      ]);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'The crawler could not be reached.');
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
   * There is no server-side conversation memory in this mode — every question is
   * answered from the index alone — so a new chat is purely a local clear.
   */
  const onReset = () => {
    setTurns([]);
    setError(null);
  };

  const indexed = health?.qdrant.collections?.ci_content ?? 0;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Company Websites"
        description="Ask about what companies publish on their own sites. Every answer links to the page it came from."
        icon={<Globe className="h-5 w-5" />}
      />

      {/* Coverage — makes it obvious what the assistant can actually see. */}
      <Card className="flex flex-wrap items-center justify-between gap-3 p-3">
        <div className="flex flex-wrap items-center gap-3 text-xs text-gray-400">
          <span className="inline-flex items-center gap-1.5">
            <Database className="h-3.5 w-3.5 text-sky-400" />
            {unreachable ? (
              <span className="text-amber-300">Crawler not running at {WEB_BASE}</span>
            ) : health ? (
              <>
                <strong className="text-white">{health.companies}</strong> companies crawled
                <span className="text-gray-600">·</span>
                {indexed} indexed passages
              </>
            ) : (
              'Checking the crawler…'
            )}
          </span>
          {health && (
            <span
              className="text-[10px] text-gray-600"
              title={`Embeddings: ${health.embedding_model} (${health.embedding_dim}d)`}
            >
              {health.embedding_provider === 'ollama' ? 'GPU embeddings' : 'local embeddings'}
              {' · '}
              {health.llm_configured ? health.llm_provider : 'quoted sources'}
            </span>
          )}
          {/* A remote embedding endpoint that is asleep stops search as well as
              crawling — the vectors it produces are the only ones this collection
              can compare against — so it is called out rather than left to fail. */}
          {health && !health.embedding_reachable && (
            <Badge tone="danger">
              {health.embedding_detail || 'The embedding endpoint is unreachable'}
            </Badge>
          )}
          {health && health.companies === 0 && (
            <Badge tone="warning">No companies crawled yet — add one below</Badge>
          )}
        </div>
        <div className="flex items-center gap-2">
          <ChatModeToggle mode="web" onChange={onModeChange} />
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

      {unreachable ? (
        <Card className="p-6 text-center">
          <Globe className="mx-auto h-7 w-7 text-amber-400" />
          <h3 className="mt-3 text-sm font-semibold text-white">The crawler service is not running</h3>
          <p className="mx-auto mt-1 max-w-lg text-xs leading-relaxed text-gray-400">
            This mode is served by <code className="text-gray-300">company_intel</code>, a separate
            service. Start it and this page will connect on its own.
          </p>
          <pre className="mx-auto mt-3 w-fit rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-left text-[11px] text-gray-300">
            cd company_intel{'\n'}.\run.ps1
          </pre>
          <Button className="mt-4" variant="secondary" size="sm" onClick={loadHealth}>
            Try again
          </Button>
        </Card>
      ) : (
        <WebCompanyManager onChanged={loadHealth} />
      )}

      {/* Transcript */}
      {!unreachable && (
        <div className="space-y-5">
          {turns.length === 0 && (
            <Card className="p-6 text-center">
              <Sparkles className="mx-auto h-7 w-7 text-sky-400" />
              <h3 className="mt-3 text-sm font-semibold text-white">Ask about their websites</h3>
              <p className="mx-auto mt-1 max-w-lg text-xs leading-relaxed text-gray-400">
                I search the pages crawled from each company's own site — what they sell, who
                they serve, how they describe their culture and their leadership — and link every
                claim to the page it came from. I answer only from those pages, and I never
                browse live while answering.
              </p>
              <div className="mt-4 flex flex-wrap justify-center gap-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => void send(s)}
                    className="rounded-lg border border-white/10 px-3 py-1.5 text-xs text-gray-300 transition hover:border-sky-500/40 hover:bg-sky-500/10 hover:text-white"
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
                <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-sky-600 px-4 py-2.5 text-sm text-white shadow-lg shadow-sky-600/10">
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
                          from a pool-wide search — and quoted text from phrased prose. */}
                      <span>
                        {turn.scope === 'company' ? 'named company' : 'searched all companies'}
                      </span>
                      <span>·</span>
                      <span>{turn.llmUsed ? 'phrased by the model' : 'quoted from sources'}</span>
                      {turn.evidenceCount !== undefined && (
                        <>
                          <span>·</span>
                          <span>
                            {turn.evidenceCount === 1 ? '1 source' : `${turn.evidenceCount} sources`}
                          </span>
                        </>
                      )}
                      {turn.durationSeconds !== undefined && (
                        <>
                          <span>·</span>
                          <span>{Math.round(turn.durationSeconds * 1000)} ms</span>
                        </>
                      )}
                    </div>
                  )}
                </div>

                {/* The numbered sources the answer text refers to as [1], [2]. */}
                {turn.citations && turn.citations.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 pl-1">
                    {turn.citations.map((c) => (
                      <a
                        key={c.n}
                        href={c.page_url}
                        target="_blank"
                        rel="noreferrer noopener"
                        title={`${c.company_name} · ${c.page_title || c.page_url}`}
                        className="inline-flex max-w-[18rem] items-center gap-1 truncate rounded-md border border-white/10 px-2 py-1 text-[10px] text-gray-400 transition hover:border-sky-500/40 hover:text-sky-300"
                      >
                        <span className="font-semibold text-sky-400">[{c.n}]</span>
                        <span className="truncate">{c.company_name}</span>
                        <ExternalLink className="h-2.5 w-2.5 shrink-0" />
                      </a>
                    ))}
                  </div>
                )}

                {/* The evidence, folded away by default.
                    The chips above already let a reader check any claim in one click;
                    printing every retrieved passage in full buries the answer under the
                    material that supports it. Auditability is preserved — this is one
                    disclosure away, and nothing is omitted from it. */}
                {turn.companies && turn.companies.length > 0 && (
                  <details className="group pl-1">
                    <summary className="inline-flex cursor-pointer list-none items-center gap-1.5 text-[11px] text-gray-500 transition hover:text-gray-300">
                      <ChevronRight className="h-3 w-3 transition group-open:rotate-90" />
                      {turn.evidenceCount === 1
                        ? '1 source'
                        : `${turn.evidenceCount ?? 0} sources`}
                      <span className="text-gray-600">the answer was written from</span>
                    </summary>
                    <div className="mt-2.5 space-y-2.5">
                      {turn.companies.map((c, i) => (
                        <WebCompanyCard
                          key={`${turn.id}-${c.company_id}`}
                          company={c}
                          rank={i + 1}
                        />
                      ))}
                    </div>
                  </details>
                )}
              </div>
            ),
          )}

          {busy && (
            <div className="flex items-center gap-2 pl-1 text-xs text-gray-400">
              <Spinner />
              <span aria-live="polite">{stage || 'Searching the crawled pages…'}</span>
            </div>
          )}

          {error && (
            <div className="rounded-lg border border-rose-500/20 bg-rose-500/10 px-4 py-2.5 text-xs text-rose-300">
              {error}
            </div>
          )}

          <div ref={endRef} />
        </div>
      )}

      {/* Composer */}
      {!unreachable && (
        <form
          onSubmit={onSubmit}
          className="sticky bottom-0 -mx-1 bg-background/80 px-1 pb-1 pt-2 backdrop-blur"
        >
          <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-card p-2 focus-within:border-sky-500/50">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="e.g. what does Acme say about their pricing?"
              aria-label="Ask about company websites"
              disabled={busy}
              className="min-w-0 flex-1 bg-transparent px-2 text-sm text-white placeholder:text-gray-600 focus:outline-none disabled:opacity-60"
            />
            <Button type="submit" size="sm" loading={busy} disabled={!input.trim()}>
              <Send className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Ask</span>
            </Button>
          </div>
          <p className="mt-1.5 px-2 text-[10px] text-gray-600">
            Answers come only from crawled pages. This mode does not search resumes or your
            company spreadsheet, and it never browses the web while answering.
          </p>
        </form>
      )}
    </div>
  );
};
