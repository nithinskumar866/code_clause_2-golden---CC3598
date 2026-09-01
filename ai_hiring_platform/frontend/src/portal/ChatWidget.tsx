import { useCallback, useEffect, useRef, useState, type FC, type FormEvent } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  ArrowRight, Bot, Brain, ChevronDown, Database, FileText, GripVertical, Loader2, Mic, MicOff,
  RefreshCw, Send, Sigma, Sparkles, Volume2, VolumeX, X,
} from 'lucide-react';
import { usePortalState } from './portal-context';
import { FileDrop } from './FileDrop';
import { MatchCard } from './MatchCard';
import { ApplyReview } from './ApplyReview';
import { ScreeningDialog } from './ScreeningDialog';
import { SuggestionCard } from './SuggestionCard';
import { Working, SkeletonRows } from './Working';
import { suggestJobs } from './api';
import { usePanelSize } from './usePanelSize';
import { useVoice } from './useVoice';
import type { JobSuggestionResult, ScoringMode } from './types';
import type { ConnectionState } from './useChatHub';

/**
 * The two scorers, as the candidate meets them.
 *
 * Both stay available on purpose. The computed one is instant and gives the same
 * answer every time; the reasoned one reads each posting properly and can see
 * that experience with one framework is evidence for a requirement written in
 * different words — at ten to twenty seconds a role. Which is better depends on
 * the question, so the choice is the candidate's rather than a default nobody
 * can see.
 */
const MODES: { id: ScoringMode; label: string; hint: string; Icon: typeof Sigma }[] = [
  {
    id: 'computed',
    label: 'Fast score',
    hint: 'Instant and repeatable. Similarity, skill overlap, title and experience, weighted.',
    Icon: Sigma,
  },
  {
    id: 'reasoned',
    label: 'AI evaluation',
    hint: 'Reads each posting requirement by requirement and judges the evidence behind every one. Slower — the closest three arrive first.',
    Icon: Brain,
  },
  {
    id: 'rag',
    label: 'RAG',
    hint: 'Retrieves the passages of your CV that bear on each requirement, then judges only those. Nothing is truncated, and every verdict cites the passage it came from.',
    Icon: Database,
  },
];

const DOT: Record<ConnectionState, { className: string; label: string }> = {
  idle: { className: 'bg-gray-500', label: 'Not connected yet' },
  connecting: { className: 'bg-amber-400 animate-pulse', label: 'Connecting' },
  connected: { className: 'bg-emerald-400', label: 'Connected' },
  reconnecting: { className: 'bg-amber-400 animate-pulse', label: 'Reconnecting' },
  disconnected: { className: 'bg-rose-500', label: 'Disconnected' },
};

/**
 * Markdown styling without the typography plugin.
 *
 * The assistant answers with tables, bold skill names and links, so the reply
 * has to render as markdown rather than as text; these child selectors are the
 * whole of what its output actually uses.
 */
const MARKDOWN =
  'text-sm leading-relaxed text-gray-200 ' +
  '[&_p]:mb-2 [&_p:last-child]:mb-0 ' +
  '[&_strong]:font-semibold [&_strong]:text-white ' +
  '[&_ul]:mb-2 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:mb-2 [&_ol]:list-decimal [&_ol]:pl-5 ' +
  '[&_li]:mb-0.5 ' +
  '[&_a]:text-indigo-400 [&_a]:underline hover:[&_a]:text-indigo-300 ' +
  '[&_code]:rounded [&_code]:bg-black/40 [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-[11px] ' +
  '[&_table]:mb-2 [&_table]:w-full [&_table]:border-collapse [&_table]:text-xs ' +
  '[&_th]:border [&_th]:border-white/10 [&_th]:bg-white/5 [&_th]:px-2 [&_th]:py-1 [&_th]:text-left ' +
  '[&_td]:border [&_td]:border-white/10 [&_td]:px-2 [&_td]:py-1 ' +
  '[&_h1]:mb-1 [&_h1]:text-base [&_h1]:font-semibold [&_h1]:text-white ' +
  '[&_h2]:mb-1 [&_h2]:text-sm [&_h2]:font-semibold [&_h2]:text-white ' +
  '[&_h3]:mb-1 [&_h3]:text-sm [&_h3]:font-semibold [&_h3]:text-white';

/**
 * The career assistant, as a popup available on every page.
 *
 * Mounted once by the app shell and never unmounted, which is what makes it a
 * popup rather than a page: closing it hides the panel, it does not end the
 * conversation or drop the socket. A reply that arrives while it is shut is
 * still there — and counted as unread — when it is reopened.
 */
export const ChatWidget: FC = () => {
  const {
    chatOpen, openChat, closeChat, unread,
    connection, turns, busy, send, openPage,
    resume, uploadResume, uploading, chatError, dismissChatError,
    health, reachable, newConversation, resetting,
    scoringMode, setScoringMode,
  } = usePortalState();

  const { size, startResize, resizing } = usePanelSize();
  const voice = useVoice();

  const [draft, setDraft] = useState('');
  // Which postings an apply-review is open for. Null means no review is open;
  // nothing can be sent while it is null, which is the whole safety property.
  const [reviewing, setReviewing] = useState<number[] | null>(null);

  /**
   * The posting whose screening questions are open, if any.
   *
   * A separate state from `reviewing` on purpose: the gate has to CLOSE before the
   * review opens, and one variable holding both would make "questions answered" and
   * "ready to send" the same fact. They are not — the whole point is that the first
   * has to happen before the second.
   *
   * Single-job only. Four questions per role across a five-role bulk apply is twenty
   * questions, which nobody finishes, so "apply to all" keeps going straight to the
   * review it always did.
   */
  const [screening, setScreening] = useState<number | null>(null);
  // Roles fetched to open with. null = still loading, so the panel can show a
  // skeleton of the right shape rather than jumping when they land.
  const [opening, setOpening] = useState<JobSuggestionResult | null>(null);
  const [highlightUpload, setHighlightUpload] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const replaceRef = useRef<HTMLInputElement>(null);

  /**
   * Applying and scoring both need a CV. Rather than disabling a button with no
   * explanation, the pinned uploader is drawn attention to — it is already on
   * screen, so this only has to point at it.
   */
  const askForResume = useCallback(() => {
    setHighlightUpload(true);
    window.setTimeout(() => setHighlightUpload(false), 2200);
  }, []);

  // Something real to look at on first open. Fetched rather than hard-coded:
  // these are the board's actual postings, so an empty board shows nothing
  // instead of a welcome message promising roles that do not exist.
  useEffect(() => {
    if (!chatOpen || resume || turns.length > 0 || opening !== null) return;
    let cancelled = false;
    suggestJobs(undefined, 3)
      .then(result => { if (!cancelled) setOpening(result); })
      .catch(() => {
        // The portal API being down is already reported at the top of the panel;
        // an empty opening is the right amount of noise to add here.
        if (!cancelled) setOpening({
          jobs: [], totalMatching: 0, appliedFilters: {},
          semanticMatching: false, embeddingModel: '', ranked: false,
        });
      });
    return () => { cancelled = true; };
  }, [chatOpen, resume, turns.length, opening]);

  // Dictated words land in the composer as they are recognised, so the candidate
  // can see what was heard while they are still speaking.
  useEffect(() => {
    if (voice.listening && voice.transcript) setDraft(voice.transcript);
  }, [voice.listening, voice.transcript]);

  // Finishing the sentence sends it. Holding a finished utterance hostage until
  // a mouse click would defeat the point of speaking to it in the first place —
  // and the words were visible in the composer the whole time.
  const wasListening = useRef(false);
  useEffect(() => {
    if (wasListening.current && !voice.listening) {
      const spoken = voice.transcript.trim();
      if (spoken.length > 2) {
        void send(spoken);
        setDraft('');
      }
    }
    wasListening.current = voice.listening;
  }, [voice.listening, voice.transcript, send]);

  // Reads each finished reply aloud, once. Keyed on the turn id rather than a
  // flag, so a re-render — or a later turn patching this one — cannot make it
  // repeat itself.
  const spokenTurns = useRef(new Set<string>());
  useEffect(() => {
    if (!voice.speakReplies) return;
    const latest = turns[turns.length - 1];
    if (!latest || latest.role !== 'assistant' || latest.streaming) return;
    if (!latest.content.trim() || spokenTurns.current.has(latest.id)) return;

    spokenTurns.current.add(latest.id);
    voice.speak(latest.content);
  }, [turns, voice]);

  // Pin to the newest message as the reply streams in.
  useEffect(() => {
    if (!chatOpen) return;
    const element = scrollRef.current;
    if (element) element.scrollTop = element.scrollHeight;
  }, [turns, chatOpen]);

  useEffect(() => {
    if (chatOpen) inputRef.current?.focus();
  }, [chatOpen]);

  // Escape closes the panel. Registered only while open so it cannot swallow
  // Escape from the dialogs and drawers in the rest of the app.
  useEffect(() => {
    if (!chatOpen) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') closeChat();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [chatOpen, closeChat]);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!draft.trim()) return;
    void send(draft);
    setDraft('');
  };

  const latestMatches = [...turns].reverse().find(turn => turn.matches)?.matches ?? null;
  const dot = DOT[connection];

  return (
    <>
      {/* Launcher. Stays mounted while the panel is open on desktop so the
          badge position never jumps; hidden under the full-screen mobile sheet. */}
      <button
        type="button"
        onClick={openChat}
        aria-label={unread > 0 ? `Open career assistant, ${unread} new replies` : 'Open career assistant'}
        aria-expanded={chatOpen}
        className={`fixed bottom-5 right-5 z-40 flex h-14 w-14 items-center justify-center rounded-full bg-indigo-600 text-white shadow-xl shadow-indigo-600/30 transition hover:bg-indigo-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 focus-visible:ring-offset-2 focus-visible:ring-offset-background ${
          chatOpen ? 'scale-0 opacity-0' : 'scale-100 opacity-100'
        }`}
      >
        <Sparkles className="h-6 w-6" />
        {unread > 0 && (
          <span className="absolute -right-0.5 -top-0.5 flex h-5 min-w-5 items-center justify-center rounded-full border-2 border-background bg-rose-500 px-1 text-[10px] font-bold">
            {unread > 9 ? '9+' : unread}
          </span>
        )}
      </button>

      {chatOpen && (
        <section
          role="dialog"
          aria-label="Career assistant"
          // The dragged size applies from `sm` up only. On a phone the panel is
          // full-screen, where a resize handle would be both useless and in the way.
          style={
            typeof window !== 'undefined' && window.innerWidth >= 640
              ? { width: size.width, height: size.height }
              : undefined
          }
          className={`fixed inset-0 z-50 flex flex-col border-white/10 bg-card shadow-2xl sm:inset-auto sm:bottom-5 sm:right-5 sm:max-h-[calc(100vh-2.5rem)] sm:max-w-[calc(100vw-2.5rem)] sm:rounded-2xl sm:border ${
            resizing ? 'select-none' : 'animate-fadeIn'
          }`}
        >
          {/* Resize grip, top-left — the corner that grows the panel into open
              screen rather than pushing it past the edge it is anchored to. */}
          <button
            type="button"
            onPointerDown={startResize}
            aria-label="Resize assistant"
            title="Drag to resize"
            className="absolute -left-1 -top-1 z-10 hidden h-6 w-6 cursor-nwse-resize items-center justify-center rounded-full text-gray-600 transition hover:text-indigo-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 sm:flex"
          >
            <GripVertical className="h-3.5 w-3.5 rotate-45" />
          </button>

          {/* -- header -- */}
          <header className="flex shrink-0 items-center gap-2.5 border-b border-border px-4 py-3">
            <div className="rounded-lg bg-indigo-600 p-1.5 text-white">
              <Bot className="h-4 w-4" />
            </div>
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-semibold text-white">Career Assistant</p>
              <p className="flex items-center gap-1.5 text-[10px] text-gray-500">
                <span className={`h-1.5 w-1.5 rounded-full ${dot.className}`} aria-hidden="true" />
                {dot.label}
                {health && ` · ${health.jobCount} roles`}
              </p>
            </div>
            {/* Read replies aloud. Hidden entirely where the browser cannot do it,
                rather than offered as a button that does nothing. */}
            {voice.canSpeak && (
              <button
                type="button"
                onClick={voice.toggleSpeakReplies}
                aria-label={voice.speakReplies ? 'Turn off spoken replies' : 'Read replies aloud'}
                aria-pressed={voice.speakReplies}
                title={voice.speakReplies ? 'Spoken replies on' : 'Spoken replies off'}
                className={`rounded-lg p-1.5 transition hover:bg-white/5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
                  voice.speakReplies ? 'text-indigo-400' : 'text-gray-400 hover:text-white'
                }`}
              >
                {voice.speakReplies
                  ? <Volume2 className={`h-4 w-4 ${voice.speaking ? 'animate-pulse' : ''}`} />
                  : <VolumeX className="h-4 w-4" />}
              </button>
            )}

            {/* Clears the conversation, not the CV — starting a fresh question
                should not mean uploading your resume again. */}
            <button
              type="button"
              onClick={() => void newConversation()}
              disabled={resetting || turns.length === 0}
              aria-label="Start a new chat"
              title="New chat — clears this conversation, keeps your CV"
              className="rounded-lg p-1.5 text-gray-400 transition hover:bg-white/5 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:opacity-30"
            >
              <RefreshCw className={`h-4 w-4 ${resetting ? 'animate-spin' : ''}`} />
            </button>
            <button
              type="button"
              onClick={closeChat}
              aria-label="Minimise assistant"
              className="rounded-lg p-1.5 text-gray-400 transition hover:bg-white/5 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
            >
              <ChevronDown className="hidden h-4 w-4 sm:block" />
              <X className="h-4 w-4 sm:hidden" />
            </button>
          </header>

          {/* -- how roles get scored --
              Outside the transcript so it is visible while reading any answer,
              and immediately above it so the connection between the setting and
              the cards below is not something anyone has to be told. Switching
              does not re-score what is already on screen: those answers keep the
              mode they were produced under, and each card says which. */}
          <div
            role="radiogroup"
            aria-label="How roles are scored"
            className="flex shrink-0 items-center gap-1 border-b border-border bg-black/20 px-3 py-2"
          >
            <span className="mr-1 text-[10px] font-semibold uppercase tracking-wider text-gray-600">
              Scoring
            </span>
            {MODES.map(({ id, label, hint, Icon }) => (
              <button
                key={id}
                type="button"
                role="radio"
                aria-checked={scoringMode === id}
                disabled={busy}
                title={hint}
                onClick={() => setScoringMode(id)}
                className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-[11px] font-semibold transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:cursor-not-allowed disabled:opacity-50 ${
                  scoringMode === id
                    ? id === 'reasoned'
                      ? 'bg-violet-500/15 text-violet-300 ring-1 ring-violet-500/30'
                      : id === 'rag'
                        ? 'bg-teal-500/15 text-teal-300 ring-1 ring-teal-500/30'
                        : 'bg-white/10 text-white ring-1 ring-white/15'
                    : 'text-gray-500 hover:bg-white/5 hover:text-gray-300'
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                {label}
              </button>
            ))}
          </div>

          {/* -- transcript -- */}
          <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
            {reachable === false && (
              <p className="rounded-lg border border-rose-500/20 bg-rose-500/10 p-2.5 text-xs text-rose-300">
                The job portal API is not reachable. Start it with <code>dotnet run</code> in{' '}
                <code>job_portal/backend/JobPortal.Api</code>.
              </p>
            )}

            {health && !health.semanticMatching && (
              <p className="rounded-lg border border-amber-500/20 bg-amber-500/10 p-2.5 text-xs text-amber-300">
                <strong>Lexical matching.</strong> No embedding model is configured, so roles are
                matched on wording rather than meaning. Set <code>Ollama__EmbedModel</code> to enable
                semantic matching.
              </p>
            )}

            {voice.voiceError && (
              <p className="rounded-lg border border-amber-500/20 bg-amber-500/10 p-2.5 text-xs text-amber-300">
                {voice.voiceError}
              </p>
            )}

            {chatError && (
              <p className="flex items-start gap-2 rounded-lg border border-rose-500/20 bg-rose-500/10 p-2.5 text-xs text-rose-300">
                <span className="flex-1">{chatError}</span>
                <button type="button" onClick={dismissChatError} aria-label="Dismiss error">
                  <X className="h-3.5 w-3.5" />
                </button>
              </p>
            )}

            {turns.length === 0 && (
              <div className="space-y-3">
                <div className="space-y-1.5 py-2 text-center text-xs text-gray-500">
                  <p>
                    {resume
                      ? 'Ask for roles, or narrow by location, work mode or salary.'
                      : 'Ask about roles, or add your CV below to be scored against them.'}
                  </p>
                  <p>You can also ask where something is — “take me to the job board”.</p>
                </div>

                {/* Opened with something to look at rather than a blank box. These
                    are real postings from the board, not a scripted welcome. */}
                {!resume && (
                  opening === null
                    ? <SkeletonRows rows={2} />
                    : opening.jobs.length > 0 && (
                      <>
                        <p className="text-[11px] text-gray-500">
                          {opening.ranked ? 'Roles on the board:' : 'Newest on the board:'}
                        </p>
                        {opening.jobs.map(job => (
                          <SuggestionCard key={job.id} job={job} onNeedResume={askForResume} />
                        ))}
                      </>
                    )
                )}
              </div>
            )}

            {turns.map(turn => (
              <div key={turn.id} className={turn.role === 'user' ? 'flex justify-end' : ''}>
                {turn.role === 'user' ? (
                  <p className="max-w-[85%] rounded-2xl rounded-br-sm bg-indigo-600 px-3 py-2 text-sm text-white">
                    {turn.content}
                  </p>
                ) : (
                  <div className="max-w-[92%] space-y-2">
                    {/* The reasoning is shown, not hidden behind a spinner: an
                        answer whose working is visible is the whole point of
                        this platform. */}
                    {turn.thoughts.length > 0 && (
                      <ul className="space-y-1 rounded-lg border border-white/5 bg-black/20 p-2">
                        {turn.thoughts.map((thought, index) => (
                          <li key={index} className="flex gap-2 text-[11px] leading-snug">
                            <span
                              className={`shrink-0 font-semibold uppercase tracking-wide ${
                                thought.stage === 'degraded' ? 'text-amber-400' : 'text-indigo-400'
                              }`}
                            >
                              {thought.stage}
                            </span>
                            <span className="text-gray-400">{thought.text}</span>
                          </li>
                        ))}
                      </ul>
                    )}

                    {(turn.content || turn.streaming) && (
                      <div className={`rounded-2xl rounded-bl-sm bg-white/5 px-3 py-2 ${MARKDOWN}`}>
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{turn.content}</ReactMarkdown>
                        {turn.streaming && (
                          <span className="ml-0.5 inline-block h-3.5 w-1.5 animate-pulse bg-indigo-400 align-middle" />
                        )}
                      </div>
                    )}

                    {/* Pages the assistant offered. Buttons rather than prose:
                        the destination is already known, so making the user
                        retype its name would be asking a question twice. */}
                    {turn.options && turn.options.length > 0 && (
                      <div className="flex flex-wrap gap-1.5">
                        {turn.options.map(option => (
                          <button
                            key={option.pageId}
                            type="button"
                            onClick={() => openPage(option)}
                            title={option.description}
                            className="inline-flex items-center gap-1 rounded-lg border border-indigo-500/30 bg-indigo-500/10 px-2 py-1 text-xs font-semibold text-indigo-300 transition hover:bg-indigo-500/20 hover:text-indigo-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
                          >
                            {option.label}
                            <ArrowRight className="h-3 w-3" />
                          </button>
                        ))}
                      </div>
                    )}

                    {/* Questions worth asking next. They do double duty: nobody
                        guesses unprompted that "what should I learn across all
                        these roles" is answerable, and when a reference could not
                        be pinned to a posting these become the ask-back — tapping
                        a real role beats re-typing the name that just failed to
                        resolve. Every chip is built server-side from the shortlist
                        that exists, so none can offer an answer we cannot give. */}
                    {!turn.streaming && turn.followUps.length > 0 && (
                      <div className="flex flex-wrap gap-1.5">
                        {turn.followUps.map(followUp => (
                          <button
                            key={followUp.message}
                            type="button"
                            disabled={busy}
                            onClick={() => void send(followUp.message)}
                            title={followUp.message}
                            className="inline-flex items-center gap-1 rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-xs font-medium text-gray-300 transition hover:border-indigo-500/40 hover:bg-indigo-500/10 hover:text-indigo-200 disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
                          >
                            {followUp.label}
                          </button>
                        ))}
                      </div>
                    )}

                    {/* Roles found without a CV: no score, because there is
                        nobody to score yet. */}
                    {turn.suggestions && turn.suggestions.jobs.length > 0 && (
                      <div className="space-y-2">
                        {!turn.suggestions.ranked && (
                          <p className="text-[10px] text-gray-500">
                            Newest first — I could not rank these against what you asked.
                          </p>
                        )}
                        {turn.suggestions.jobs.map(job => (
                          <SuggestionCard key={job.id} job={job} onNeedResume={askForResume} />
                        ))}
                        {turn.suggestions.totalMatching > turn.suggestions.jobs.length && (
                          <p className="text-[10px] text-gray-500">
                            {turn.suggestions.totalMatching} match in total.
                          </p>
                        )}
                      </div>
                    )}

                    {turn.matches && turn.matches.matches.length > 0 && (
                      <div className="space-y-2">
                        {turn.matches.matches.map(match => (
                          <MatchCard
                            key={match.job.id}
                            match={match}
                            compact
                            // Applying to ONE role goes through the screening
                            // questions first; the review opens only once they pass.
                            onApply={resume ? (jobId) => setScreening(jobId) : undefined}
                          />
                        ))}

                        {/* Still judging.
                            Shown under the cards that HAVE landed rather than in
                            place of them: the whole reason the shortlist arrives
                            in pieces is so there is something to read while the
                            rest is worked out, and a spinner covering the answer
                            would undo that. */}
                        {!turn.matches.complete && turn.matches.pending > 0 && (
                          <p className="flex items-center gap-2 rounded-lg border border-violet-500/15 bg-violet-500/[0.07] px-2.5 py-2 text-[11px] text-violet-200/80">
                            <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-violet-400" />
                            Reading {turn.matches.pending} more{' '}
                            {turn.matches.pending === 1 ? 'role' : 'roles'} against your CV. They
                            appear here as each one is judged — keep asking in the meantime.
                          </p>
                        )}

                        {resume && turn.matches.matches.length > 1 && (
                          <button
                            type="button"
                            onClick={() => setReviewing(turn.matches!.matches.map(m => m.job.id))}
                            className="w-full rounded-lg border border-indigo-500/30 bg-indigo-500/10 px-3 py-2 text-xs font-semibold text-indigo-300 transition hover:bg-indigo-500/20"
                          >
                            Apply to all {turn.matches.matches.length} — review first
                          </button>
                        )}
                      </div>
                    )}

                    {turn.error && (
                      <p className="rounded-lg border border-rose-500/20 bg-rose-500/10 p-2.5 text-xs text-rose-300">
                        {turn.error}
                      </p>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* -- the CV, pinned --
              Outside the scroller on purpose. This used to live at the top of the
              transcript, which meant it scrolled out of sight the moment anything
              was said and disappeared entirely once a CV existed — leaving no way
              to change it. It is the thing the whole conversation depends on, so
              it stays on screen. */}
          <div className="shrink-0 border-t border-border px-3 pt-3">
            {uploading ? (
              <Working
                label="Reading your CV…"
                detail="Extracting your profile and embedding it. Large PDFs take a few seconds."
              />
            ) : resume ? (
              <div className="flex items-center gap-2 rounded-lg border border-white/10 bg-black/20 px-2.5 py-2">
                <FileText className="h-3.5 w-3.5 shrink-0 text-indigo-400" />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-[11px] font-medium text-gray-200">
                    {resume.candidateName || resume.filename}
                  </p>
                  <p className="truncate text-[10px] text-gray-500">
                    {[resume.currentTitle, resume.skills.length ? `${resume.skills.length} skills` : null]
                      .filter(Boolean).join(' · ') || resume.filename}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => replaceRef.current?.click()}
                  className="shrink-0 rounded-md border border-white/10 px-2 py-1 text-[10px] font-semibold text-gray-300 transition hover:bg-white/5 hover:text-white"
                >
                  Replace
                </button>
                <input
                  ref={replaceRef}
                  type="file"
                  accept=".pdf,.docx,.txt,.md"
                  className="hidden"
                  onChange={event => {
                    const file = event.target.files?.[0];
                    event.target.value = '';
                    if (file) void uploadResume(file);
                  }}
                />
              </div>
            ) : (
              <FileDrop
                compact
                label="Add your CV to be scored and to apply"
                hint="PDF, DOCX, TXT or MD — or click to browse"
                onFile={(file) => void uploadResume(file)}
                className={highlightUpload ? 'ring-2 ring-indigo-400' : ''}
              />
            )}
          </div>

          {/* -- composer -- */}
          <form onSubmit={submit} className="flex shrink-0 items-center gap-2 px-3 py-3">
            <input
              ref={inputRef}
              value={draft}
              onChange={event => setDraft(event.target.value)}
              disabled={busy}
              aria-label="Message the career assistant"
              placeholder={resume ? 'Remote only? Why this score? Where is…?' : 'Ask a question, or take me to…'}
              className="min-w-0 flex-1 rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-white placeholder:text-gray-500 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:opacity-50"
            />
            {voice.canListen && (
              <button
                type="button"
                onClick={voice.listening ? voice.stopListening : voice.startListening}
                disabled={busy}
                aria-label={voice.listening ? 'Stop dictating' : 'Ask by voice'}
                aria-pressed={voice.listening}
                title={voice.listening ? 'Listening — click to stop' : 'Ask by voice'}
                className={`shrink-0 rounded-lg p-2.5 transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:opacity-40 ${
                  voice.listening
                    ? 'bg-rose-500/15 text-rose-400 ring-1 ring-rose-500/40'
                    : 'text-gray-400 hover:bg-white/5 hover:text-white'
                }`}
              >
                {voice.listening
                  ? <Mic className="h-4 w-4 animate-pulse" />
                  : <MicOff className="h-4 w-4" />}
              </button>
            )}
            <button
              type="submit"
              disabled={busy || !draft.trim()}
              aria-label="Send"
              className="shrink-0 rounded-lg bg-indigo-600 p-2.5 text-white transition hover:bg-indigo-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <Send className="h-4 w-4" />
            </button>
          </form>

          {/* Matches also appear on the board; this is the reminder that they do. */}
          {latestMatches && latestMatches.matches.length > 0 && (
            <p className="shrink-0 border-t border-border px-4 py-2 text-[10px] text-gray-500">
              {latestMatches.totalCandidates} role(s) considered · showing the closest{' '}
              {latestMatches.matches.length}
            </p>
          )}
        </section>
      )}

      {/* Rendered outside the panel so it is never clipped by the popup, and so
          closing the chat cannot leave a half-confirmed submission behind. */}

      {/* The gate. Closes itself and hands the job on only when the server says the
          answers pass — declining leaves the review unopened, which is the point. */}
      {screening !== null && resume && (
        <ScreeningDialog
          jobId={screening}
          onPassed={() => {
            setReviewing([screening]);
            setScreening(null);
          }}
          onClose={() => setScreening(null)}
        />
      )}

      {reviewing && resume && (
        <ApplyReview
          resumeId={resume.id}
          jobIds={reviewing}
          onClose={() => setReviewing(null)}
        />
      )}
    </>
  );
};
