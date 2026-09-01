import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { api } from '../api/client';
import { useChatHub } from '../hooks/useChatHub';
import { MatchCard } from '../components/MatchCard';
import { usePortalState } from '../state/PortalState';
import type { Health, ResumeProfile } from '../types';

// Type declarations for Web Speech API
interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: (event: SpeechRecognitionEvent) => void;
  onend: () => void;
  onerror: (event: SpeechRecognitionErrorEvent) => void;
}

interface SpeechRecognitionEvent extends Event {
  resultIndex: number;
  results: SpeechRecognitionResultList;
}

interface SpeechRecognitionResultList {
  length: number;
  item(index: number): SpeechRecognitionResult;
  [index: number]: SpeechRecognitionResult;
}

interface SpeechRecognitionResult {
  isFinal: boolean;
  length: number;
  item(index: number): SpeechRecognitionAlternative;
  [index: number]: SpeechRecognitionAlternative;
}

interface SpeechRecognitionAlternative {
  transcript: string;
  confidence: number;
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string;
  message: string;
}

interface WindowWithSpeechRecognition extends Window {
  SpeechRecognition: new () => SpeechRecognition;
  webkitSpeechRecognition: new () => SpeechRecognition;
}

// Voice recognition hook
function useVoiceRecognition() {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  useEffect(() => {
    const win = window as unknown as WindowWithSpeechRecognition;
    if ('webkitSpeechRecognition' in win || 'SpeechRecognition' in win) {
      const SpeechRecognition = win.SpeechRecognition || win.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = false;
      recognitionRef.current.interimResults = true;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.onresult = (event: SpeechRecognitionEvent) => {
        let finalTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          }
        }
        if (finalTranscript) {
          setTranscript(finalTranscript);
        }
      };

      recognitionRef.current.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current.onerror = (event: SpeechRecognitionErrorEvent) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
      };
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.abort();
      }
    };
  }, []);

  const startListening = useCallback(() => {
    if (recognitionRef.current && !isListening) {
      setTranscript('');
      recognitionRef.current.start();
      setIsListening(true);
    }
  }, [isListening]);

  const stopListening = useCallback(() => {
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
  }, [isListening]);

  const win = window as unknown as WindowWithSpeechRecognition;
  const isSupported = 'webkitSpeechRecognition' in win || 'SpeechRecognition' in win;

  return { isListening, transcript, startListening, stopListening, isSupported };
}

/**
 * The session id is kept in localStorage so a refresh resumes the same
 * conversation rather than silently starting a new one and losing the resume the
 * candidate already uploaded.
 */
function useSessionId(): string {
  return useMemo(() => {
    const existing = localStorage.getItem('portal.sessionId');
    if (existing) return existing;
    const created = crypto.randomUUID();
    localStorage.setItem('portal.sessionId', created);
    return created;
  }, []);
}

export function ChatPage({ health }: { health: Health | null }) {
  const sessionId = useSessionId();
  const { applyAction } = usePortalState();
  const { state, turns, filters, busy, send, analyzeResume, queuedMessages, hasQueuedMessages } = useChatHub(sessionId, applyAction);
  const { isListening, transcript, startListening, stopListening, isSupported } = useVoiceRecognition();

  const [resume, setResume] = useState<ResumeProfile | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [draft, setDraft] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-fill draft when voice transcript is available
  useEffect(() => {
    if (transcript) {
      setDraft(transcript);
    }
  }, [transcript]);

  // Restore whatever the session already knows, so a refresh does not present an
  // empty chat to someone who has already uploaded a resume.
  useEffect(() => {
    api.getSession(sessionId)
      .then(session => setResume(session.resume))
      .catch(() => { /* A fresh session simply has nothing to restore. */ });
  }, [sessionId]);

  useEffect(() => {
    const element = scrollRef.current;
    if (element) element.scrollTop = element.scrollHeight;
  }, [turns]);

  const onDrop = useCallback(async (files: File[]) => {
    const file = files[0];
    if (!file) return;

    setUploading(true);
    setError(null);
    try {
      const profile = await api.uploadResume(file);
      setResume(profile);
      // Upload over REST, narrate over the hub: the file does not belong in a
      // WebSocket frame, but the analysis is what the candidate watches.
      await analyzeResume(profile.id);
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : 'The resume could not be read.');
    } finally {
      setUploading(false);
    }
  }, [analyzeResume]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: false,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt', '.md'],
    },
  });

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    void send(draft);
    setDraft('');
  };

  const latestMatches = [...turns].reverse().find(turn => turn.matches)?.matches ?? null;

  return (
    <div className="page">
      <h1>Find your next role</h1>
      <p className="lede">
        Drop in your resume and I will match it against every open role on the board,
        then tell you where you stand on each one.
      </p>

      {state === 'disconnected' && (
        <div className="banner error">
          Not connected to the API. Start it with <code>dotnet run</code> in
          <code> job_portal/backend/JobPortal.Api</code>.
        </div>
      )}
      {state === 'reconnecting' && (
        <div className="banner warning">
          <strong>Reconnecting…</strong> Your messages will be sent once the connection is restored.
          {hasQueuedMessages && <span> {queuedMessages.length} message(s) queued.</span>}
        </div>
      )}
      {state === 'connected' && hasQueuedMessages && (
        <div className="banner info">
          <strong>Sending {queuedMessages.length} queued message(s)…</strong>
        </div>
      )}
      {health && !health.semanticMatching && (
        <div className="banner">
          <strong>Lexical matching.</strong> No embedding model is configured, so roles are matched
          on wording rather than meaning — a resume saying “React” will not surface for a posting
          asking for a “modern frontend framework”. Set <code>Ollama__EmbedModel</code> to enable
          semantic matching.
        </div>
      )}
      {error && <div className="banner error">{error}</div>}

      <div className="chat-layout">
        <div className="card chat">
          {!resume && (
            <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
              <input {...getInputProps()} />
              <strong>{uploading ? 'Reading your resume…' : 'Drop your resume here'}</strong>
              PDF, DOCX, TXT or MD — or click to browse
            </div>
          )}

          <div className="chat-scroll" ref={scrollRef}>
            {turns.length === 0 && resume && (
              <div className="empty">Ask me to find roles, or narrow by location, work mode or salary.</div>
            )}

            {turns.map(turn => (
              <div key={turn.id} className={`bubble ${turn.role}`}>
                {turn.thoughts.length > 0 && (
                  <div className="thoughts">
                    {turn.thoughts.map((thought, index) => (
                      <div
                        key={index}
                        className={`thought ${thought.stage === 'degraded' ? 'degraded' : ''}`}
                      >
                        <span className="stage">{thought.stage}</span>
                        <span>{thought.text}</span>
                      </div>
                    ))}
                  </div>
                )}

                {turn.role === 'assistant' ? (
                  <div className="markdown">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{turn.content}</ReactMarkdown>
                    {turn.streaming && <span className="cursor" />}
                  </div>
                ) : (
                  turn.content
                )}

                {turn.error && <div className="banner error" style={{ marginTop: 10 }}>{turn.error}</div>}
              </div>
            ))}
          </div>

          <form className="composer" onSubmit={submit}>
            <input
              value={draft}
              onChange={event => setDraft(event.target.value)}
              placeholder={state === 'disconnected'
                ? 'Connection lost. Start the API to send messages.'
                : resume ? 'Remote only? Senior roles? Ask why a score is what it is…'
                         : 'Upload your resume to get started'}
              disabled={busy || state === 'disconnected'}
            />
            {isSupported && (
              <button
                type="button"
                className={isListening ? 'primary listening' : ''}
                onClick={isListening ? stopListening : startListening}
                disabled={busy || state === 'disconnected'}
                aria-label={isListening ? 'Stop voice input' : 'Start voice input'}
                title={isListening ? 'Stop listening' : 'Voice input'}
              >
                {isListening ? '🎤' : '🎤'}
              </button>
            )}
            <button className="primary" type="submit" disabled={busy || !draft.trim() || state === 'disconnected'}>Send</button>
          </form>
        </div>

        <aside className="side">
          {resume && (
            <section className="card">
              <h4>Your profile</h4>
              <div className="kv"><span>Name</span><span>{resume.candidateName || '—'}</span></div>
              <div className="kv"><span>Title</span><span>{resume.currentTitle || '—'}</span></div>
              <div className="kv">
                <span>Experience</span>
                <span>{resume.yearsExperience ? `${resume.yearsExperience} yrs` : 'not stated'}</span>
              </div>
              <div className="chips" style={{ marginTop: 10 }}>
                {resume.skills.slice(0, 14).map(skill => <span key={skill} className="chip">{skill}</span>)}
              </div>
            </section>
          )}

          {!!Object.values(filters).filter(Boolean).length && (
            <section className="card">
              <h4>Active filters</h4>
              {Object.entries(filters)
                .filter(([, value]) => value !== null && value !== undefined &&
                                       !(Array.isArray(value) && value.length === 0))
                .map(([key, value]) => (
                  <div className="kv" key={key}>
                    <span>{key.replace(/([A-Z])/g, ' $1').toLowerCase()}</span>
                    <span>{Array.isArray(value) ? value.join(', ') : String(value)}</span>
                  </div>
                ))}
            </section>
          )}

          <section className="card">
            <h4>Board</h4>
            <div className="kv"><span>Jobs</span><span>{health?.jobCount ?? '—'}</span></div>
            <div className="kv"><span>Indexed</span><span>{health?.indexedJobCount ?? '—'}</span></div>
            <div className="kv"><span>Matching</span><span>{health?.semanticMatching ? 'semantic' : 'lexical'}</span></div>
            <div className="kv"><span>Reasoning</span><span>{health?.chatModel || 'deterministic'}</span></div>
          </section>
        </aside>
      </div>

      {latestMatches && latestMatches.matches.length > 0 && (
        <>
          <h1 style={{ marginTop: 30 }}>Your matches</h1>
          <p className="lede">
            {latestMatches.totalCandidates} role(s) matched; showing the closest {latestMatches.matches.length}.
          </p>
          <div className="job-grid">
            {latestMatches.matches.map(match => <MatchCard key={match.job.id} match={match} />)}
          </div>
        </>
      )}
    </div>
  );
}
