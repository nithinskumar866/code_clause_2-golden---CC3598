import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Speaking to the assistant, and being answered out loud.
 *
 * Built on the browser's own Web Speech API rather than a server-side model, for
 * three reasons: it needs no upload, no extra model pulled onto a machine already
 * spending ~50s on each LLM call, and — for dictation — the audio never touches
 * this application at all.
 *
 * Support is genuinely uneven, so both halves are reported separately and the UI
 * is expected to hide what is unavailable rather than offer a button that does
 * nothing. Recognition is Chrome and Edge; synthesis is effectively everywhere.
 */

/* The Web Speech API is not in the DOM lib's stable types. */
interface SpeechRecognitionAlternative { transcript: string; confidence: number }
interface SpeechRecognitionResult {
  readonly length: number;
  item(index: number): SpeechRecognitionAlternative;
  [index: number]: SpeechRecognitionAlternative;
  isFinal: boolean;
}
interface SpeechRecognitionResultList {
  readonly length: number;
  item(index: number): SpeechRecognitionResult;
  [index: number]: SpeechRecognitionResult;
}
interface SpeechRecognitionEventLike extends Event {
  resultIndex: number;
  results: SpeechRecognitionResultList;
}
interface SpeechRecognitionErrorEventLike extends Event { error: string }
interface SpeechRecognitionLike extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null;
  onend: (() => void) | null;
}
type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

function recognitionCtor(): SpeechRecognitionCtor | null {
  if (typeof window === 'undefined') return null;
  const w = window as unknown as {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

const SPEAK_ENABLED_KEY = 'portal.voiceReplies';

export interface UseVoice {
  /** Dictation is available in this browser. */
  canListen: boolean;
  /** Reading replies aloud is available in this browser. */
  canSpeak: boolean;

  listening: boolean;
  /** Words recognised so far in the current utterance, final and interim. */
  transcript: string;
  /** Set when the mic could not be used — permission denied, no device. */
  voiceError: string | null;
  startListening: () => void;
  stopListening: () => void;

  /** Whether replies should be read aloud. Remembered between visits. */
  speakReplies: boolean;
  toggleSpeakReplies: () => void;
  speaking: boolean;
  speak: (text: string) => void;
  stopSpeaking: () => void;
}

export function useVoice(): UseVoice {
  const [canListen] = useState(() => recognitionCtor() !== null);
  const [canSpeak] = useState(
    () => typeof window !== 'undefined' && 'speechSynthesis' in window,
  );

  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const [speaking, setSpeaking] = useState(false);

  const [speakReplies, setSpeakReplies] = useState(() => {
    try {
      return localStorage.getItem(SPEAK_ENABLED_KEY) === 'true';
    } catch {
      return false;
    }
  });

  const recognition = useRef<SpeechRecognitionLike | null>(null);

  // -- listening ----------------------------------------------------------

  const stopListening = useCallback(() => {
    recognition.current?.stop();
    setListening(false);
  }, []);

  const startListening = useCallback(() => {
    const Ctor = recognitionCtor();
    if (!Ctor) return;

    setVoiceError(null);
    setTranscript('');

    const engine = new Ctor();
    engine.lang = navigator.language || 'en-US';
    // Interim results so the words appear as they are spoken. Without them the
    // composer sits empty for the whole sentence and reads as a dead button.
    engine.interimResults = true;
    engine.continuous = false;

    engine.onresult = (event) => {
      let text = '';
      for (let i = 0; i < event.results.length; i += 1) {
        text += event.results[i][0].transcript;
      }
      setTranscript(text);
    };

    engine.onerror = (event) => {
      setListening(false);
      // "aborted" and "no-speech" are ordinary — the user changed their mind or
      // said nothing. Only real failures are worth putting on screen.
      if (event.error === 'aborted' || event.error === 'no-speech') return;
      setVoiceError(
        event.error === 'not-allowed'
          ? 'Microphone access was blocked. Allow it in your browser settings to dictate.'
          : 'The microphone could not be used.',
      );
    };

    engine.onend = () => setListening(false);

    recognition.current = engine;
    try {
      engine.start();
      setListening(true);
    } catch {
      // start() throws if called while already running.
      setListening(false);
    }
  }, []);

  // -- speaking -----------------------------------------------------------

  const stopSpeaking = useCallback(() => {
    if (!canSpeak) return;
    window.speechSynthesis.cancel();
    setSpeaking(false);
  }, [canSpeak]);

  const speak = useCallback((text: string) => {
    if (!canSpeak) return;

    // Markdown is written to be read, not heard: asterisks, pipes and hyphens
    // become noise out loud, and a comparison table read cell by cell is
    // unlistenable. Strip to the prose and let the cards carry the detail.
    const spoken = text
      .replace(/```[\s\S]*?```/g, ' ')
      .replace(/^\s*\|.*\|\s*$/gm, ' ')
      .replace(/[*_`#>]/g, '')
      .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
      .replace(/\s+/g, ' ')
      .trim();

    if (!spoken) return;

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(spoken);
    utterance.lang = navigator.language || 'en-US';
    utterance.rate = 1.02;
    utterance.onend = () => setSpeaking(false);
    utterance.onerror = () => setSpeaking(false);

    setSpeaking(true);
    window.speechSynthesis.speak(utterance);
  }, [canSpeak]);

  const toggleSpeakReplies = useCallback(() => {
    setSpeakReplies(previous => {
      const next = !previous;
      try {
        localStorage.setItem(SPEAK_ENABLED_KEY, String(next));
      } catch { /* a blocked store is not worth failing a toggle over */ }
      // Turning it off should silence what is already mid-sentence.
      if (!next && typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        setSpeaking(false);
      }
      return next;
    });
  }, []);

  // Nothing should keep talking after the page is gone.
  useEffect(() => () => {
    recognition.current?.abort();
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
  }, []);

  return {
    canListen, canSpeak,
    listening, transcript, voiceError, startListening, stopListening,
    speakReplies, toggleSpeakReplies, speaking, speak, stopSpeaking,
  };
}
