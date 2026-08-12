import { useEffect, useState, type FC } from 'react';
import { X, Download, FileText, Loader2 } from 'lucide-react';
import type { ResumeContent } from '../../types';
import { getResumeContent, resumeFileUrl } from '../../api/documents';

interface ResumeViewerProps {
  resumeId: number;
  name: string;
  /** Terms to highlight — the skills that made this candidate match. */
  highlight?: string[];
  onClose: () => void;
}

/**
 * The candidate's resume, as the platform actually holds it.
 *
 * Two things a recruiter needs, and they are not the same document:
 *
 *   the PARSED text  — what was indexed, with the matched terms highlighted, so "why
 *                      did this person come up?" is answerable by looking
 *   the ORIGINAL file — what the candidate actually sent, formatting intact, to read
 *                      properly or forward to a hiring manager
 *
 * The parsed view is served from the document layer rather than re-parsed on demand:
 * showing text that differs from what was scored would make the evidence trail a lie.
 */
export const ResumeViewer: FC<ResumeViewerProps> = ({ resumeId, name, highlight = [], onClose }) => {
  const [content, setContent] = useState<ResumeContent | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    getResumeContent(resumeId)
      .then((c) => alive && setContent(c))
      .catch((e) => alive && setError(e instanceof Error ? e.message : 'Could not load the resume.'));
    return () => { alive = false; };
  }, [resumeId]);

  // Escape closes, as in any modal.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
      role="dialog"
      aria-modal="true"
      aria-label={`Resume for ${name}`}
      onClick={onClose}
    >
      <div
        className="flex max-h-[88vh] w-full max-w-3xl flex-col rounded-xl border border-white/10 bg-card shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-center gap-3 border-b border-white/10 p-4">
          <FileText className="h-4 w-4 shrink-0 text-indigo-400" />
          <div className="min-w-0 flex-1">
            <h3 className="truncate text-sm font-semibold text-white">{name}</h3>
            {content?.filename && (
              <p className="truncate text-[11px] text-gray-500">{content.filename}</p>
            )}
          </div>
          <a
            href={resumeFileUrl(resumeId)}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 px-2.5 py-1.5 text-xs text-gray-300 transition hover:border-indigo-500/40 hover:text-white"
          >
            <Download className="h-3.5 w-3.5" /> Original file
          </a>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded-lg p-1.5 text-gray-400 transition hover:bg-white/5 hover:text-white"
          >
            <X className="h-4 w-4" />
          </button>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto p-4">
          {error && <p className="text-xs text-rose-300">{error}</p>}
          {!content && !error && (
            <p className="flex items-center gap-2 text-xs text-gray-400">
              <Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading the resume…
            </p>
          )}
          {content && !content.in_working_set && (
            <p className="text-xs text-amber-300">
              This resume has not been chunked yet, so there is no parsed text to show.
              Add it to the index on the Documents screen, or download the original above.
            </p>
          )}
          {content?.sections.map((s, i) => (
            <section key={`${s.section}-${i}`} className="mb-4">
              <h4 className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-indigo-400">
                {s.section}
                <span className="ml-2 font-normal text-gray-600">page {s.page}</span>
              </h4>
              <p className="whitespace-pre-wrap text-xs leading-relaxed text-gray-300">
                {highlightTerms(s.text, highlight)}
              </p>
            </section>
          ))}
        </div>
      </div>
    </div>
  );
};

/**
 * Mark the matched terms inside the resume text.
 *
 * Done on the rendered string rather than with `dangerouslySetInnerHTML`: resume text
 * is untrusted input, and injecting it as HTML to get a highlight would be trading a
 * real vulnerability for a visual nicety.
 */
function highlightTerms(text: string, terms: string[]) {
  const wanted = terms.map((t) => t.trim()).filter((t) => t.length > 1);
  if (!wanted.length) return text;

  const escaped = wanted.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  const pattern = new RegExp(`(${escaped.join('|')})`, 'gi');
  const parts = text.split(pattern);

  return parts.map((part, i) =>
    pattern.test(part) && wanted.some((w) => w.toLowerCase() === part.toLowerCase()) ? (
      <mark key={i} className="rounded bg-indigo-500/30 px-0.5 text-indigo-100">
        {part}
      </mark>
    ) : (
      part
    ),
  );
}
