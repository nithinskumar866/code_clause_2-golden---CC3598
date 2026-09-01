import type { FC } from 'react';
import { Building2, Globe, Users } from 'lucide-react';

/**
 * Which knowledge base the assistant answers from.
 *
 * A hard switch, not a blend. The modes search different corpora with different
 * notions of relevance, so letting one question hit several would put unrelated
 * populations behind a single ranking. Switching mode also switches conversation
 * memory, which is why a follow-up never crosses from one to the other.
 *
 * `web` is served by the standalone `company_intel` crawler, not by this backend. It
 * sits beside `companies` rather than replacing it because the two answer from
 * genuinely different things: `companies` reads the curated spreadsheet records, `web`
 * reads what each company currently publishes on its own site. Citing a stored field
 * and citing a live URL are different claims, and merging them would blur that.
 */
export type ChatMode = 'candidates' | 'companies' | 'web';

interface ChatModeToggleProps {
  mode: ChatMode;
  onChange: (mode: ChatMode) => void;
  /** Disabled with a reason when the company store is absent or unreachable. */
  companiesDisabled?: boolean;
  companiesTitle?: string;
  /** Disabled with a reason when the crawler service is not running. */
  webDisabled?: boolean;
  webTitle?: string;
}

const BASE =
  'inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[11px] font-semibold transition disabled:cursor-not-allowed disabled:opacity-40';

export const ChatModeToggle: FC<ChatModeToggleProps> = ({
  mode,
  onChange,
  companiesDisabled = false,
  companiesTitle,
  webDisabled = false,
  webTitle,
}) => (
  <div
    className="flex items-center rounded-lg border border-white/10 p-0.5"
    role="group"
    aria-label="Knowledge base"
  >
    <button
      type="button"
      aria-pressed={mode === 'candidates'}
      onClick={() => onChange('candidates')}
      title="Search your indexed resumes"
      className={`${BASE} ${
        mode === 'candidates' ? 'bg-indigo-600 text-white' : 'text-gray-400 hover:text-white'
      }`}
    >
      <Users className="h-3.5 w-3.5" />
      Candidates
    </button>
    <button
      type="button"
      aria-pressed={mode === 'companies'}
      disabled={companiesDisabled}
      onClick={() => onChange('companies')}
      title={companiesTitle ?? 'Search your company database'}
      className={`${BASE} ${
        mode === 'companies' ? 'bg-emerald-600 text-white' : 'text-gray-400 hover:text-white'
      }`}
    >
      <Building2 className="h-3.5 w-3.5" />
      Companies
    </button>
    <button
      type="button"
      aria-pressed={mode === 'web'}
      disabled={webDisabled}
      onClick={() => onChange('web')}
      title={webTitle ?? 'Search company websites crawled by the crawler service'}
      className={`${BASE} ${
        mode === 'web' ? 'bg-sky-600 text-white' : 'text-gray-400 hover:text-white'
      }`}
    >
      <Globe className="h-3.5 w-3.5" />
      Web
    </button>
  </div>
);
