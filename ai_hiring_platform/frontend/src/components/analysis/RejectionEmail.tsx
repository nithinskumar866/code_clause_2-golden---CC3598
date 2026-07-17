import { useState, type FC } from 'react';
import { Mail, Copy, Check } from 'lucide-react';

interface RejectionEmailProps {
  email: string;
}

/** Draft rejection email with copy-to-clipboard. Self-contained copy state. */
export const RejectionEmail: FC<RejectionEmailProps> = ({ email }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(email);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="rounded-xl border border-white/5 bg-card p-5 space-y-4">
      <div className="flex justify-between items-center border-b border-white/5 pb-2">
        <div className="flex items-center gap-2 text-sm font-semibold text-rose-400">
          <Mail className="h-4 w-4" /> Recruiter Email Draft (Dynamic Rejection)
        </div>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 text-xs border border-white/10 rounded-lg px-2.5 py-1 text-gray-400 hover:text-white hover:bg-white/5 transition"
        >
          {copied ? (
            <>
              <Check className="h-3 w-3 text-emerald-400" /> Copied!
            </>
          ) : (
            <>
              <Copy className="h-3 w-3" /> Copy Draft
            </>
          )}
        </button>
      </div>

      <pre className="rounded-lg bg-black/40 border border-white/5 p-4 text-xs text-gray-400 overflow-x-auto whitespace-pre-wrap font-mono leading-relaxed max-h-[300px]">
        {email}
      </pre>
    </div>
  );
};
