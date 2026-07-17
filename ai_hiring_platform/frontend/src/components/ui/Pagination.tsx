import type { FC } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

interface PaginationProps {
  page: number;
  totalPages: number;
  onPage: (page: number) => void;
}

type Token = number | 'ellipsis';

/** Windowed page tokens: 1 … (p-1) p (p+1) … last. */
function pageTokens(page: number, total: number): Token[] {
  if (total <= 7) {
    return Array.from({ length: total }, (_, i) => i + 1);
  }
  const windowPages: number[] = [];
  for (let i = Math.max(1, page - 1); i <= Math.min(total, page + 1); i += 1) {
    windowPages.push(i);
  }
  const tokens: Token[] = [];
  if (windowPages[0] > 1) {
    tokens.push(1);
    if (windowPages[0] > 2) tokens.push('ellipsis');
  }
  tokens.push(...windowPages);
  const last = windowPages[windowPages.length - 1];
  if (last < total) {
    if (last < total - 1) tokens.push('ellipsis');
    tokens.push(total);
  }
  return tokens;
}

const btnBase =
  'inline-flex h-8 min-w-8 items-center justify-center rounded-lg px-2 text-sm font-medium transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:cursor-not-allowed disabled:opacity-40';

/** Accessible pagination controls. Renders nothing for a single page. */
export const Pagination: FC<PaginationProps> = ({ page, totalPages, onPage }) => {
  if (totalPages <= 1) return null;
  const tokens = pageTokens(page, totalPages);

  return (
    <nav aria-label="Pagination" className="flex items-center justify-center gap-1.5">
      <button
        type="button"
        className={`${btnBase} border border-white/10 text-gray-300 hover:bg-white/5`}
        onClick={() => onPage(page - 1)}
        disabled={page <= 1}
        aria-label="Previous page"
      >
        <ChevronLeft className="h-4 w-4" />
      </button>

      {tokens.map((token, i) =>
        token === 'ellipsis' ? (
          <span key={`e${i}`} className="px-1 text-sm text-gray-600" aria-hidden="true">
            …
          </span>
        ) : (
          <button
            key={token}
            type="button"
            className={`${btnBase} ${
              token === page
                ? 'bg-indigo-600 text-white'
                : 'border border-white/10 text-gray-300 hover:bg-white/5'
            }`}
            onClick={() => onPage(token)}
            aria-label={`Page ${token}`}
            aria-current={token === page ? 'page' : undefined}
          >
            {token}
          </button>
        ),
      )}

      <button
        type="button"
        className={`${btnBase} border border-white/10 text-gray-300 hover:bg-white/5`}
        onClick={() => onPage(page + 1)}
        disabled={page >= totalPages}
        aria-label="Next page"
      >
        <ChevronRight className="h-4 w-4" />
      </button>
    </nav>
  );
};
