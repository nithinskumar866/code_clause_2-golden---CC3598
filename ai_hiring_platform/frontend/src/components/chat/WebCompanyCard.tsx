import { type FC } from 'react';
import { ExternalLink, Globe } from 'lucide-react';
import type { WebCompanyResult } from '../../types';
import { Card } from '../ui/Card';
import { Badge } from '../ui/Badge';

/**
 * One company behind an answer, with the crawled passages that put it there.
 *
 * Every passage links to the page it came from, and that link is the point: unlike a
 * stored spreadsheet field, the reader can open the source and check it themselves.
 * Nothing on this card was written by a language model — these are the exact strings
 * retrieved from the index.
 */
interface WebCompanyCardProps {
  company: WebCompanyResult;
  rank: number;
}

/** Page type → how it reads in a sentence, so a citation is not a column header. */
const PAGE_TYPE_LABEL: Record<string, string> = {
  home: 'homepage',
  about: 'about',
  products: 'products',
  services: 'services',
  clients: 'clients',
  careers: 'careers',
  leadership: 'leadership',
  news: 'news',
  contact: 'contact',
  other: 'page',
};

export const WebCompanyCard: FC<WebCompanyCardProps> = ({ company, rank }) => (
  <Card className="overflow-hidden p-0">
    <div className="flex items-start justify-between gap-3 border-b border-white/5 p-3">
      <div className="flex min-w-0 items-start gap-2.5">
        <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-sky-500/15 text-[11px] font-bold text-sky-300">
          {rank}
        </span>
        <h4 className="flex min-w-0 items-center gap-1.5 truncate text-sm font-semibold text-white">
          <Globe className="h-3.5 w-3.5 shrink-0 text-sky-400" />
          {company.company_name}
        </h4>
      </div>
      {/* Cosine similarity, shown as-is. Presenting it as a percentage would imply a
          precision it does not have. */}
      <Badge tone="info">{company.relevance.toFixed(2)}</Badge>
    </div>

    <div className="space-y-3 p-3">
      {company.matches.map((match, i) => (
        <div key={`${match.page_url}-${i}`}>
          <div className="flex flex-wrap items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wide text-sky-400/80">
            <span>{PAGE_TYPE_LABEL[match.page_type] ?? match.page_type}</span>
            {match.section && (
              <>
                <span className="text-gray-600">·</span>
                <span className="normal-case tracking-normal text-gray-500">{match.section}</span>
              </>
            )}
            <span className="text-gray-600">·</span>
            <span className="text-gray-500">{match.score.toFixed(3)}</span>
          </div>
          <p className="mt-1 text-xs leading-relaxed text-gray-300">{match.text}</p>
          <a
            href={match.page_url}
            target="_blank"
            rel="noreferrer noopener"
            className="mt-1 inline-flex max-w-full items-center gap-1 truncate text-[10px] text-gray-500 transition hover:text-sky-300"
          >
            <ExternalLink className="h-3 w-3 shrink-0" />
            <span className="truncate">{match.page_url}</span>
          </a>
        </div>
      ))}
    </div>
  </Card>
);
