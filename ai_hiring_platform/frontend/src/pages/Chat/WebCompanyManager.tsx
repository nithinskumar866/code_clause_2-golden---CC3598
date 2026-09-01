import { useCallback, useEffect, useState, type FC, type FormEvent } from 'react';
import {
  ChevronDown, Globe, Loader2, Plus, RefreshCw, Trash2, UserRound,
} from 'lucide-react';
import type { WebCompany, WebCompanyStatus } from '../../types';
import {
  deleteWebCompany,
  listWebCompanies,
  registerWebCompany,
  getWebCompanyStatus,
  startWebCrawl,
} from '../../api/webChat';
import { Button } from '../../components/ui/Button';
import { Card } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { Spinner } from '../../components/ui/Spinner';

/**
 * Add companies and crawl them, without leaving the app.
 *
 * The assistant can only answer from pages that have been crawled, so a chat screen
 * with no way to add a company is a screen that answers nothing until someone opens a
 * terminal. This panel exists so the loop — register, crawl, ask — closes here.
 *
 * Crawls run in the background on the service, which is why this polls rather than
 * waiting: a twelve-page crawl takes minutes, and an HTTP request held open that long
 * is one browser timeout away from a half-indexed company.
 */
interface WebCompanyManagerProps {
  /** Ask the parent to refresh its health badge after anything changes the index. */
  onChanged?: () => void;
}

const relativeTime = (seconds: number | null): string => {
  if (!seconds) return 'never';
  const delta = Date.now() / 1000 - seconds;
  if (delta < 90) return 'just now';
  if (delta < 3600) return `${Math.round(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.round(delta / 3600)}h ago`;
  return `${Math.round(delta / 86400)}d ago`;
};

export const WebCompanyManager: FC<WebCompanyManagerProps> = ({ onChanged }) => {
  const [companies, setCompanies] = useState<WebCompany[]>([]);
  const [statuses, setStatuses] = useState<Record<string, WebCompanyStatus>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [crawling, setCrawling] = useState<Set<string>>(new Set());
  const [open, setOpen] = useState(false);

  const [name, setName] = useState('');
  const [domain, setDomain] = useState('');
  const [linkedin, setLinkedin] = useState('');
  const [adding, setAdding] = useState(false);

  const loadStatuses = useCallback(async (list: WebCompany[]) => {
    const entries = await Promise.all(
      list.map(async (c) => {
        try {
          return [c.company_id, await getWebCompanyStatus(c.company_id)] as const;
        } catch {
          return null;
        }
      }),
    );
    setStatuses(Object.fromEntries(entries.filter(Boolean) as [string, WebCompanyStatus][]));
  }, []);

  const refresh = useCallback(async () => {
    try {
      const list = await listWebCompanies();
      setCompanies(list);
      setError(null);
      await loadStatuses(list);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not reach the crawler.');
    } finally {
      setLoading(false);
    }
  }, [loadStatuses]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // While a crawl is running, poll its page count so the panel shows progress rather
  // than a spinner that could mean anything. Stops as soon as nothing is crawling.
  useEffect(() => {
    if (crawling.size === 0) return;
    const timer = window.setInterval(() => {
      void (async () => {
        const done: string[] = [];
        for (const id of crawling) {
          try {
            const status = await getWebCompanyStatus(id);
            setStatuses((s) => ({ ...s, [id]: status }));
            // A crawl is finished when nothing is left pending.
            if ((status.pages_by_status.pending ?? 0) === 0 && status.pages_known > 0) {
              done.push(id);
            }
          } catch {
            done.push(id); // unreachable — stop polling rather than spinning forever
          }
        }
        if (done.length) {
          setCrawling((prev) => {
            const next = new Set(prev);
            done.forEach((id) => next.delete(id));
            return next;
          });
          onChanged?.();
        }
      })();
    }, 4000);
    return () => window.clearInterval(timer);
  }, [crawling, onChanged]);

  const onAdd = async (e: FormEvent) => {
    e.preventDefault();
    if (!domain.trim() || adding) return;
    setAdding(true);
    setError(null);
    try {
      const created = await registerWebCompany({
        name: name.trim() || domain.trim(),
        domain: domain.trim(),
        linkedin_urls: linkedin
          .split(',')
          .map((u) => u.trim())
          .filter(Boolean),
      });
      setName('');
      setDomain('');
      setLinkedin('');
      await refresh();
      // Crawl straight away: a company registered but never crawled is invisible to
      // the assistant, and having to press a second button to make it real is a step
      // people forget.
      await startWebCrawl(created.company_id, 20);
      setCrawling((prev) => new Set(prev).add(created.company_id));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not add that company.');
    } finally {
      setAdding(false);
    }
  };

  const onCrawl = async (companyId: string) => {
    setBusyId(companyId);
    setError(null);
    try {
      await startWebCrawl(companyId, 20);
      setCrawling((prev) => new Set(prev).add(companyId));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not start that crawl.');
    } finally {
      setBusyId(null);
    }
  };

  const onDelete = async (company: WebCompany) => {
    const status = statuses[company.company_id];
    const detail = status ? ` (${status.pages_known} pages, ${status.chunks} passages)` : '';
    if (!window.confirm(`Remove ${company.name} and everything crawled for it${detail}?`)) return;
    setBusyId(company.company_id);
    try {
      await deleteWebCompany(company.company_id);
      await refresh();
      onChanged?.();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not remove that company.');
    } finally {
      setBusyId(null);
    }
  };

  return (
    <Card className="p-0">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between gap-3 p-3 text-left"
        aria-expanded={open}
      >
        <span className="inline-flex items-center gap-2 text-xs font-semibold text-white">
          <Globe className="h-3.5 w-3.5 text-sky-400" />
          Crawled companies
          <span className="font-normal text-gray-500">
            {loading ? '…' : `${companies.length} registered`}
          </span>
        </span>
        <ChevronDown
          className={`h-4 w-4 shrink-0 text-gray-500 transition ${open ? 'rotate-180' : ''}`}
        />
      </button>

      {open && (
        <div className="space-y-3 border-t border-white/5 p-3">
          <form onSubmit={onAdd} className="grid gap-2 sm:grid-cols-[1fr_1fr_1fr_auto]">
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Company name"
              aria-label="Company name"
              className="rounded-lg border border-white/10 bg-card px-2.5 py-1.5 text-xs text-white placeholder:text-gray-600 focus:border-sky-500/50 focus:outline-none"
            />
            <input
              value={domain}
              onChange={(e) => setDomain(e.target.value)}
              placeholder="acme.com"
              aria-label="Company domain"
              required
              className="rounded-lg border border-white/10 bg-card px-2.5 py-1.5 text-xs text-white placeholder:text-gray-600 focus:border-sky-500/50 focus:outline-none"
            />
            <input
              value={linkedin}
              onChange={(e) => setLinkedin(e.target.value)}
              placeholder="LinkedIn URLs (optional, comma separated)"
              aria-label="LinkedIn profile URLs of officials"
              title="Stored and shown as links. This service never fetches LinkedIn — see docs/linkedin.md."
              className="rounded-lg border border-white/10 bg-card px-2.5 py-1.5 text-xs text-white placeholder:text-gray-600 focus:border-sky-500/50 focus:outline-none"
            />
            <Button
              type="submit"
              size="sm"
              loading={adding}
              disabled={!domain.trim()}
              leftIcon={<Plus className="h-3.5 w-3.5" />}
            >
              Add &amp; crawl
            </Button>
          </form>

          <p className="text-[10px] leading-relaxed text-gray-600">
            Adding a company crawls up to 20 pages of its website — About, Products and
            Leadership before the blog. LinkedIn URLs are stored as links only and are never
            fetched.
          </p>

          {loading && (
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <Spinner /> Reading the crawler…
            </div>
          )}

          {!loading && companies.length === 0 && !error && (
            <p className="text-xs text-gray-500">
              No companies yet. Add a domain above and the assistant will have something to
              answer from.
            </p>
          )}

          <div className="space-y-1.5">
            {companies.map((company) => {
              const status = statuses[company.company_id];
              const isCrawling = crawling.has(company.company_id);
              return (
                <div
                  key={company.company_id}
                  className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-white/5 bg-white/[0.02] px-2.5 py-2"
                >
                  <div className="min-w-0">
                    <div className="flex items-center gap-1.5 text-xs font-medium text-white">
                      <span className="truncate">{company.name}</span>
                      <span className="truncate text-[10px] font-normal text-gray-500">
                        {company.domain}
                      </span>
                      {company.client_rendered && (
                        <span
                          title={
                            'Most pages on this site returned identical content — it builds its ' +
                            'pages in the browser, so a plain crawl reads one shell repeated. ' +
                            'Only the distinct pages were indexed.'
                          }
                        >
                          <Badge tone="warning">client-rendered</Badge>
                        </span>
                      )}
                      {company.linkedin_urls.length > 0 && (
                        <a
                          href={company.linkedin_urls[0]}
                          target="_blank"
                          rel="noreferrer noopener"
                          title="Open this official's LinkedIn profile — never crawled"
                          className="text-gray-600 transition hover:text-sky-300"
                        >
                          <UserRound className="h-3 w-3" />
                        </a>
                      )}
                    </div>
                    <div className="mt-0.5 flex flex-wrap items-center gap-1.5 text-[10px] text-gray-500">
                      {isCrawling ? (
                        <span className="inline-flex items-center gap-1 text-sky-300">
                          <Loader2 className="h-3 w-3 animate-spin" />
                          crawling… {status?.pages_known ?? 0} pages so far
                        </span>
                      ) : status ? (
                        <>
                          <span>
                            {(status.pages_by_status.live ?? 0)} indexed
                            {(status.pages_by_status.duplicate ?? 0) > 0 && (
                              <span
                                className="text-amber-500/80"
                                title="Pages holding content already indexed under another URL. Skipped, because duplicates crowd real answers out of the results."
                              >
                                {' '}+{status.pages_by_status.duplicate} duplicate
                              </span>
                            )}
                          </span>
                          <span className="text-gray-700">·</span>
                          <span>{status.chunks} passages</span>
                          <span className="text-gray-700">·</span>
                          <span>crawled {relativeTime(status.last_crawled_at)}</span>
                          {status.failures.length > 0 && (
                            <Badge tone="warning">{status.failures.length} failed</Badge>
                          )}
                        </>
                      ) : (
                        <span>not crawled yet</span>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      title="Re-crawl. Unchanged pages cost one request and are skipped."
                      disabled={busyId === company.company_id || isCrawling}
                      onClick={() => void onCrawl(company.company_id)}
                    >
                      <RefreshCw className="h-3.5 w-3.5" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      title="Remove this company and everything crawled for it"
                      disabled={busyId === company.company_id}
                      onClick={() => void onDelete(company)}
                    >
                      <Trash2 className="h-3.5 w-3.5 text-rose-400" />
                    </Button>
                  </div>
                </div>
              );
            })}
          </div>

          {error && (
            <div className="rounded-lg border border-rose-500/20 bg-rose-500/10 px-3 py-2 text-[11px] text-rose-300">
              {error}
            </div>
          )}
        </div>
      )}
    </Card>
  );
};
