import { useCallback, useEffect, useMemo, useState, type FC } from 'react';
import {
  ClipboardCheck, Play, Save, RefreshCw, GitCompare, History as HistoryIcon,
  ListChecks, TerminalSquare, Trash2, AlertTriangle, ArrowRight, Gauge,
} from 'lucide-react';
import type {
  PromptCase, PromptRunResult, PromptRunSummary, PromptRuleInfo, PromptScoreResult,
  PromptSuite, VariantResult,
} from '../../types';
import {
  createSuite, deleteSuite, getRules, getRun, getStarterSuite, listRuns, listSuites,
  runPrompts, scoreAnswer, updateSuite,
} from '../../api/promptLab';
import { PageHeader } from '../../components/ui/PageHeader';
import { Card } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Badge } from '../../components/ui/Badge';
import { Spinner } from '../../components/ui/Spinner';
import { AddCaseButton, CaseEditor } from './CaseEditor';
import { CaseResults, RuleMatrix } from './RuleMatrix';
import { blankCase, pct, scoreTone } from './promptLabHelpers';

type Tab = 'run' | 'cases' | 'playground' | 'history';

const TABS: { id: Tab; label: string; icon: FC<{ className?: string }> }[] = [
  { id: 'run', label: 'Compliance & A/B', icon: ClipboardCheck },
  { id: 'cases', label: 'Test cases', icon: ListChecks },
  { id: 'playground', label: 'Playground', icon: TerminalSquare },
  { id: 'history', label: 'Regression history', icon: HistoryIcon },
];

const textareaClass =
  'w-full rounded-lg border border-white/10 bg-black/40 px-3 py-2 font-mono text-xs leading-relaxed text-white placeholder:text-gray-600 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500';

const labelClass = 'mb-1.5 block text-[10px] font-semibold uppercase tracking-wider text-gray-400';

/**
 * Prompt Lab — a bench for the system prompt itself.
 *
 * WHY THIS IS A SEPARATE SECTION
 * "The prompt is not working properly" is not a bug report anyone can act on. Which of
 * its clauses does it break, on which question, and does the rewrite that fixes that one
 * quietly break two others? Everywhere else in the platform an answer is judged against
 * retrieved evidence; here the PROMPT is the thing under test, and the evidence is a
 * fixed set of turns it is run over.
 *
 * WHAT MAKES THE VERDICT TRUSTWORTHY
 * The LLM only produces the answer. Every judgement about that answer is deterministic
 * (backend `prompt_rules_service`) — line count, markdown links, whether a stated year
 * figure exists in the context at all. An LLM judge would score the same answer
 * differently on a rerun, which is precisely what a regression suite cannot tolerate.
 *
 * FOUR VIEWS, ONE STATE
 * The prompt and the cases are shared; the tabs only change what you do with them —
 * measure one prompt, diff two, poke at a single turn by hand, or look at how the score
 * moved over past runs.
 */
export const PromptLab: FC = () => {
  const [tab, setTab] = useState<Tab>('run');

  const [promptA, setPromptA] = useState('');
  const [promptB, setPromptB] = useState('');
  const [compare, setCompare] = useState(false);
  const [cases, setCases] = useState<PromptCase[]>([]);
  const [temperature, setTemperature] = useState(0.2);

  const [suites, setSuites] = useState<PromptSuite[]>([]);
  const [suiteId, setSuiteId] = useState<number | null>(null);
  const [suiteName, setSuiteName] = useState('');

  const [result, setResult] = useState<PromptRunResult | null>(null);
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const [catalog, setCatalog] = useState<PromptRuleInfo[]>([]);
  const [runs, setRuns] = useState<PromptRunSummary[]>([]);
  const [openRun, setOpenRun] = useState<VariantResult | null>(null);

  // Playground
  const [playCase, setPlayCase] = useState(0);
  const [playAnswer, setPlayAnswer] = useState('');
  const [playScore, setPlayScore] = useState<PromptScoreResult | null>(null);
  const [playBusy, setPlayBusy] = useState(false);

  // The prompt in the field, and the turns that exercise it, come from the backend so
  // that what is graded here is the same artefact the assistant actually runs.
  useEffect(() => {
    (async () => {
      try {
        const [starter, savedSuites, rules] = await Promise.all([
          getStarterSuite(), listSuites(), getRules(),
        ]);
        setPromptA(starter.prompt);
        setCases(starter.cases);
        setSuiteName(starter.name);
        setSuites(savedSuites);
        setCatalog(rules);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Could not load the Prompt Lab.');
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const loadRuns = useCallback(async () => {
    try {
      setRuns(await listRuns(suiteId));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not load the run history.');
    }
  }, [suiteId]);

  useEffect(() => {
    if (tab === 'history') void loadRuns();
  }, [tab, loadRuns]);

  const usableCases = useMemo(() => cases.filter((c) => c.question.trim()), [cases]);

  const onRun = async () => {
    if (!usableCases.length || running) return;
    setRunning(true);
    setError(null);
    setNotice(null);
    setResult(null);
    try {
      const variants = compare
        ? [{ label: 'Current', prompt: promptA }, { label: 'Rewrite', prompt: promptB }]
        : [{ label: 'Current', prompt: promptA }];
      const data = await runPrompts(variants, usableCases, { temperature, suiteId });
      setResult(data);
      const generated = data.variants.some((v) => v.cases.some((c) => c.answer.trim()));
      if (!data.llm_available) {
        setNotice(
          'No LLM is configured, so no answers were generated. Set a provider or runtime in the backend .env, ' +
          'or paste an answer into the Playground — grading itself needs no LLM.',
        );
      } else if (!generated) {
        // Scores over empty answers would read as a catastrophic prompt failure when the
        // real fault is the endpoint. Say so, and say that nothing was recorded.
        setNotice(
          `The configured model (${data.model}) returned nothing — open a case below for the error. ` +
          'Nothing was written to the regression history.',
        );
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'The prompt run failed.');
    } finally {
      setRunning(false);
    }
  };

  const onSaveSuite = async () => {
    try {
      if (suiteId) {
        const saved = await updateSuite(suiteId, { name: suiteName, prompt: promptA, cases: usableCases });
        setSuites((s) => s.map((x) => (x.id === saved.id ? saved : x)));
        setNotice(`Suite "${saved.name}" updated.`);
      } else {
        const saved = await createSuite(suiteName || 'Untitled suite', promptA, usableCases);
        setSuites((s) => [saved, ...s]);
        setSuiteId(saved.id);
        setNotice(`Suite "${saved.name}" saved — runs against it are now tracked.`);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not save the suite.');
    }
  };

  const onLoadSuite = (id: number) => {
    const suite = suites.find((s) => s.id === id);
    if (!suite) return;
    setSuiteId(suite.id);
    setSuiteName(suite.name);
    setPromptA(suite.prompt);
    setCases(suite.cases);
    setResult(null);
    setNotice(`Loaded "${suite.name}".`);
  };

  const onDeleteSuite = async (id: number) => {
    try {
      await deleteSuite(id);
      setSuites((s) => s.filter((x) => x.id !== id));
      if (suiteId === id) setSuiteId(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not delete the suite.');
    }
  };

  const onGradeAnswer = async () => {
    const target = cases[playCase];
    if (!target || !playAnswer.trim()) return;
    setPlayBusy(true);
    setError(null);
    try {
      setPlayScore(await scoreAnswer(promptA, target, playAnswer));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not grade the answer.');
    } finally {
      setPlayBusy(false);
    }
  };

  /** Generate one answer for the selected case, then grade it — the manual loop. */
  const onGenerateAnswer = async () => {
    const target = cases[playCase];
    if (!target) return;
    setPlayBusy(true);
    setError(null);
    try {
      const data = await runPrompts([{ label: 'Current', prompt: promptA }], [target], {
        temperature, persist: false, suiteId: null,
      });
      const first = data.variants[0]?.cases[0];
      if (!data.llm_available || !first) {
        setNotice('No LLM is configured — paste an answer instead and grade it.');
        return;
      }
      setPlayAnswer(first.answer);
      setPlayScore({
        rules: first.rules, passed: first.passed, failed: first.failed,
        not_applicable: first.not_applicable, score: first.score, violations: first.violations,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Generation failed.');
    } finally {
      setPlayBusy(false);
    }
  };

  const onOpenRun = async (id: number) => {
    try {
      const detail = await getRun(id);
      setOpenRun(detail.results);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not open that run.');
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner className="h-6 w-6" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Prompt Lab"
        icon={<Gauge className="h-5 w-5" />}
        description="Run a system prompt over a fixed set of turns and measure, clause by clause, how often it obeys itself. The model writes the answers; the scoring is deterministic."
        actions={
          <>
            <Button
              variant="secondary"
              size="sm"
              leftIcon={<Save className="h-4 w-4" />}
              onClick={onSaveSuite}
            >
              {suiteId ? 'Update suite' : 'Save as suite'}
            </Button>
            <Button
              size="sm"
              loading={running}
              leftIcon={<Play className="h-4 w-4" />}
              onClick={onRun}
              disabled={!usableCases.length}
            >
              Run {usableCases.length} case{usableCases.length === 1 ? '' : 's'}
            </Button>
          </>
        }
      />

      {error && (
        <div className="flex items-start gap-2 rounded-lg border border-rose-500/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          {error}
        </div>
      )}
      {notice && (
        <div className="rounded-lg border border-amber-500/20 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
          {notice}
        </div>
      )}

      <div className="flex flex-wrap gap-1 border-b border-white/5">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            type="button"
            onClick={() => setTab(id)}
            aria-current={tab === id ? 'page' : undefined}
            className={`flex items-center gap-2 border-b-2 px-3 py-2 text-sm font-medium transition ${
              tab === id
                ? 'border-indigo-500 text-white'
                : 'border-transparent text-gray-400 hover:text-gray-200'
            }`}
          >
            <Icon className="h-4 w-4" />
            {label}
          </button>
        ))}
      </div>

      {/* ------------------------------------------------------------------ run */}
      {tab === 'run' && (
        <div className="space-y-5">
          <div className="grid gap-4 lg:grid-cols-2">
            <Card className="p-4">
              <div className="mb-2 flex items-center justify-between gap-2">
                <label className={`${labelClass} mb-0`} htmlFor="prompt-a">
                  Prompt under test
                </label>
                <Badge tone="info">Current</Badge>
              </div>
              <textarea
                id="prompt-a"
                rows={14}
                className={textareaClass}
                value={promptA}
                onChange={(e) => setPromptA(e.target.value)}
              />
            </Card>

            <Card className="p-4">
              <div className="mb-2 flex items-center justify-between gap-2">
                <label className={`${labelClass} mb-0`} htmlFor="prompt-b">
                  Rewrite to compare
                </label>
                <label className="flex items-center gap-2 text-xs text-gray-300">
                  <input
                    type="checkbox"
                    className="h-3.5 w-3.5 rounded border-white/20 bg-black/40 accent-indigo-500"
                    checked={compare}
                    onChange={(e) => {
                      setCompare(e.target.checked);
                      if (e.target.checked && !promptB.trim()) setPromptB(promptA);
                    }}
                  />
                  Compare A/B
                </label>
              </div>
              <textarea
                id="prompt-b"
                rows={14}
                disabled={!compare}
                className={`${textareaClass} disabled:opacity-40`}
                value={promptB}
                onChange={(e) => setPromptB(e.target.value)}
                placeholder="Both prompts see byte-identical cases, so any difference in the score is the prompt."
              />
            </Card>
          </div>

          <div className="flex flex-wrap items-center gap-4">
            <label className="flex items-center gap-2 text-xs text-gray-400">
              Temperature
              <input
                type="range"
                min={0}
                max={1}
                step={0.1}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                className="accent-indigo-500"
              />
              <span className="w-8 text-gray-300">{temperature.toFixed(1)}</span>
            </label>
            <p className="text-xs text-gray-500">
              Lower is steadier. A prompt whose score moves with temperature is under-specified,
              which is itself a finding.
            </p>
          </div>

          {result && (
            <div className="space-y-5">
              <div className="flex flex-wrap items-center gap-3 rounded-lg border border-white/5 bg-card px-4 py-3">
                <Badge tone={result.llm_available ? 'success' : 'danger'}>
                  {result.llm_available ? 'Generated' : 'No LLM'}
                </Badge>
                <span className="text-xs text-gray-400">model: {result.model}</span>
                {result.variants.map((v) => (
                  <span key={v.label} className="flex items-center gap-2 text-xs text-gray-400">
                    <span className="text-gray-300">{v.label}</span>
                    <Badge tone={scoreTone(v.score)}>{pct(v.score)}</Badge>
                    <span className="text-gray-600">#{v.prompt_hash}</span>
                  </span>
                ))}
              </div>

              {result.deltas.length > 0 && (
                <Card className="overflow-hidden">
                  <div className="flex items-center gap-2 border-b border-white/5 px-4 py-3">
                    <GitCompare className="h-4 w-4 text-indigo-400" />
                    <h3 className="text-sm font-semibold text-white">What the rewrite changed</h3>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full min-w-[480px] text-left text-sm">
                      <thead>
                        <tr className="border-b border-white/5 text-[10px] uppercase tracking-wider text-gray-500">
                          <th className="px-4 py-2 font-semibold">Rule</th>
                          <th className="px-3 py-2 text-right font-semibold">Current</th>
                          <th className="px-3 py-2 text-right font-semibold">Rewrite</th>
                          <th className="px-4 py-2 text-right font-semibold">Change</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.deltas.map((d) => (
                          <tr key={d.rule} className="border-b border-white/5 last:border-0">
                            <td className="px-4 py-2 text-gray-200">{d.title}</td>
                            <td className="px-3 py-2 text-right text-gray-400">{pct(d.baseline_score)}</td>
                            <td className="px-3 py-2 text-right text-gray-400">{pct(d.variant_score)}</td>
                            <td className="px-4 py-2 text-right">
                              {d.delta === null || d.delta === 0 ? (
                                <span className="text-gray-600">no change</span>
                              ) : (
                                <Badge tone={d.delta > 0 ? 'success' : 'danger'}>
                                  {d.delta > 0 ? '+' : ''}{d.delta}%
                                </Badge>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              )}

              <div className="grid gap-5 lg:grid-cols-2">
                {result.variants.map((variant) => (
                  <div key={variant.label} className="space-y-4">
                    <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-400">
                      {variant.label}
                    </h2>
                    <RuleMatrix variant={variant} />
                    <CaseResults variant={variant} />
                  </div>
                ))}
              </div>
            </div>
          )}

          {!result && !running && (
            <Card className="p-6 text-center text-sm text-gray-400">
              Run the suite to see, rule by rule, where this prompt holds and where it slips.
            </Card>
          )}
        </div>
      )}

      {/* ---------------------------------------------------------------- cases */}
      {tab === 'cases' && (
        <div className="space-y-4">
          <Card className="p-4">
            <div className="grid gap-3 sm:grid-cols-[1fr_auto] sm:items-end">
              <div>
                <label className={labelClass} htmlFor="suite-name">Suite name</label>
                <input
                  id="suite-name"
                  className="w-full rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                  value={suiteName}
                  onChange={(e) => setSuiteName(e.target.value)}
                />
              </div>
              <AddCaseButton onAdd={() => setCases((c) => [...c, blankCase(c.length + 1)])} />
            </div>
            {suites.length > 0 && (
              <div className="mt-4 space-y-2">
                <p className={labelClass}>Saved suites</p>
                {suites.map((suite) => (
                  <div
                    key={suite.id}
                    className={`flex items-center gap-2 rounded-lg border px-3 py-2 ${
                      suite.id === suiteId ? 'border-indigo-500/40 bg-indigo-500/5' : 'border-white/5'
                    }`}
                  >
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm text-white">{suite.name}</span>
                      <span className="block text-xs text-gray-500">
                        {suite.cases.length} case{suite.cases.length === 1 ? '' : 's'} · updated{' '}
                        {new Date(suite.updated_at).toLocaleString()}
                      </span>
                    </span>
                    <Button variant="ghost" size="sm" onClick={() => onLoadSuite(suite.id)}>
                      Load
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      aria-label={`Delete ${suite.name}`}
                      onClick={() => onDeleteSuite(suite.id)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </Card>

          <p className="text-xs text-gray-500">
            A case describes what the turn <em>permits</em> — never what the prompt says. That is what
            lets the same suite grade a rewritten prompt without being rewritten itself.
          </p>

          {cases.map((value, i) => (
            <CaseEditor
              key={value.id || i}
              index={i}
              value={value}
              onChange={(next) => setCases((c) => c.map((x, j) => (j === i ? next : x)))}
              onRemove={() => setCases((c) => c.filter((_, j) => j !== i))}
            />
          ))}
        </div>
      )}

      {/* ----------------------------------------------------------- playground */}
      {tab === 'playground' && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card className="space-y-3 p-4">
            <div>
              <label className={labelClass} htmlFor="play-case">Case</label>
              <select
                id="play-case"
                className="w-full rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none"
                value={playCase}
                onChange={(e) => { setPlayCase(Number(e.target.value)); setPlayScore(null); }}
              >
                {cases.map((c, i) => (
                  <option key={c.id || i} value={i}>{c.name || c.question || `Case ${i + 1}`}</option>
                ))}
              </select>
            </div>

            {cases[playCase] && (
              <>
                <div>
                  <p className={labelClass}>Question</p>
                  <p className="rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-gray-200">
                    {cases[playCase].question}
                  </p>
                </div>
                <div>
                  <p className={labelClass}>Context the answer may use</p>
                  <pre className="max-h-40 overflow-auto whitespace-pre-wrap rounded-lg border border-white/10 bg-black/40 px-3 py-2 font-mono text-[11px] text-gray-400">
                    {cases[playCase].context || '(none)'}
                  </pre>
                </div>
              </>
            )}

            <div>
              <label className={labelClass} htmlFor="play-answer">
                Answer — generate one, or paste what the live assistant said
              </label>
              <textarea
                id="play-answer"
                rows={6}
                className={textareaClass}
                value={playAnswer}
                onChange={(e) => setPlayAnswer(e.target.value)}
              />
            </div>

            <div className="flex flex-wrap gap-2">
              <Button
                size="sm"
                loading={playBusy}
                leftIcon={<RefreshCw className="h-4 w-4" />}
                onClick={onGenerateAnswer}
              >
                Generate & grade
              </Button>
              <Button
                variant="secondary"
                size="sm"
                loading={playBusy}
                leftIcon={<ClipboardCheck className="h-4 w-4" />}
                onClick={onGradeAnswer}
                disabled={!playAnswer.trim()}
              >
                Grade this answer
              </Button>
            </div>
            <p className="text-xs text-gray-500">
              Grading needs no LLM — it is the same deterministic check the suite runs, so a pasted
              transcript can be scored on a machine with no key configured.
            </p>
          </Card>

          <div className="space-y-4">
            {playScore ? (
              <Card className="overflow-hidden">
                <div className="flex items-center justify-between gap-2 border-b border-white/5 px-4 py-3">
                  <h3 className="text-sm font-semibold text-white">Verdict</h3>
                  <Badge tone={scoreTone(playScore.score)}>{pct(playScore.score)}</Badge>
                </div>
                <ul className="space-y-2 p-4">
                  {playScore.rules.map((verdict) => (
                    <li key={verdict.rule} className="text-xs">
                      <span
                        className={
                          verdict.status === 'fail' ? 'text-rose-400'
                          : verdict.status === 'pass' ? 'text-emerald-400'
                          : 'text-gray-600'
                        }
                      >
                        {verdict.status === 'fail' ? '✕' : verdict.status === 'pass' ? '✓' : '—'}
                      </span>{' '}
                      <span className="font-medium text-gray-200">{verdict.title}</span>
                      <span className="text-gray-500"> — {verdict.reason}</span>
                    </li>
                  ))}
                </ul>
              </Card>
            ) : (
              <Card className="p-6 text-center text-sm text-gray-400">
                No verdict yet. Generate an answer, or paste one and grade it.
              </Card>
            )}

            <Card className="p-4">
              <h3 className="mb-3 text-sm font-semibold text-white">What each rule measures</h3>
              <ul className="space-y-2">
                {catalog.map((rule) => (
                  <li key={rule.id} className="text-xs text-gray-500">
                    <span className="font-medium text-gray-300">{rule.title}</span> — {rule.description}
                  </li>
                ))}
              </ul>
            </Card>
          </div>
        </div>
      )}

      {/* -------------------------------------------------------------- history */}
      {tab === 'history' && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-3">
            <Button variant="secondary" size="sm" leftIcon={<RefreshCw className="h-4 w-4" />} onClick={() => void loadRuns()}>
              Refresh
            </Button>
            <p className="text-xs text-gray-500">
              Each row stores the prompt fingerprint and the model beside the score — a history that
              cannot tell a prompt edit from a model swap is noise.
            </p>
          </div>

          {runs.length === 0 ? (
            <Card className="p-6 text-center text-sm text-gray-400">
              No runs recorded yet. Runs are stored automatically whenever an LLM is configured.
            </Card>
          ) : (
            <Card className="overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full min-w-[640px] text-left text-sm">
                  <thead>
                    <tr className="border-b border-white/5 text-[10px] uppercase tracking-wider text-gray-500">
                      <th className="px-4 py-2 font-semibold">When</th>
                      <th className="px-3 py-2 font-semibold">Variant</th>
                      <th className="px-3 py-2 font-semibold">Prompt</th>
                      <th className="px-3 py-2 font-semibold">Model</th>
                      <th className="px-3 py-2 text-right font-semibold">Clean</th>
                      <th className="px-3 py-2 text-right font-semibold">Score</th>
                      <th className="px-4 py-2" />
                    </tr>
                  </thead>
                  <tbody>
                    {runs.map((run) => (
                      <tr key={run.id} className="border-b border-white/5 last:border-0">
                        <td className="px-4 py-2 text-gray-300">{new Date(run.created_at).toLocaleString()}</td>
                        <td className="px-3 py-2 text-gray-300">{run.label}</td>
                        <td className="px-3 py-2 font-mono text-xs text-gray-500">#{run.prompt_hash}</td>
                        <td className="px-3 py-2 text-xs text-gray-500">{run.model}</td>
                        <td className="px-3 py-2 text-right text-gray-400">
                          {run.clean_cases}/{run.total_cases}
                        </td>
                        <td className="px-3 py-2 text-right">
                          <Badge tone={scoreTone(run.score)}>{pct(run.score)}</Badge>
                        </td>
                        <td className="px-4 py-2 text-right">
                          <Button variant="ghost" size="sm" onClick={() => void onOpenRun(run.id)}>
                            Open <ArrowRight className="h-3.5 w-3.5" />
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {openRun && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-400">
                  Run detail — {openRun.label} #{openRun.prompt_hash}
                </h2>
                <Button variant="ghost" size="sm" onClick={() => setOpenRun(null)}>Close</Button>
              </div>
              <RuleMatrix variant={openRun} />
              <CaseResults variant={openRun} />
            </div>
          )}
        </div>
      )}
    </div>
  );
};
