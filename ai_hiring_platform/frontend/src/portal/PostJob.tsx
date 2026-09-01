import { useEffect, useState, type ChangeEvent, type FC, type FormEvent } from 'react';
import { Megaphone } from 'lucide-react';
import { portalApi } from './api';
import { usePortalState } from './portal-context';
import { FileDrop } from './FileDrop';
import type { JobBulkUploadResult, JobDetail } from './types';
import { PageHeader } from '../components/ui/PageHeader';
import { Button } from '../components/ui/Button';
import { Banner } from '../components/ui/Banner';
import { Input } from '../components/ui/Input';
import { Select } from '../components/ui/Select';

const asOptions = (values: string[], blank = 'Unspecified') =>
  values.map(value => ({ value, label: value || blank }));

/** Mirrors JobsController.MaxBulkFiles. The server enforces it; this only says so. */
const MAX_BULK_FILES = 50;

const WORK_MODES = ['', 'Remote', 'Hybrid', 'Onsite'];
const EMPLOYMENT = ['', 'Full-time', 'Part-time', 'Contract', 'Internship'];
const SENIORITIES = ['', 'Junior', 'Mid', 'Senior', 'Lead', 'Staff', 'Principal'];

const FIELD_LABEL = 'mb-1.5 block text-xs font-semibold uppercase tracking-wider text-gray-400';
const TEXTAREA =
  'w-full rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm text-white placeholder:text-gray-500 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500';

/**
 * Posting a role to the candidate-facing board.
 *
 * Upload runs the same extraction the recruiter platform's documents go through:
 * the posting is parsed, structured and embedded before the request returns, so
 * a job is searchable the moment it appears rather than after a separate
 * indexing step someone has to remember.
 */
export const PostJob: FC = () => {
  const { engage, reachable } = usePortalState();

  const [posted, setPosted] = useState<JobDetail | null>(null);
  const [bulk, setBulk] = useState<JobBulkUploadResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(engage, [engage]);

  const [form, setForm] = useState({
    title: '', company: '', location: '',
    workMode: '', employmentType: '', seniorityLevel: '',
    minYears: '', maxYears: '', salaryMin: '', salaryMax: '', salaryCurrency: 'USD',
    description: '', requiredSkills: '', preferredSkills: '', applyUrl: '',
  });

  const set = (key: keyof typeof form) => (
    event: ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>,
  ) => setForm(previous => ({ ...previous, [key]: event.target.value }));

  const onFile = async (file: File) => {
    setBusy(true);
    setError(null);
    setBulk(null);
    try {
      const result = await portalApi.uploadJob(file);
      setPosted(result.job);
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : 'The job could not be read.');
    } finally {
      setBusy(false);
    }
  };

  /**
   * A batch is reported per file rather than as one verdict.
   *
   * Extraction may call the model per document, so thirty JDs is minutes of work —
   * and a couple of unreadable scans in that batch is normal. Collapsing it to
   * "3 failed" would leave the uploader with no idea which three or what to fix,
   * and re-uploading everything to retry two files is the wrong remedy.
   */
  const onFiles = async (files: File[]) => {
    if (files.length === 1) {
      await onFile(files[0]);
      return;
    }
    setBusy(true);
    setError(null);
    setPosted(null);
    setBulk(null);
    try {
      setBulk(await portalApi.uploadJobsBulk(files));
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : 'The batch could not be uploaded.');
    } finally {
      setBusy(false);
    }
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const job = await portalApi.createJob({
        title: form.title,
        company: form.company || undefined,
        location: form.location || undefined,
        workMode: form.workMode || undefined,
        employmentType: form.employmentType || undefined,
        seniorityLevel: form.seniorityLevel || undefined,
        // An empty box means "not stated", which is different from zero: a job
        // requiring 0 years is a real claim, and one that says nothing is not.
        minYearsExperience: form.minYears ? Number(form.minYears) : null,
        maxYearsExperience: form.maxYears ? Number(form.maxYears) : null,
        salaryMin: form.salaryMin ? Number(form.salaryMin) : null,
        salaryMax: form.salaryMax ? Number(form.salaryMax) : null,
        salaryCurrency: form.salaryCurrency || undefined,
        description: form.description || undefined,
        requiredSkills: splitList(form.requiredSkills),
        preferredSkills: splitList(form.preferredSkills),
        applyUrl: form.applyUrl || undefined,
      });
      setPosted(job);
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : 'The job could not be posted.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader
        icon={<Megaphone className="h-5 w-5" />}
        title="Post a Job"
        description="Upload a job description and it is parsed, structured and indexed automatically — or fill in the form."
      />

      {reachable === false && (
        <Banner variant="error" title="Job portal API unreachable">
          Start it with <code>dotnet run</code> in <code>job_portal/backend/JobPortal.Api</code>.
        </Banner>
      )}
      {error && <Banner variant="error" onDismiss={() => setError(null)}>{error}</Banner>}

      {posted && (
        <Banner variant="success" title={`Posted: ${posted.summary.title}`}>
          <p>
            {posted.summary.company || 'company not stated'} ·{' '}
            {posted.summary.location || 'location not stated'} · {posted.summary.workMode} · extracted{' '}
            {posted.extractionMode} · {posted.summary.isIndexed ? 'indexed' : 'NOT indexed'}
          </p>
          {posted.summary.requiredSkills.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {posted.summary.requiredSkills.map(skill => (
                <span key={skill} className="rounded-md border border-white/10 bg-white/5 px-1.5 py-0.5 text-[10px] text-gray-300">
                  {skill}
                </span>
              ))}
            </div>
          )}
        </Banner>
      )}

      {bulk && (
        <Banner
          variant={bulk.failed === 0 ? 'success' : 'warning'}
          title={
            `${bulk.succeeded} of ${bulk.total} job description${bulk.total === 1 ? '' : 's'} posted` +
            (bulk.failed > 0 ? ` — ${bulk.failed} could not be read` : '')
          }
          onDismiss={() => setBulk(null)}
        >
          {/* Per file, so a failure names itself and only the failures get retried. */}
          <ul className="mt-2 max-h-56 space-y-1 overflow-y-auto pr-1">
            {bulk.items.map((item, index) => (
              <li
                key={`${item.filename}-${index}`}
                className="flex items-start gap-2 text-xs"
              >
                <span className={item.success ? 'text-emerald-400' : 'text-amber-400'}>
                  {item.success ? '✓' : '✕'}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="text-gray-300">{item.filename}</span>
                  {item.success && item.job
                    ? <span className="text-gray-500"> — {item.job.title}</span>
                    : <span className="text-amber-400/80"> — {item.error}</span>}
                </span>
              </li>
            ))}
          </ul>
        </Banner>
      )}

      <FileDrop
        multiple
        busy={busy}
        busyLabel="Processing…"
        label="Drop job descriptions here"
        hint={`PDF, DOCX, TXT or MD — select up to ${MAX_BULK_FILES} files for a bulk upload`}
        onFile={(file) => void onFile(file)}
        onFiles={(files) => void onFiles(files)}
      />

      <form onSubmit={submit} className="space-y-4 rounded-xl border border-white/5 bg-card p-5">
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="sm:col-span-2">
            <Input
              id="portal-title"
              label="Job title *"
              value={form.title}
              onChange={set('title')}
              required
              placeholder="Senior .NET Engineer"
            />
          </div>

          <Input id="portal-company" label="Company" value={form.company} onChange={set('company')} />
          <Input id="portal-location" label="Location" value={form.location} onChange={set('location')} />

          <Select label="Work mode" className="w-full" options={asOptions(WORK_MODES)}
                  value={form.workMode} onChange={set('workMode')} />
          <Select label="Employment type" className="w-full" options={asOptions(EMPLOYMENT)}
                  value={form.employmentType} onChange={set('employmentType')} />
          <Select label="Seniority" className="w-full" options={asOptions(SENIORITIES)}
                  value={form.seniorityLevel} onChange={set('seniorityLevel')} />
          <Input id="portal-apply" label="Apply URL" value={form.applyUrl} onChange={set('applyUrl')} />

          <Input id="portal-min-years" label="Min years" type="number" min="0"
                 value={form.minYears} onChange={set('minYears')} />
          <Input id="portal-max-years" label="Max years" type="number" min="0"
                 value={form.maxYears} onChange={set('maxYears')} />
          <Input id="portal-salary-min" label="Salary min" type="number" min="0"
                 value={form.salaryMin} onChange={set('salaryMin')} />
          <Input id="portal-salary-max" label="Salary max" type="number" min="0"
                 value={form.salaryMax} onChange={set('salaryMax')} />

          <div className="sm:col-span-2">
            <label htmlFor="portal-description" className={FIELD_LABEL}>Description</label>
            <textarea
              id="portal-description"
              rows={7}
              value={form.description}
              onChange={set('description')}
              className={TEXTAREA}
              placeholder="Paste the full job description. Requirements and responsibilities are extracted from it."
            />
          </div>

          <div className="sm:col-span-2">
            <label htmlFor="portal-required" className={FIELD_LABEL}>
              Required skills (comma or newline separated)
            </label>
            <textarea id="portal-required" rows={2} value={form.requiredSkills}
                      onChange={set('requiredSkills')} className={TEXTAREA} />
          </div>

          <div className="sm:col-span-2">
            <label htmlFor="portal-preferred" className={FIELD_LABEL}>Preferred skills</label>
            <textarea id="portal-preferred" rows={2} value={form.preferredSkills}
                      onChange={set('preferredSkills')} className={TEXTAREA} />
          </div>
        </div>

        <Button type="submit" loading={busy} disabled={!form.title.trim()}>
          {busy ? 'Posting…' : 'Post job'}
        </Button>
      </form>
    </div>
  );
};

function splitList(value: string): string[] {
  return value
    .split(/[\n,;]/)
    .map(part => part.trim())
    .filter(part => part.length > 0);
}
