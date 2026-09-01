import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { api } from '../api/client';
import type { JobDetail } from '../types';

/**
 * Posting a job, either way.
 *
 * Upload runs the same extraction the recruiter platform's documents go through:
 * the posting is parsed, structured and embedded before the request returns, so a
 * job is searchable the moment it appears rather than after a separate indexing
 * step someone has to remember.
 */
export function PostJobPage() {
  const [posted, setPosted] = useState<JobDetail | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [form, setForm] = useState({
    title: '', company: '', location: '',
    workMode: '', employmentType: '', seniorityLevel: '',
    minYears: '', maxYears: '', salaryMin: '', salaryMax: '', salaryCurrency: 'USD',
    description: '', requiredSkills: '', preferredSkills: '', applyUrl: '',
  });

  const set = (key: keyof typeof form) => (
    event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>,
  ) => setForm(previous => ({ ...previous, [key]: event.target.value }));

  const onDrop = useCallback(async (files: File[]) => {
    const file = files[0];
    if (!file) return;

    setBusy(true);
    setError(null);
    try {
      const result = await api.uploadJob(file);
      setPosted(result.job);
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : 'The job could not be read.');
    } finally {
      setBusy(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: false,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt', '.md'],
    },
  });

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const job = await api.createJob({
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
    <div className="page">
      <h1>Post a job</h1>
      <p className="lede">
        Upload a job description and it is parsed, structured and indexed automatically —
        or fill in the form.
      </p>

      {error && <div className="banner error">{error}</div>}

      {posted && (
        <div className="card" style={{ padding: 16, marginBottom: 20 }}>
          <h3 style={{ margin: '0 0 6px' }}>Posted: {posted.summary.title}</h3>
          <div className="meta" style={{ color: 'var(--muted)', fontSize: 13 }}>
            {posted.summary.company || 'company not stated'} · {posted.summary.location || 'location not stated'}
            {' '}· {posted.summary.workMode} · extracted {posted.extractionMode}
            {' '}· {posted.summary.isIndexed ? 'indexed' : 'NOT indexed'}
          </div>
          <div className="chips" style={{ marginTop: 10 }}>
            {posted.summary.requiredSkills.map(skill => <span key={skill} className="chip">{skill}</span>)}
          </div>
        </div>
      )}

      <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`} style={{ margin: '0 0 22px' }}>
        <input {...getInputProps()} />
        <strong>{busy ? 'Processing…' : 'Drop a job description here'}</strong>
        PDF, DOCX, TXT or MD — or click to browse
      </div>

      <form className="card" style={{ padding: 18 }} onSubmit={submit}>
        <div className="form-grid">
          <div className="field full">
            <label>Job title *</label>
            <input value={form.title} onChange={set('title')} required placeholder="Senior .NET Engineer" />
          </div>
          <div className="field"><label>Company</label><input value={form.company} onChange={set('company')} /></div>
          <div className="field"><label>Location</label><input value={form.location} onChange={set('location')} /></div>

          <div className="field">
            <label>Work mode</label>
            <select value={form.workMode} onChange={set('workMode')}>
              <option value="">Unspecified</option><option>Remote</option><option>Hybrid</option><option>Onsite</option>
            </select>
          </div>
          <div className="field">
            <label>Employment type</label>
            <select value={form.employmentType} onChange={set('employmentType')}>
              <option value="">Unspecified</option><option>Full-time</option><option>Part-time</option>
              <option>Contract</option><option>Internship</option>
            </select>
          </div>
          <div className="field">
            <label>Seniority</label>
            <select value={form.seniorityLevel} onChange={set('seniorityLevel')}>
              <option value="">Unspecified</option><option>Junior</option><option>Mid</option>
              <option>Senior</option><option>Lead</option><option>Staff</option><option>Principal</option>
            </select>
          </div>
          <div className="field"><label>Apply URL</label><input value={form.applyUrl} onChange={set('applyUrl')} /></div>

          <div className="field"><label>Min years</label><input type="number" min="0" value={form.minYears} onChange={set('minYears')} /></div>
          <div className="field"><label>Max years</label><input type="number" min="0" value={form.maxYears} onChange={set('maxYears')} /></div>
          <div className="field"><label>Salary min</label><input type="number" min="0" value={form.salaryMin} onChange={set('salaryMin')} /></div>
          <div className="field"><label>Salary max</label><input type="number" min="0" value={form.salaryMax} onChange={set('salaryMax')} /></div>

          <div className="field full">
            <label>Description</label>
            <textarea rows={7} value={form.description} onChange={set('description')}
              placeholder="Paste the full job description. Requirements and responsibilities are extracted from it." />
          </div>
          <div className="field full">
            <label>Required skills (comma or newline separated)</label>
            <textarea rows={2} value={form.requiredSkills} onChange={set('requiredSkills')} />
          </div>
          <div className="field full">
            <label>Preferred skills</label>
            <textarea rows={2} value={form.preferredSkills} onChange={set('preferredSkills')} />
          </div>
        </div>

        <button className="primary" type="submit" disabled={busy || !form.title.trim()} style={{ marginTop: 16 }}>
          {busy ? 'Posting…' : 'Post job'}
        </button>
      </form>
    </div>
  );
}

function splitList(value: string): string[] {
  return value
    .split(/[\n,;]/)
    .map(part => part.trim())
    .filter(part => part.length > 0);
}
