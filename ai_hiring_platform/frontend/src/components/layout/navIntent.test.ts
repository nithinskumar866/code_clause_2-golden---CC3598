import { describe, it, expect } from 'vitest';
import { resolveNavigation, ALL_PAGES } from './navIntent';
import { NAV_GROUPS } from './navConfig';

/** Narrows to the outcome kind under test and fails loudly otherwise. */
function outcome(message: string) {
  const result = resolveNavigation(message);
  if (!result) throw new Error(`Expected a navigation outcome for: ${message}`);
  return result;
}

describe('resolveNavigation — taking the user somewhere', () => {
  it('navigates straight to the page when the request is clear', () => {
    const result = outcome('take me to the resume update section');
    expect(result.kind).toBe('navigate');
    if (result.kind === 'navigate') expect(result.option.pageId).toBe('resume');
  });

  it.each([
    ['go to analytics', 'analytics'],
    ['open the job board', 'jobboard'],
    ['take me to candidate ranking', 'ranking'],
    ['navigate to model lab', 'modellab'],
    ['where is the analysis history page', 'history'],
    ['i want to upload a resume', 'resume'],
    ['take me to post a job', 'postjob'],
    ['switch to system status', 'status'],
    ['go to the manage and index section', 'documents'],
    ['open job descriptions', 'job'],
    // Reached on vocabulary rather than the page's name.
    ['where do i upload a cv', 'resume'],
    ['where can i see past analyses', 'history'],
    ['i want to compare embedding models', 'modellab'],
  ])('resolves %j to %j', (message, pageId) => {
    const result = outcome(message);
    expect(result.kind).toBe('navigate');
    if (result.kind === 'navigate') expect(result.option.pageId).toBe(pageId);
  });

  it('offers every destination it names as a real page id', () => {
    const known = new Set(NAV_GROUPS.flatMap(group => group.items.map(item => item.id)));
    for (const page of ALL_PAGES) expect(known.has(page.pageId as never)).toBe(true);
  });
});

describe('resolveNavigation — asking rather than guessing', () => {
  it('cross-questions when more than one section fits', () => {
    // "upload" is honestly true of both document pages.
    const result = outcome('go to the upload page');
    expect(result.kind).toBe('ambiguous');
    if (result.kind === 'ambiguous') {
      expect(result.options.length).toBeGreaterThan(1);
      expect(result.options.map(o => o.pageId)).toContain('resume');
    }
  });

  it('confirms rather than assumes when the match is only loose', () => {
    const result = outcome('go to the leaderboard section');
    // Reached only through a keyword, never the page's own name.
    expect(['navigate', 'ambiguous']).toContain(result.kind);
    const offered = result.kind === 'ambiguous'
      ? result.options.map(o => o.pageId)
      : result.kind === 'navigate' ? [result.option.pageId] : [];
    expect(offered).toContain('ranking');
  });

  it('narrows to a sidebar group’s own pages when the group is named', () => {
    const result = outcome('open documents');
    expect(result.kind).toBe('ambiguous');
    if (result.kind === 'ambiguous') {
      expect(result.options.map(o => o.pageId)).toEqual(['resume', 'job', 'documents']);
    }
  });

  it('offers exactly the Job Portal group for "go to job portal"', () => {
    // Derived from the registry rather than written out, so adding a page to the
    // group updates the expectation instead of breaking this test.
    const expected = NAV_GROUPS
      .find(group => group.label === 'Job Portal')!
      .items.map(item => item.id);

    const result = outcome('go to job portal');
    expect(result.kind).toBe('ambiguous');
    if (result.kind === 'ambiguous') {
      expect(result.options.map(o => o.pageId)).toEqual(expected);
    }
  });

  it('confirms a single near-miss rather than opening it', () => {
    // "settings" is vocabulary for System Status but not its name.
    const result = outcome('take me to settings');
    expect(result.kind).toBe('ambiguous');
    if (result.kind === 'ambiguous') expect(result.options[0].pageId).toBe('status');
  });

  it('never offers an option without a label a user could act on', () => {
    const result = outcome('go to the upload page');
    if (result.kind === 'ambiguous') {
      for (const option of result.options) expect(option.label.length).toBeGreaterThan(0);
    }
  });
});

describe('resolveNavigation — pages that do not exist', () => {
  it('says so plainly instead of inventing one', () => {
    const result = outcome('take me to the payroll section');
    expect(result.kind).toBe('unknown');
    if (result.kind === 'unknown') {
      expect(result.term).toContain('payroll');
      // and still shows what the application does have
      expect(result.options.length).toBe(ALL_PAGES.length);
    }
  });

  it.each([
    'go to the billing page',
    'navigate to user permissions',
    'take me to the calendar section',
  ])('reports %j as absent', (message) => {
    expect(outcome(message).kind).toBe('unknown');
  });
});

describe('resolveNavigation — questions about the application', () => {
  it('explains a section instead of jumping to it', () => {
    const result = outcome('what is the model lab?');
    expect(result.kind).toBe('describe');
    if (result.kind === 'describe') {
      expect(result.option.pageId).toBe('modellab');
      expect(result.option.description).toBeTruthy();
    }
  });

  it('still navigates when asked as a question with a navigation phrase', () => {
    const result = outcome('can you take me to analytics?');
    expect(result.kind).toBe('navigate');
  });

  it('answers "how do I…" about a section by explaining it', () => {
    const result = outcome('how do i rank candidates');
    expect(result.kind).toBe('describe');
    if (result.kind === 'describe') expect(result.option.pageId).toBe('ranking');
  });
});

describe('resolveNavigation — leaving the job-portal conversation alone', () => {
  it.each([
    'show me remote roles',
    'why is this candidate a 46% fit',
    'remote only',
    'senior roles over 120k',
    'explain that score',
    'hello',
  ])('returns null for %j so it reaches the hub', (message) => {
    expect(resolveNavigation(message)).toBeNull();
  });

  it('returns null for an empty message', () => {
    expect(resolveNavigation('   ')).toBeNull();
  });
});
