import { describe, it, expect } from 'vitest';
import { blankCase, pct, scoreTone } from './promptLabHelpers';

/**
 * A null score means no rule applied to that turn. Rendering it as 0% would read as a
 * total failure and as 100% as a clean sweep — both are lies about a measurement that
 * was never taken, which is the whole reason the backend distinguishes N/A from a pass.
 */
describe('score rendering', () => {
  it('shows an absent score as a dash rather than a number', () => {
    expect(pct(null)).toBe('—');
    expect(scoreTone(null)).toBe('neutral');
  });

  it('drops the decimal only when there is nothing to lose', () => {
    expect(pct(100)).toBe('100%');
    expect(pct(87.5)).toBe('87.5%');
  });

  it('reads green, amber and red at the documented boundaries', () => {
    expect(scoreTone(90)).toBe('success');
    expect(scoreTone(89.9)).toBe('warning');
    expect(scoreTone(70)).toBe('warning');
    expect(scoreTone(69.9)).toBe('danger');
  });
});

describe('a new case', () => {
  it('starts with the prompt’s own 1-2 line limit and nothing else assumed', () => {
    const created = blankCase(3);
    expect(created.expectations.max_lines).toBe(2);
    expect(created.expectations.link_allowed).toBe(false);
    expect(created.expectations.must_contain).toEqual([]);
    expect(created.history).toEqual([]);
  });

  it('does not share expectation objects between cases', () => {
    const first = blankCase(1);
    const second = blankCase(2);
    first.expectations.must_contain.push('x');
    expect(second.expectations.must_contain).toEqual([]);
  });
});
