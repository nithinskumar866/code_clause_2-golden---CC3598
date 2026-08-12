import { describe, it, expect } from 'vitest';
import { NAV_GROUPS, PAGE_TITLES } from './navConfig';
import type { PageId } from './navConfig';
import { resolveNavigation } from './navIntent';

const ITEMS = NAV_GROUPS.flatMap(group => group.items);

/**
 * The registry is what the assistant can reach. A route added without the words
 * a person would use for it is a page the assistant can never take anyone to,
 * and nothing else in the app would notice — hence these tests.
 */
describe('navConfig as the assistant’s route registry', () => {
  it('gives every route a description and vocabulary', () => {
    for (const item of ITEMS) {
      expect(item.description, `${item.id} needs a description`).toBeTruthy();
      expect(item.keywords.length, `${item.id} needs keywords`).toBeGreaterThan(0);
    }
  });

  it('titles every page it lists', () => {
    for (const item of ITEMS) {
      expect(PAGE_TITLES[item.id], `${item.id} needs a title`).toBeTruthy();
    }
  });

  it('uses each route id exactly once', () => {
    const ids = ITEMS.map(item => item.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('can reach every route by asking for it by name', () => {
    for (const item of ITEMS) {
      const result = resolveNavigation(`take me to ${item.name}`);
      expect(result, `${item.id} is unreachable by name`).not.toBeNull();

      // Reaching it as one of a couple of offered choices is acceptable; being
      // absent, or resolving to a different page, is not.
      const reached =
        result?.kind === 'navigate' ? [result.option.pageId]
        : result?.kind === 'describe' ? [result.option.pageId]
        : result?.kind === 'ambiguous' ? result.options.map(option => option.pageId)
        : [];

      expect(reached, `"${item.name}" resolved to ${JSON.stringify(reached)}`).toContain(item.id);
    }
  });

  it('never reports a real page as missing', () => {
    for (const item of ITEMS) {
      expect(resolveNavigation(`go to ${item.name}`)?.kind).not.toBe('unknown');
    }
  });

  it('excludes the profile page, which needs a selected candidate', () => {
    const ids = ITEMS.map(item => item.id) as PageId[];
    expect(ids).not.toContain('profile');
  });
});
