import { NAV_GROUPS } from './navConfig';
import type { NavOption, NavOutcome } from '../../portal/navigator';

/**
 * Reading a recruiter's sentence as a request to move around the application.
 *
 * Entirely deterministic, and scored against `NAV_GROUPS` — the same registry
 * the sidebar renders. That is the point: the assistant can only ever offer a
 * page that genuinely exists, so "sorry, there is no such page" is a fact about
 * the app rather than a model's guess. An LLM asked "which page do they want?"
 * will happily answer "the Settings page" for an app that has none, and the
 * user then hunts for something that was never built.
 *
 * The other half of that rule is restraint: `resolve` returns null unless the
 * message really reads as navigation, because everything it claims is taken
 * away from the job-portal conversation. "Show me remote roles" is a search,
 * not a request for the Job Board page, and answering it with navigation would
 * break the assistant's actual job.
 */

/** Filler, and the verbs/nouns that signal navigation rather than name a page. */
const STOPWORDS = new Set([
  'go', 'goto', 'to', 'take', 'bring', 'me', 'my', 'i', 'we', 'us', 'the', 'a', 'an',
  'of', 'for', 'on', 'in', 'at', 'into', 'from', 'and', 'or', 'is', 'are', 'was',
  'can', 'could', 'would', 'will', 'shall', 'do', 'does', 'did', 'you', 'your',
  'please', 'want', 'wanna', 'need', 'like', 'let', 'get', 'got', 'see', 'look',
  'where', 'how', 'what', 'whats', 'which', 'why', 'when', 'this', 'that', 'it',
  'there', 'here', 'am', 'be', 'been', 'have', 'has', 'had', 'me', 'us', 'about',
  'section', 'sections', 'page', 'pages', 'screen', 'tab', 'view', 'area', 'part',
  'place', 'module', 'menu', 'application', 'app', 'website', 'site', 'system',
  'platform', 'show', 'display', 'jump', 'switch', 'navigate', 'redirect', 'move',
  'head', 'visit', 'open', 'find', 'tell', 'explain', 'again', 'now', 'then',
]);

/** Unambiguous "move me somewhere" phrasing. */
const STRONG_NAV =
  /\b(go\s*to|goto|take\s+me|bring\s+me|send\s+me|navigate|redirect|switch\s+to|jump\s+to|move\s+to|head\s+to|open\s+up|back\s+to)\b/;

/** Naming a destination as a place is itself a navigation cue. */
const PLACE_NOUN = /\b(page|section|screen|tab|view|area|module|menu)\b/;

/** Weaker cues: enough only when a page's actual name is also mentioned. */
const WEAK_NAV = /\b(open|show|display|see|find|where|visit|browse|need|want)\b/;

/** Asking what something is, rather than asking to be taken there. */
const QUESTION = /\b(what|whats|which|how|why|explain|describe|meaning|purpose|about)\b/;

interface Page extends NavOption {
  group: string;
  nameTokens: Set<string>;
  keywordTokens: Set<string>;
  groupTokens: Set<string>;
  descriptionTokens: Set<string>;
  /** Multi-word forms matched against the raw sentence, not token by token. */
  phrases: string[];
}

const normalise = (text: string): string =>
  text.toLowerCase().replace(/[^a-z0-9\s&]/g, ' ').replace(/\s+/g, ' ').trim();

/** Crude singularisation — enough to make "resumes" reach "resume". */
const singular = (token: string): string =>
  token.length > 3 && token.endsWith('s') && !token.endsWith('ss') ? token.slice(0, -1) : token;

const tokenise = (text: string): string[] =>
  normalise(text).split(' ').filter(Boolean).map(singular);

/**
 * Words left after the filler is removed.
 *
 * Stopwords are matched BEFORE singularising, because singularising first turns
 * "this" into "thi", which is in no stopword list and then scores against any
 * page unlucky enough to share the prefix.
 */
const contentOf = (text: string): string[] =>
  normalise(text).split(' ').filter(Boolean).filter(token => !STOPWORDS.has(token)).map(singular);

const tokenSet = (text: string): Set<string> => new Set(tokenise(text));

/** The route registry, flattened once into the shape matching needs. */
const PAGES: Page[] = NAV_GROUPS.flatMap(group =>
  group.items.map(item => ({
    pageId: item.id,
    label: item.name,
    description: item.description,
    group: group.label,
    nameTokens: tokenSet(item.name),
    keywordTokens: tokenSet(item.keywords.join(' ')),
    groupTokens: tokenSet(group.label),
    descriptionTokens: tokenSet(item.description),
    phrases: [item.name.toLowerCase(), ...item.keywords.filter(k => k.includes(' '))],
  })),
);

/** Every destination, for when the assistant has to show what does exist. */
export const ALL_PAGES: NavOption[] = PAGES.map(({ pageId, label, description }) => ({
  pageId, label, description,
}));

interface Scored {
  page: Page;
  score: number;
  /** Hits on the page's actual name, which is what licenses acting on a weak cue. */
  nameScore: number;
}

function score(message: string, contentTokens: string[]): Scored[] {
  const normalised = normalise(message);

  return PAGES
    .map(page => {
      let total = 0;
      let nameScore = 0;

      // Each query token counts once, at its strongest tier. A token that hits
      // the name should not also collect a point for appearing in the blurb.
      for (const token of new Set(contentTokens)) {
        if (page.nameTokens.has(token)) { total += 3; nameScore += 3; }
        else if (page.keywordTokens.has(token)) total += 2;
        else if (page.groupTokens.has(token)) total += 1;
        else if (page.descriptionTokens.has(token)) total += 0.5;
      }

      // "job description" naming the page beats two loose tokens landing apart.
      for (const phrase of page.phrases) {
        if (normalised.includes(phrase)) total += 2;
      }

      return { page, score: total, nameScore };
    })
    .sort((a, b) => b.score - a.score);
}

/** A clear win needs a real name-or-keyword hit AND daylight over the runner-up. */
const DECISIVE_SCORE = 3;
const DECISIVE_MARGIN = 1.5;

/**
 * Reads `message` as a navigation request.
 *
 * Returns null when it is not one, in which case the caller must pass the
 * message to the job-portal conversation untouched.
 */
export function resolveNavigation(message: string): NavOutcome | null {
  const trimmed = message.trim();
  if (!trimmed) return null;

  const normalised = normalise(trimmed);
  const strongCue = STRONG_NAV.test(normalised) || PLACE_NOUN.test(normalised);
  const weakCue = WEAK_NAV.test(normalised);
  const asking = QUESTION.test(normalised) || trimmed.endsWith('?');

  if (!strongCue && !weakCue && !asking) return null;

  const contentTokens = contentOf(trimmed);
  const ranked = score(trimmed, contentTokens);
  const [best, runnerUp] = ranked;

  // How much evidence it takes to claim a sentence, by how clearly it asked.
  //
  //  - "take me to…" / "the X page": claim it outright, including reporting that
  //    no such page exists — the user plainly asked to be moved.
  //  - "show/find/where X": only if a page's own NAME was said. This is what
  //    keeps "show me remote roles" a job search rather than a trip to the board.
  //  - a bare question: only if a page's FULL name appears. Otherwise "why is
  //    this candidate a 46% fit" gets claimed by Candidate Ranking on the word
  //    "candidate", and a question about a score silently becomes navigation.
  const decisive =
    best.score >= DECISIVE_SCORE && best.score - (runnerUp?.score ?? 0) >= DECISIVE_MARGIN;

  /** The sentence names a page outright ("the Model Lab"). */
  const namedInFull = PAGES.some(page => normalised.includes(page.label.toLowerCase()));
  /** The sentence names a sidebar group ("documents"), which narrows to its pages. */
  const namedGroup = NAV_GROUPS.find(group => normalised.includes(group.label.toLowerCase()));

  // Licence to claim the sentence, scaled to how plainly it asked:
  //
  //  - "show/find/where X" needs the page's own name, a group name, or a solid
  //    keyword match. "Roles" alone is not enough, which is what keeps "show me
  //    remote roles" a job search rather than a trip to the board.
  //  - a bare question needs a full page or group name, or an unmistakable
  //    match. Otherwise "why is this candidate a 46% fit" gets claimed by
  //    Candidate Ranking on the word "candidate", and a question about a score
  //    silently becomes navigation.
  //
  // A strong cue needs no licence: "take me to…" is unambiguous about intent
  // even when the destination turns out not to exist.
  if (!strongCue) {
    const licensed = weakCue
      ? best.nameScore > 0 || best.score >= DECISIVE_SCORE || Boolean(namedGroup)
      : namedInFull || Boolean(namedGroup) || (best.nameScore > 0 && decisive);
    if (!licensed) return null;
  }

  const term = contentTokens.join(' ');

  // Nothing in the application resembles what was asked for.
  if (best.score === 0 && !namedGroup) {
    return { kind: 'unknown', term, options: ALL_PAGES };
  }

  if (!decisive) {
    // Two pages fit, or one fits only loosely. Ask instead of guessing: opening
    // the wrong page is worse than one extra question, because the user cannot
    // tell a wrong guess from the app simply not having what they wanted.
    //
    // Naming a group ("take me to documents") is a narrowing, not a failure, so
    // offer exactly that group's pages rather than the global best guesses.
    const options = namedGroup && !namedInFull
      ? namedGroup.items.map(item => ({
          pageId: item.id, label: item.name, description: item.description,
        }))
      : ranked.filter(entry => entry.score > 0).slice(0, 3).map(entry => ({
          pageId: entry.page.pageId,
          label: entry.page.label,
          description: entry.page.description,
        }));
    return { kind: 'ambiguous', term, options };
  }

  const option: NavOption = {
    pageId: best.page.pageId,
    label: best.page.label,
    description: best.page.description,
  };

  // "What is the Model Lab?" wants an answer; "take me to the Model Lab" wants
  // the page. A strong navigation phrase wins even when it is phrased as a
  // question ("can you open analytics?").
  const describing = asking && !STRONG_NAV.test(normalised) && !PLACE_NOUN.test(normalised);

  return describing ? { kind: 'describe', option } : { kind: 'navigate', option };
}
