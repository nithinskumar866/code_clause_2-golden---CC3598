/**
 * What the assistant needs in order to move around the application.
 *
 * Declared here, on the consumer's side, and implemented by the host
 * (`components/layout/navIntent.ts`). The popup therefore knows that "some
 * page" can be opened without knowing which pages exist — the route registry
 * stays the host's business, and this file stays free of page ids.
 */

export interface NavOption {
  /** Opaque to the popup; handed straight back to `go`. */
  pageId: string;
  label: string;
  description?: string;
}

/**
 * The outcome of reading a message as a request to move around the app.
 *
 * `null` from `resolve` means "not a navigation request" — the message belongs
 * to the job-portal conversation and goes to the hub untouched. Every other
 * outcome is answered locally and never reaches the .NET API, so navigating by
 * chat keeps working when the portal API is down.
 */
export type NavOutcome =
  /** One page clearly matched. Open it. */
  | { kind: 'navigate'; option: NavOption }
  /** A question about a section rather than a request to open it. */
  | { kind: 'describe'; option: NavOption }
  /** Several pages fit, or one fits weakly. Ask rather than guess. */
  | { kind: 'ambiguous'; term: string; options: NavOption[] }
  /** Nothing in the application matches. Say so; never invent a page. */
  | { kind: 'unknown'; term: string; options: NavOption[] };

export interface AssistantNavigator {
  /** Reads a message as a navigation request, or returns null if it is not one. */
  resolve: (message: string) => NavOutcome | null;
  /** Opens a page by the id carried on a NavOption. */
  go: (pageId: string) => void;
}
