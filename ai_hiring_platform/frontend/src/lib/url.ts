/**
 * Update the browser URL's query string in place without a router.
 * Reads the current search params, applies `mutate`, and writes the result
 * back. Preserves any params the caller doesn't touch (e.g. the `view` param
 * used for the active page and the history filter params can coexist).
 */
export function updateSearchParams(
  mutate: (params: URLSearchParams) => void,
  mode: 'push' | 'replace' = 'replace',
): void {
  const params = new URLSearchParams(window.location.search);
  mutate(params);
  const qs = params.toString();
  const url = `${window.location.pathname}${qs ? `?${qs}` : ''}`;
  if (mode === 'push') {
    window.history.pushState(null, '', url);
  } else {
    window.history.replaceState(null, '', url);
  }
}
