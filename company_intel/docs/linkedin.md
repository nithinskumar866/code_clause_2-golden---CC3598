# LinkedIn: how officials get into the corpus

This project is given two kinds of URL per company: a website, and the LinkedIn
profiles of its officials. The website half is built. This note is the design for the
other half, and the reasoning behind not building it the obvious way.

## Why the obvious way does not work

The obvious design puts a "LinkedIn extractor" beside the website crawler, feeding the
same pipeline. Built as drawn, it returns nothing:

- **Profiles are authwalled.** An unauthenticated GET of `linkedin.com/in/<person>`
  returns a login interstitial, not a profile. There is no public HTML to extract.
- **Automated access is actively blocked.** LinkedIn runs bot detection and rate-limits
  or bans the IPs behind it. A crawler that trips it can take the whole service's
  outbound address with it.
- **It is prohibited by their terms.** Automated collection is disallowed regardless of
  how it is performed.

The consequence for architecture: **LinkedIn must not be load-bearing.** If the answer
to "who is the CEO?" depends on a source we cannot reliably fetch, the feature is one
policy change away from silently returning nothing. So the schema keeps the slot, and
the *fetcher* behind it is pluggable.

## The seam

One interface, several implementations:

```python
PersonSource.fetch(profile_url, company_id) -> PersonRecord | None
```

`PersonRecord` is `{name, role, company_id, summary, profile_url, source_type}`. Whatever
produces it, the record lands in `ci_content` as a point with its own vector, exactly
like a page chunk, so a *descriptive* question — "who leads engineering at a logistics
company?" — is answerable by meaning and not only by typing a name correctly. The
payload's `source_type` records where it came from, so an answer can say so.

Nothing downstream changes: retrieval, citation, guardrails and refresh already work on
whatever is in the content collection.

## The four implementations, in adoption order

**1. `SiteLeadershipSource` — free, permitted, and already half-built.**
The crawler classifies pages as `leadership` today (`/leadership`, `/our-team`,
`/management`, `/board`, …). Those pages name the same officials, with the same titles,
publicly, on a domain we are already crawling. What is left to build is a person-block
extractor: split a leadership page into name / role / bio triples rather than treating
it as prose. This is the highest-value piece of work remaining, and it covers most of
what a recruiter actually asks.

**2. `ManualSource` — the reliable floor.**
A `POST /companies/{id}/people` endpoint (or a CSV) taking name, role, LinkedIn URL and
a short bio. The registry **already stores `linkedin_urls` and never fetches them** —
they are surfaced as links for a person to open in their own browser, which is exactly
what LinkedIn permits. This is the honest answer to "we have the LinkedIn URLs of the
officials": show them, do not scrape them.

**3. `ProviderSource` — licensed data, if depth is genuinely needed.**
Commercial providers (Proxycurl, People Data Labs, Coresignal, and others) sell profile
lookups by URL and return structured JSON, which maps onto `PersonRecord` with no HTML
parsing at all. Costs cents per profile and moves the compliance question onto the
provider's licence — their terms differ, and are worth reading before committing.
LinkedIn's own APIs (Talent Solutions, Marketing) require partner approval and do not
offer arbitrary profile lookup, so they do not fit this use case.

**4. Authenticated browser automation — not recommended.**
Playwright driving a logged-in session is technically achievable and a bad idea: it
violates LinkedIn's terms directly, gets accounts permanently banned, and puts the
product's data supply on a foundation that can disappear without warning. Listed here
because it will be suggested, not because it should be done.

## Refresh

Profiles change rarely. A 90-day cadence, well above the site cadences in
`classify.refresh_days`. Because provider calls cost money, the same discipline applies
as for pages: hash the returned record and skip the write when nothing changed.

## Degradation

With no provider configured, person questions are answered from leadership pages and
manual entries, and the LinkedIn URL is returned as a link alongside the answer. Adding
a provider later is one new `PersonSource` implementation and one config key. Nothing in
the store, the retrieval path or the API changes.
