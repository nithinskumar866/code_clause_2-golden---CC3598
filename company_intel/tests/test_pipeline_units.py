"""
Unit tests for the deterministic half of the pipeline.

Everything here runs without Qdrant, without a network and without the embedding model,
which is what makes it the suite you can run on every change. The parts that need those
are covered separately in `test_ingest_flow.py` with fakes.
"""
from __future__ import annotations

import pytest

from app.crawl import urls as urlutil
from app.crawl.frontier import Frontier
from app.extract import classify, html_text
from app.process import chunker
from app.process.hashing import content_hash, url_hash


# --- URL normalisation ------------------------------------------------------
class TestNormalize:
    def test_strips_fragment_and_tracking_params(self):
        assert urlutil.normalize("https://acme.com/about?utm_source=x&id=7#team") == (
            "https://acme.com/about?id=7"
        )

    def test_collapses_variants_to_one_url(self):
        """The whole point: these are one page and must produce one key."""
        variants = [
            "https://acme.com/about/",
            "https://acme.com/about",
            "https://acme.com/about#top",
            "https://acme.com/about?utm_campaign=spring",
            "HTTPS://ACME.COM/about",
        ]
        assert len({urlutil.normalize(v) for v in variants}) == 1

    def test_sorts_query_params(self):
        assert urlutil.normalize("https://a.com/p?b=2&a=1") == urlutil.normalize(
            "https://a.com/p?a=1&b=2"
        )

    def test_rejects_non_pages(self):
        for bad in ("mailto:x@y.com", "javascript:void(0)", "tel:+1", "", "#anchor",
                    "https://acme.com/brochure.pdf", "https://acme.com/logo.png"):
            assert urlutil.normalize(bad) is None

    def test_resolves_relative_against_base(self):
        assert urlutil.normalize("/careers", base="https://acme.com/about") == (
            "https://acme.com/careers"
        )

    def test_drops_default_port(self):
        assert urlutil.normalize("https://acme.com:443/x") == "https://acme.com/x"


class TestScope:
    def test_registrable_domain_handles_two_part_suffixes(self):
        assert urlutil.registrable_domain("careers.bbc.co.uk") == "bbc.co.uk"
        assert urlutil.registrable_domain("www.acme.com") == "acme.com"

    def test_subdomains_are_in_scope(self):
        assert urlutil.same_site("https://careers.acme.com/jobs", "acme.com")

    def test_other_domains_are_not(self):
        assert not urlutil.same_site("https://twitter.com/acme", "acme.com")


# --- hashing ----------------------------------------------------------------
class TestHashing:
    def test_whitespace_and_case_do_not_change_the_hash(self):
        """Cosmetic reflow must not read as a content change, or refresh never skips."""
        assert content_hash("We  build\n\nsoftware.") == content_hash("we build software.")

    def test_real_edits_do_change_it(self):
        assert content_hash("We build software.") != content_hash("We build hardware.")

    def test_url_hash_is_stable_and_short(self):
        assert url_hash("https://acme.com") == url_hash("https://acme.com")
        assert len(url_hash("https://acme.com")) == 32


# --- classification ---------------------------------------------------------
class TestClassify:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://acme.com/", "home"),
            ("https://acme.com/about-us", "about"),
            ("https://acme.com/about/leadership", "leadership"),
            ("https://acme.com/our-team", "leadership"),
            ("https://acme.com/products/crm", "products"),
            ("https://acme.com/solutions", "services"),
            ("https://acme.com/careers", "careers"),
            ("https://acme.com/blog/2024/hello", "news"),
            ("https://acme.com/case-studies/x", "clients"),
            ("https://acme.com/contact", "contact"),
            ("https://acme.com/xyz", "other"),
        ],
    )
    def test_from_path(self, url, expected):
        assert classify.classify(url) == expected

    def test_leadership_wins_over_about(self):
        """Rule order matters: /about/leadership is a leadership page, not an about page."""
        assert classify.classify("https://acme.com/about/leadership") == "leadership"

    def test_title_is_the_fallback_when_the_path_says_nothing(self):
        assert classify.classify("https://acme.com/p/42", title="Our Leadership Team") == (
            "leadership"
        )

    def test_news_refreshes_faster_than_leadership(self):
        assert classify.refresh_days("news") < classify.refresh_days("leadership")

    def test_important_pages_crawl_before_the_blog(self):
        assert classify.crawl_priority("about") < classify.crawl_priority("news")


# --- extraction -------------------------------------------------------------
SAMPLE_HTML = """
<html lang="en"><head><title>Acme — Products</title></head>
<body>
  <nav><a href="/">Home</a><a href="/careers">Careers</a></nav>
  <div class="cookie-banner"><h2>We use cookies</h2><p>Accept all cookies to continue.</p></div>
  <main>
    <h1>Our Products</h1>
    <p>Acme builds tools for logistics teams across three continents.</p>
    <h2>Acme CRM</h2>
    <p>Acme CRM tracks customers, shipments and renewals in one shared workspace.</p>
    <h2>Acme Analytics</h2>
    <p>Acme Analytics reports on fleet utilisation and delivery performance in real time.</p>
    <a href="/products/crm">CRM details</a>
    <a href="https://twitter.com/acme">Twitter</a>
  </main>
  <footer><p>Copyright Acme 2026. Privacy Policy.</p></footer>
</body></html>
"""


class TestExtract:
    def setup_method(self):
        self.page = html_text.extract(SAMPLE_HTML, "https://acme.com/products")

    def test_reads_title_h1_and_lang(self):
        assert self.page.title == "Acme — Products"
        assert self.page.h1 == "Our Products"
        assert self.page.lang == "en"

    def test_keeps_the_real_content(self):
        assert "logistics teams" in self.page.text
        assert "Acme CRM tracks customers" in self.page.text

    def test_drops_the_cookie_banner(self):
        """Boilerplate that repeats site-wide is what poisons a corpus."""
        assert "Accept all cookies" not in " ".join(s.text for s in self.page.sections)

    def test_sections_carry_their_heading_path(self):
        paths = [s.heading_path for s in self.page.sections]
        assert any("Acme CRM" in p for p in paths)

    def test_links_are_normalised_and_include_offsite(self):
        """Scope is the frontier's job — extraction reports every link it found."""
        assert "https://acme.com/products/crm" in self.page.links
        assert "https://twitter.com/acme" in self.page.links

    def test_empty_html_is_empty_not_an_error(self):
        assert html_text.extract("", "https://acme.com").empty


# --- chunking ---------------------------------------------------------------
class TestChunker:
    def test_heading_path_is_embedded_with_the_prose(self):
        page = html_text.extract(SAMPLE_HTML, "https://acme.com/products")
        chunks = chunker.chunk_page(page)
        assert chunks
        assert any("Acme CRM" in c.text for c in chunks)

    def test_body_excludes_the_heading_so_citations_read_cleanly(self):
        sections = [html_text.Section(heading_path="Products > CRM", text="A" * 300)]
        chunk = chunker.chunk_sections(sections)[0]
        assert chunk.text.startswith("Products > CRM")
        assert not chunk.body.startswith("Products > CRM")

    def test_long_sections_are_split(self):
        long_text = " ".join(f"Sentence number {i} about logistics." for i in range(400))
        chunks = chunker.chunk_sections([html_text.Section("Big", long_text)])
        assert len(chunks) > 1

    def test_short_sections_are_merged_not_emitted_alone(self):
        sections = [
            html_text.Section("A", "Tiny one."),
            html_text.Section("B", "Tiny two."),
            html_text.Section("C", "Tiny three."),
        ]
        assert len(chunker.chunk_sections(sections)) < 3

    def test_indexes_are_contiguous(self):
        page = html_text.extract(SAMPLE_HTML, "https://acme.com/products")
        chunks = chunker.chunk_page(page)
        assert [c.index for c in chunks] == list(range(len(chunks)))


# --- frontier ---------------------------------------------------------------
class TestFrontier:
    def _frontier(self, **kwargs):
        kwargs.setdefault("domain", "acme.com")
        return Frontier(**kwargs)

    def test_rejects_offsite_links(self, monkeypatch):
        monkeypatch.setattr("app.crawl.frontier.robots.allowed", lambda url: True)
        f = self._frontier()
        assert not f.offer("https://twitter.com/acme", depth=1)
        assert "Off-site" in list(f.skipped.values())[0]

    def test_deduplicates_url_variants(self, monkeypatch):
        monkeypatch.setattr("app.crawl.frontier.robots.allowed", lambda url: True)
        f = self._frontier()
        assert f.offer("https://acme.com/about", depth=1)
        assert not f.offer("https://acme.com/about/#team", depth=1)

    def test_important_pages_come_out_first(self, monkeypatch):
        """A capped crawl must spend its budget on About before the blog."""
        monkeypatch.setattr("app.crawl.frontier.robots.allowed", lambda url: True)
        f = self._frontier()
        f.offer("https://acme.com/blog/post-1", depth=1)
        f.offer("https://acme.com/about", depth=1)
        f.offer("https://acme.com/products", depth=1)
        assert [f.next().url for _ in range(3)] == [
            "https://acme.com/about",
            "https://acme.com/products",
            "https://acme.com/blog/post-1",
        ]

    def test_page_cap_stops_emission_and_is_reported(self, monkeypatch):
        monkeypatch.setattr("app.crawl.frontier.robots.allowed", lambda url: True)
        f = self._frontier(max_pages=2)
        for i in range(5):
            f.offer(f"https://acme.com/p{i}", depth=1)
        assert len([1 for _ in iter(f.next, None)]) == 2
        assert f.budget_exhausted is True

    def test_depth_limit(self, monkeypatch):
        monkeypatch.setattr("app.crawl.frontier.robots.allowed", lambda url: True)
        f = self._frontier(max_depth=1)
        assert not f.offer("https://acme.com/deep", depth=2)

    def test_deny_patterns_win_over_allow(self, monkeypatch):
        monkeypatch.setattr("app.crawl.frontier.robots.allowed", lambda url: True)
        f = self._frontier(allow_patterns=["/products"], deny_patterns=["/products/legacy"])
        assert f.offer("https://acme.com/products/new", depth=1)
        assert not f.offer("https://acme.com/products/legacy", depth=1)

    def test_allow_patterns_exclude_everything_else(self, monkeypatch):
        monkeypatch.setattr("app.crawl.frontier.robots.allowed", lambda url: True)
        f = self._frontier(allow_patterns=["/products"])
        assert not f.offer("https://acme.com/careers", depth=1)


class TestHostCanonicalisation:
    """Both bugs the first live crawl exposed, pinned as tests."""

    def test_www_and_apex_are_one_url(self):
        assert urlutil.normalize("https://www.acme.com/about") == "https://acme.com/about"

    def test_domain_root_has_no_trailing_slash(self):
        assert urlutil.normalize("https://www.python.org/") == "https://python.org"

    def test_www_homepage_is_not_crawled_twice(self, monkeypatch):
        monkeypatch.setattr("app.crawl.frontier.robots.allowed", lambda url: True)
        f = Frontier(domain="acme.com")
        assert f.offer("https://acme.com", depth=0)
        assert not f.offer("https://www.acme.com/", depth=1)

    def test_mark_seen_blocks_a_redirect_target(self, monkeypatch):
        monkeypatch.setattr("app.crawl.frontier.robots.allowed", lambda url: True)
        f = Frontier(domain="acme.com")
        f.mark_seen("https://acme.com/landing/")
        assert not f.offer("https://acme.com/landing", depth=1)


class TestLiveCrawlRegressions:
    """Three bugs a real crawl of basecamp.com and zapier.com exposed."""

    def test_text_inside_a_chrome_container_is_dropped(self):
        """
        The paragraph carries no class of its own — the banner around it does.

        Checking only the node itself put "We'd like to use cookies to help understand
        if our ads are working" into the live corpus, where it was then returned as
        evidence about the company.
        """
        html = (
            "<html><body><main><h1>About</h1>"
            "<div class='cookie-consent-banner'><p>We would like to use cookies to help "
            "understand if our advertising is working or not. Accept all to continue.</p></div>"
            "<p>Basecamp is a project management tool used by small teams everywhere.</p>"
            "</main></body></html>"
        )
        text = " ".join(s.text for s in html_text.extract(html, "https://x.com").sections)
        assert "cookies" not in text.lower()
        assert "project management tool" in text

    def test_deeply_nested_chrome_is_still_caught(self):
        html = (
            "<html><body><main><h1>About</h1>"
            "<div class='newsletter-signup'><div><div><p>Subscribe to our newsletter for "
            "weekly updates and product announcements from the team.</p></div></div></div>"
            "<p>Zapier connects the apps that businesses already use every day.</p>"
            "</main></body></html>"
        )
        text = " ".join(s.text for s in html_text.extract(html, "https://x.com").sections)
        assert "newsletter" not in text.lower()
        assert "connects the apps" in text

    def test_chunks_stay_inside_the_embedding_models_input_limit(self):
        """
        A chunk longer than the model's 512-token window is truncated when embedded, so
        its tail never reaches the vector — while the full text is still stored and
        cited. The chunk then claims to contain text the vector has never seen. Measured
        at 14% of a live corpus before the target was lowered.
        """
        from app.core.config import settings

        long_text = " ".join(f"Sentence {i} about logistics software." for i in range(1200))
        chunks = chunker.chunk_sections([html_text.Section("Big", long_text)])
        assert chunks
        assert max(len(c.text) for c in chunks) <= settings.CHUNK_TARGET_CHARS
        # ~4.6 characters per token on this corpus, so the budget must clear 512 tokens.
        assert settings.CHUNK_TARGET_CHARS < 2350

    def test_the_retrieval_floor_is_calibrated_per_model(self):
        """
        One floor for all models is a trap. Measured on the same live corpus, off-topic
        questions top out at 0.43 with bge but 0.553 with nomic — so bge's correct 0.55
        floor lets "who won the 2018 world cup" through on nomic.
        """
        from app.core.config import Settings

        bge = Settings(EMBED_PROVIDER="fastembed", EMBED_MODEL="BAAI/bge-small-en-v1.5",
                       EMBED_MIN_SIMILARITY=0.0)
        nomic = Settings(EMBED_PROVIDER="ollama", EMBED_OLLAMA_MODEL="nomic-embed-text",
                         EMBED_MIN_SIMILARITY=0.0)
        assert bge.min_similarity == 0.55
        assert nomic.min_similarity == 0.62
        assert nomic.min_similarity > 0.553  # the highest off-topic score measured

    def test_an_explicit_floor_overrides_the_calibrated_one(self):
        from app.core.config import Settings

        assert Settings(EMBED_MIN_SIMILARITY=0.71).min_similarity == 0.71

    def test_an_unknown_model_gets_the_strictest_floor_not_a_guess(self):
        from app.core.config import Settings

        s = Settings(EMBED_PROVIDER="ollama", EMBED_OLLAMA_MODEL="some-new-model",
                     EMBED_MIN_SIMILARITY=0.0)
        assert s.min_similarity == 0.62

    def test_aria_hidden_content_is_dropped(self):
        """
        Basecamp's real cookie banner, reduced: `class="tracking tracking--hidden"`
        with `aria-hidden="true"`. No 'cookie' or 'consent' in the markup, so the hint
        list could not catch it — the page's own visibility declaration does.
        """
        html = (
            "<html><body><main><h1>Testimonials</h1>"
            "<div class='tracking tracking--hidden' aria-hidden='true'><div><p>We would "
            "like to use cookies to help understand if our ads are working or not.</p></div></div>"
            "<p>Service is not an afterthought, it is one of Basecamp's best features.</p>"
            "</main></body></html>"
        )
        text = " ".join(s.text for s in html_text.extract(html, "https://x.com").sections)
        assert "cookies" not in text.lower()
        assert "afterthought" in text

    def test_display_none_content_is_dropped(self):
        html = (
            "<html><body><main><h1>About</h1>"
            "<div style='display: none'><p>Hidden marketing modal that no visitor ever "
            "sees on this page at all.</p></div>"
            "<p>Acme builds freight routing software for European carriers.</p>"
            "</main></body></html>"
        )
        text = " ".join(s.text for s in html_text.extract(html, "https://x.com").sections)
        assert "Hidden marketing" not in text
        assert "freight routing" in text

    def test_a_real_word_in_the_content_is_not_treated_as_chrome(self):
        """
        The reason the fix is an accessibility attribute and not another keyword:
        'tracking' is ordinary vocabulary for a logistics company.
        """
        html = (
            "<html><body><main><h1>Products</h1>"
            "<p>Acme shipment tracking shows every parcel in transit across the network.</p>"
            "</main></body></html>"
        )
        text = " ".join(s.text for s in html_text.extract(html, "https://x.com").sections)
        assert "shipment tracking" in text


class TestGpuEraRegressions:
    """Bugs a live crawl of Indian IT company sites exposed."""

    def test_locale_variants_collapse_to_one_page(self):
        """
        wipro.com spent eleven of its twelve crawled pages on ONE careers page in
        eleven languages, each stored separately and competing at retrieval.
        """
        variants = [f"https://careers.wipro.com?locale={l}" for l in
                    ("en_US", "fr_CA", "ja_JP", "de_DE", "ar_SA")]
        assert len({urlutil.normalize(v) for v in variants}) == 1

    def test_a_section_subdomain_names_the_page_type(self):
        """The root of careers.acme.com is a careers page, not a homepage."""
        assert classify.classify("https://careers.acme.com") == "careers"
        assert classify.classify("https://careers.acme.com/openings") == "careers"
        assert classify.classify("https://newsroom.acme.com") == "news"
        assert classify.classify("https://www.acme.com") == "home"

    def test_a_bot_wall_is_not_content(self):
        """
        The most damaging crawl failure is a 200 OK. techmahindra.com returned this on
        every page, and 21 chunks of it were indexed and made citable as company
        information before this check existed.
        """
        from app.extract.blocked import looks_blocked

        assert looks_blocked(
            "JavaScript is disabled\nIn order to continue, we need to verify that "
            "you're not a robot. This requires JavaScript."
        )
        assert looks_blocked("Checking your browser before accessing the site.")

    def test_a_long_page_about_bot_detection_is_still_content(self):
        """The check must not delete a real page for discussing the subject."""
        from app.extract.blocked import looks_blocked

        article = (
            "We build fraud detection systems for banks. " * 60
            + " Our platform can verify you are not a robot without a challenge page."
        )
        assert not looks_blocked(article)

    def test_tracking_params_are_stripped_from_a_redirect_target(self):
        """A citation must not carry junk a redirect chain attached to it."""
        assert urlutil.normalize("https://www.mphasis.com/home.html?utm_source=chatgpt.com") == (
            "https://mphasis.com/home.html"
        )

    def test_a_redirect_to_another_domain_widens_the_scope(self):
        """
        ltimindtree.com redirects to ltm.com. Without adopting the landed domain the
        crawl reads one page and rejects every link on it as off-site — observed live,
        one page indexed out of twelve requested.
        """
        f = Frontier(domain="ltimindtree.com")
        assert not f.offer("https://www.ltm.com/about", depth=1)
        assert f.adopt_domain_of("https://www.ltm.com/")
        f2 = Frontier(domain="ltimindtree.com")
        f2.adopt_domain_of("https://www.ltm.com/")
        assert f2.offer("https://www.ltm.com/about", depth=1)

    def test_adopting_is_idempotent_and_ignores_the_same_domain(self):
        f = Frontier(domain="acme.com")
        assert f.adopt_domain_of("https://www.acme.com/x") is False
        assert f.adopt_domain_of("https://acme-global.com/") is True
        assert f.adopt_domain_of("https://acme-global.com/") is False


class TestRenderingGuards:
    """
    Rendering is opt-in and must never make a corpus worse.

    Measured against three real Indian IT sites, a headless browser added 2, 0 and 0
    new in-scope links over a plain fetch, and on ltm.com returned 426 characters where
    a plain fetch returned 2217. These tests pin the guards that make that survivable.
    """

    def test_a_short_render_falls_back_to_the_plain_fetch(self, monkeypatch):
        from app.crawl.fetcher import FetchResult
        from app.core.config import settings

        rich = (
            "<html><head><title>Acme</title></head><body><main><h1>Acme</h1><p>"
            + "Acme builds logistics software for European freight carriers. " * 30
            + "</p></main></body></html>"
        )
        shell = "<html><head><title>Acme</title></head><body><main><p>Loading…</p></main></body></html>"

        monkeypatch.setattr(
            "app.ingest.pipeline.browser.fetch",
            lambda url: FetchResult(url=url, status="ok", http_status=200, html=shell),
        )
        monkeypatch.setattr(
            "app.ingest.pipeline.fetcher.fetch",
            lambda url, etag=None, last_modified=None: FetchResult(
                url=url, status="ok", http_status=200, html=rich
            ),
        )
        from app.extract import html_text

        assert len(html_text.extract(shell, "https://acme.com").text) < settings.BROWSER_MIN_TEXT_CHARS
        assert len(html_text.extract(rich, "https://acme.com").text) > settings.BROWSER_MIN_TEXT_CHARS

    def test_render_is_off_by_default(self, store):
        """Every company pays the plain-fetch cost unless someone opts it in."""
        from app.sources import registry

        assert registry.register(name="Acme", domain="acme.com")["render"] is False

    def test_render_can_be_opted_into_per_company(self, store):
        from app.sources import registry

        assert registry.register(name="Acme", domain="acme.com", render=True)["render"] is True

    def test_an_empty_render_is_a_failure_not_an_empty_page(self):
        """So the caller falls back rather than storing a blank."""
        from app.crawl.fetcher import FetchResult

        blank = FetchResult(url="https://acme.com", status="failed", error="empty document")
        assert not blank.ok
