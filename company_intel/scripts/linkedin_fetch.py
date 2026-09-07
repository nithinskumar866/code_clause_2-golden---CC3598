"""
Fetch everything the LinkedIn API will actually give us, and report what it refuses.

    python scripts/linkedin_fetch.py --client-id <id> --client-secret <secret>
    python scripts/linkedin_fetch.py --vanity techwaukee --vanity gorecruitaitechnologies

This is an API client, not a scraper. It signs a real member in, uses the token they
grant, and asks LinkedIn's documented endpoints. Everything it cannot get, it says so
and says why — the point is an honest inventory rather than a partial success that
hides a permission problem.

Two facts shape what comes back, and no configuration changes either:

  * Every LinkedIn permission is 3-legged: a member signs in and consents. There is no
    app-only token that returns data about other people or other companies.
  * `rw_organization_admin` is role-gated on top of that — restricted to organizations
    where the signed-in member holds the ADMINISTRATOR role. For a Page you administer
    you get the full record; for any other Page, a handful of public fields.

Before this works the app must be ENABLED (a super admin of the associated LinkedIn
Page verifies it in the Developer Portal) and `http://localhost:8765/callback` must be
listed as an authorized redirect URL on the app's Auth tab.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import secrets
import sys
import threading
import urllib.parse
import webbrowser

import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AUTHORIZE = "https://www.linkedin.com/oauth/v2/authorization"
TOKEN = "https://www.linkedin.com/oauth/v2/accessToken"
API = "https://api.linkedin.com"
REDIRECT = "http://localhost:8765/callback"
API_VERSION = "202608"

# Requested in descending order of usefulness. LinkedIn rejects the whole request if the
# app is not approved for a scope, so the run retries with a smaller set rather than
# failing outright — that way a bare sign-in still tells us something.
SCOPE_SETS = [
    ("full", "openid profile email r_organization_admin rw_organization_admin r_organization_social"),
    ("org-admin", "openid profile email rw_organization_admin"),
    ("sign-in only", "openid profile email"),
]

_code_holder: dict = {}


class _Callback(http.server.BaseHTTPRequestHandler):
    """Catches the one redirect LinkedIn makes back to us after consent."""

    def do_GET(self):  # noqa: N802
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        _code_holder.update({k: v[0] for k, v in params.items()})
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        done = "code" in _code_holder
        self.wfile.write(
            (
                "<h2>You can close this tab.</h2><p>Returning to the terminal.</p>"
                if done
                else f"<h2>LinkedIn returned an error.</h2><pre>{query}</pre>"
            ).encode("utf-8")
        )

    def log_message(self, *args):  # silence the default request logging
        return


def authorize(client_id: str, scope: str) -> str:
    """Open the consent screen and wait for the redirect. Returns the auth code."""
    state = secrets.token_urlsafe(16)
    url = f"{AUTHORIZE}?" + urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": REDIRECT,
            "state": state,
            "scope": scope,
        }
    )
    server = http.server.HTTPServer(("localhost", 8765), _Callback)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    print(f"\n  Opening the consent screen for scopes: {scope}")
    print(f"  If no browser opens, paste this:\n    {url}\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass

    thread.join(timeout=300)
    server.server_close()

    if "error" in _code_holder:
        raise RuntimeError(
            f"{_code_holder.get('error')}: {_code_holder.get('error_description', '')}"
        )
    code = _code_holder.get("code")
    if not code:
        raise RuntimeError("No authorization code came back within 5 minutes.")
    if _code_holder.get("state") != state:
        raise RuntimeError("State mismatch — the redirect did not come from our request.")
    return code


def exchange(client_id: str, client_secret: str, code: str) -> dict:
    r = httpx.post(
        TOKEN,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT,
            "client_id": client_id,
            "client_secret": client_secret,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def call(client: httpx.Client, path: str, label: str) -> dict | None:
    """
    One API call, reported honestly.

    A 403 here is not a bug — it is LinkedIn stating that the signed-in member does not
    administer the thing being asked about. Printing that plainly is the whole purpose
    of this script.
    """
    r = client.get(API + path, timeout=30)
    print(f"\n  {label}\n    GET {path}\n    HTTP {r.status_code}")
    if r.status_code == 200:
        try:
            return r.json()
        except Exception:
            print(f"    (unparseable body: {r.text[:200]})")
            return None
    if r.status_code == 403:
        print("    Refused: the signed-in member is not an ADMINISTRATOR of this Page,")
        print("    or the app lacks the product that grants this scope.")
    print(f"    {r.text[:300]}")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch what LinkedIn's API will give us.")
    parser.add_argument("--client-id", default=os.getenv("LINKEDIN_CLIENT_ID", ""))
    parser.add_argument("--client-secret", default=os.getenv("LINKEDIN_CLIENT_SECRET", ""))
    parser.add_argument(
        "--vanity",
        action="append",
        default=[],
        help="Page vanity name from its URL, e.g. linkedin.com/company/<vanity>/. Repeatable.",
    )
    args = parser.parse_args()

    if not args.client_id or not args.client_secret:
        print("Need --client-id and --client-secret (or LINKEDIN_CLIENT_ID / _SECRET).")
        return 2

    print("=" * 76)
    print("LINKEDIN API — WHAT WE CAN ACTUALLY GET")
    print("=" * 76)

    # Cheap pre-flight. A disabled app cannot mint any token, so there is no point
    # sending the user to a consent screen that will fail on the exchange.
    probe = httpx.post(
        TOKEN,
        data={
            "grant_type": "client_credentials",
            "client_id": args.client_id,
            "client_secret": args.client_secret,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    if probe.status_code == 401 and "disabled_client" in probe.text:
        print("\n  The app is DISABLED — LinkedIn will not issue any token.")
        print("  A super admin of the associated LinkedIn Page must verify it:")
        print("    Developer Portal -> My Apps -> your app -> Settings -> verify the Page")
        print("\n  Nothing can be fetched until then. Re-run this once it is enabled.")
        return 1

    token_payload = None
    granted = ""
    for label, scope in SCOPE_SETS:
        try:
            code = authorize(args.client_id, scope)
            token_payload = exchange(args.client_id, args.client_secret, code)
            granted = scope
            print(f"\n  Token obtained with the '{label}' scope set.")
            break
        except Exception as e:
            print(f"\n  '{label}' scope set failed: {e}")
            _code_holder.clear()

    if not token_payload:
        print("\n  No token could be obtained with any scope set.")
        return 1

    token = token_payload["access_token"]
    print(f"  Scopes actually granted: {token_payload.get('scope', granted)}")

    client = httpx.Client(
        headers={
            "Authorization": f"Bearer {token}",
            "X-Restli-Protocol-Version": "2.0.0",
            "LinkedIn-Version": API_VERSION,
        }
    )

    print("\n" + "=" * 76)
    print("THE SIGNED-IN MEMBER (always available with openid/profile/email)")
    print("=" * 76)
    me = call(client, "/v2/userinfo", "Your own profile")
    if me:
        for key in ("sub", "name", "given_name", "family_name", "email", "email_verified", "locale", "picture"):
            if key in me:
                value = str(me[key])
                print(f"      {key:<16} {value[:90]}")

    print("\n" + "=" * 76)
    print("PAGES YOU ADMINISTER")
    print("=" * 76)
    acls = call(
        client,
        "/rest/organizationAcls?q=roleAssignee&role=ADMINISTRATOR&state=APPROVED",
        "Which Pages does this member administer?",
    )
    admin_ids = []
    if acls:
        for element in acls.get("elements", []):
            urn = element.get("organization", "")
            if urn.rsplit(":", 1)[-1].isdigit():
                admin_ids.append(urn.rsplit(":", 1)[-1])
        print(f"      administered organization ids: {admin_ids or 'none'}")

    for org_id in admin_ids:
        full = call(client, f"/rest/organizations/{org_id}", f"FULL record for organization {org_id}")
        if full:
            print(json.dumps(full, indent=6)[:2500])

    for vanity in args.vanity:
        print("\n" + "=" * 76)
        print(f"PAGE BY VANITY NAME: {vanity}")
        print("=" * 76)
        found = call(
            client,
            f"/rest/organizations?q=vanityName&vanityName={urllib.parse.quote(vanity)}",
            "Public (non-admin) fields — works for any Page",
        )
        if found:
            print(json.dumps(found, indent=6)[:1800])

    print("\n" + "=" * 76)
    print("WHAT IS STRUCTURALLY UNAVAILABLE")
    print("=" * 76)
    print("""
  * Any individual's profile by URL. There is no endpoint for it at any tier — only
    the member who signs in, via /v2/userinfo above.
  * Full details of a Page you do not administer. `rw_organization_admin` is
    restricted to organizations where the signed-in member is an ADMINISTRATOR, so a
    third-party Page returns name, vanityName, website, logo, locations and nothing
    more, however many products are approved.
  * The Company Intelligence API, which came closest to third-party company data, is
    documented as not accepting new applications.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
