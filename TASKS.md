# Implementation Addendum - Hardening & Compliance Plan

## 🔴 P0 — Command Validation Layer
- [x] Server-side validation in `CopilotCommandValidator` (Contracts.cs)
- [x] Client-side validation in `validateCommand` (useChatHub.ts)
- [x] Strict command schema defined in types/index.ts and Contracts.cs
- [x] Navigate route allowlist enforced on both sides
- [x] Unknown keys stripped from update_filters payload
- [x] Silent rejection with logging for invalid commands
- [x] commandId deduplication (LRU cache) implemented client-side
- [x] Unit tests for malformed commands, out-of-allowlist routes, duplicate commandIds

## 🔴 P0 — Regression Guards for the Squashed Bugs
- [x] CI guard: Fail build if `Microsoft.AspNetCore.SignalR.Common` is referenced — **GitHub Actions workflow created**
- [x] Test: useChatHub double-mount (StrictMode) stays Connected
- [x] CI guard: Port consistency check between launchSettings.json and frontend/.env (csproj CheckPortConsistency + GitHub Actions)

## 🔴 P0 — Reconnect UX
- [x] Connection state exposed: connecting/connected/reconnecting/disconnected
- [x] Message queueing while reconnecting implemented
- [x] Flush queue on reconnect implemented
- [x] Connection state banners shown in ChatPage (disconnected, reconnecting, connected with queued messages)
- [x] Disable input and explain why while fully Disconnected
- [ ] Test: drop connection mid-typing → message neither lost nor duplicated

## 🟡 P1 — Skill-Verdict Caching
- [x] Cache table in SQLite: (jobId, candidateSkillSetHash, requiredSkill) → { verdict, confidence, modelVersion, promptVersion, createdAt } (PortalSkillVerdictCache entity)
- [x] Low inference temperature (~0.1-0.2) for skill transfer calls (configured in OllamaOptions)
- [x] Cache invalidation on promptVersion bump (PromptVersion field in cache, checked in SkillTransferService)
- [ ] Test: re-scoring same candidate/job hits cache (call-count spy on Ollama client)

## 🟡 P1 — Cover-Letter Groundedness Check
- [x] Extract skill/technology claims from generated letter (ExtractSkillClaims in CoverLetterService)
- [x] Diff against candidate's parsed résumé skill list (CheckGroundedness in CoverLetterService)
- [x] Ungrounded claim → regenerate with correction or strip sentence (falls back to deterministic letter)
- [x] Log groundedness failures (rate + which skills)
- [ ] Test: résumé missing skill, job requiring it → never produces letter claiming that skill

## 🟡 P1 — Fit Snapshot Versioning
- [x] Add modelVersion and promptVersion to portal_applications payload (ModelVersion, PromptVersion in PortalApplication)
- [x] Store skill-verdict cache keys used (SkillVerdictCacheKeysJson in PortalApplication)
- [ ] Test: given stored Fit Snapshot, can answer "which model, which prompt, which skill judgments"

## 🟡 P1 — Ollama Concurrency Control
- [ ] Request queue/semaphore in front of OllamaClient
- [ ] Cap concurrent heavy calls (parsing/ranking)
- [ ] Prioritize interactive chat over background jobs
- [ ] Load test: N concurrent uploads + live chat → chat latency within bound

## 🟡 P1 — Accessibility Pass on ChatWidget
- [ ] Focus trap while open; Esc closes; focus returns to trigger on close
- [ ] role="dialog" / aria-modal on panel; aria-live="polite" on streaming region
- [ ] Unread badge announced via aria-live/aria-label, not color/shape alone
- [ ] Full keyboard-only pass (open, converse, close, reopen)
- [ ] Screen reader smoke test on streaming region

## 🟡 P1 — FileDrop Edge Cases
- [ ] Multiple files dragged at once
- [ ] MIME-type and size validation with clear error state
- [ ] Paste-from-clipboard support
- [ ] Test matrix: single valid, multiple, oversized, wrong MIME, paste event

## 🟢 P2 — Auto-Apply Safety Rails
- [x] Mandatory review-and-confirm: PreviewAsync does all work, SubmitAsync only persists what came back
- [ ] Idempotency key on Hangfire job: generate at POST /api/apply, persist, check-and-set
- [ ] Terminal status enum (Submitted, NeedsReview, Blocked, Failed) per attempt, pushed via SignalR
- [ ] Blocked-session fallback: CAPTCHA/bot-check → "here's your form, pre-filled — finish it yourself"
- [ ] Legal review of ToS exposure (Workday, Greenhouse, iCIMS prohibit automated submissions)
- [ ] Per-target-ATS evaluation: official API vs Playwright scraping

## ⚪ P3 — Monitor, No Immediate Build
- [ ] SQLite vector store write contention monitoring
- [ ] Document intentional split: bge-m3/Elasticsearch (recruiter) vs nomic-embed-text/SQLite (Job Portal)

---

# NEW: Immediate Next Steps (from follow-up verification)

## 🔴 P0 — SignalR CI Guard Workflow
- [x] Create `.github/workflows/ci.yml` with grep-based check for `Microsoft.AspNetCore.SignalR.Common` in csproj

## 🔴 P0 — Reconnect UX Completion
- [x] Disable input in ChatPage when state === 'disconnected' with explanatory text
- [ ] Add integration test for message not lost/duplicated on reconnect

## 🟡 P1 — Missing Tests for Completed Work
- [ ] Skill-verdict cache: call-count spy test
- [ ] Groundedness check: résumé missing skill test
- [ ] Fit Snapshot versioning: reconstruction test

## 🟡 P1 — Not Started Items
- [ ] Ollama Concurrency Control
- [ ] Accessibility Pass on ChatWidget
- [ ] FileDrop Edge Cases