# Session Log

Running record of what each session accomplished, so context survives across cleared sessions. Newest first. Keep entries short: date · who (backend/frontend/qa/orchestrator) · what changed · verify result.

## 2026-07-17 · Orchestrator (main) — knowledge base v2
Expanded `CLAUDE.md` into the full "permanent brain": System Overview, end-to-end runtime flow, explicit Agent 1 / Agent 2 contracts (+ "evidence intelligence" note, no class rename), Algorithms-vs-LLM split, 10 Golden Rules, Project Principles, truthful repo tree, feature recipe, debug playbook, Future Vision pointer. Added `.claude/project/future_vision.md`; extended `interview_notes.md` (cosine similarity, deterministic vs LLM, single-LLM). No runtime code touched.

## 2026-07-17 · Orchestrator (main)
Set up parallel development. Built the permanent knowledge base (`CLAUDE.md`, `.claude/skills`, `.claude/commands`, `.claude/project`), committed to `main`. Created worktrees `cc2-backend` (`dev/backend`) and `cc2-frontend` (`dev/frontend`); configured envs (backend reuses main venv; frontend `npm ci`). Verified: backend **11 pass** from the worktree via reused venv, frontend **build ✓**. `main` is ahead of `origin/main` by the knowledge-base commits (not yet pushed).

## 2026-07-16 · Orchestrator (main)
Pushed the stable baseline. Resolved an in-progress rebase, split hiring-platform vs legacy into two commits, stripped Claude attribution, force-pushed clean history. Generalized skill semantics (removed hardcoded category dict + transfer if-chains) and removed placeholder agents. Baseline: **11 tests pass**, **build ✓**.

---
### Template for new entries
```
## <date> · <backend|frontend|qa|orchestrator>
<slice/feature>. <files touched at a high level>. Contract change: <yes/no + note>.
Verify: pytest <n pass>, build <ok/fail>, integration <ok/fail>.
```
