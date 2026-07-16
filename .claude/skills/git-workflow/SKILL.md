---
name: git-workflow
description: Git and worktree workflow for the AI Hiring Platform — branch roles, local-first merging, contract-first coordination, verified-checkpoint pushes, and the no-Claude-attribution rule. Use for any commit, merge, branch, or push decision.
---

# Git & Worktree Workflow

## Worktrees & branches
- `codeclause2/` (main worktree) — branch `main` — integration, QA, pushes.
- `../cc2-backend` — branch `dev/backend` — Session 1.
- `../cc2-frontend` — branch `dev/frontend` — Session 2.
- List worktrees: `git worktree list`.

## Principles
- **Local-first.** Develop and commit on `dev/*` branches locally. Do not push feature branches.
- **Contract-first.** Coordinate across sessions via the API contract (backend schemas ↔ frontend types), not by pushing. Both sessions work independently and merge only when a vertical slice is complete.
- **Verified checkpoints only.** `origin/main` receives a push only after the full QA gate is green (pytest + build + integration).

## Slice lifecycle
1. Backend + Frontend build the slice on their branches (backend lands the contract first).
2. In the main worktree: `git merge --no-ff dev/backend` then `git merge --no-ff dev/frontend`.
3. Run the full verification gate (see `qa-verify`).
4. Green → `git push origin main`. Then sync dev branches forward: in each worktree `git merge main`.

## Commit rules (CRITICAL)
- **Never** add Claude/AI authorship traces: no `Co-Authored-By: Claude …`, no "🤖 Generated with Claude Code". Commits are authored solely by the user (`nithinskumar866`).
- Clear, conventional messages (`feat:`, `fix:`, `chore:`, `docs:`, `test:`). Describe the slice, not the tool.

## Safety
- Keep `main` green and pushes fast-forward. Never force-push a shared checkpoint unless explicitly asked.
- gitignored per-worktree files (`.env`, `.venv`, `node_modules`) do not travel with the branch — each worktree is set up locally.
