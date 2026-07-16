# Decision Log (ADRs)

Append-only record of significant decisions. Newest first. Format: date · decision · why.

## 2026-07-17 · Parallel dev via git worktrees (2 + main hub)
Two worktrees (`cc2-backend` → `dev/backend`, `cc2-frontend` → `dev/frontend`) plus the main worktree as integration/QA/push hub. QA is on-demand in the main worktree, not a 3rd permanent worktree. **Why:** the main worktree already has the full env; a dedicated QA worktree would add duplication and constant merges. Fewer moving parts, faster for a 2-day sprint.

## 2026-07-17 · Backend worktrees reuse the main `.venv`
Backend sessions run pytest via the main worktree's interpreter rather than creating a per-worktree venv. **Why:** the venv holds multi-GB ML deps (torch/faiss/transformers); duplicating per worktree costs minutes + disk with no benefit. Code resolves from each worktree's own cwd, so isolation is preserved. Verified green from `cc2-backend`.

## 2026-07-17 · Permanent knowledge base committed to `main`
`CLAUDE.md` + `.claude/{skills,commands,project}` committed before worktrees branch off, then worktrees fast-forwarded. **Why:** worktrees check out the branch tip; the base must exist first so every session starts with full context.

## 2026-07-16 · No Claude/AI attribution in git
No `Co-Authored-By: Claude` or "Generated with Claude Code" in commits/PRs; author is the user only. **Why:** explicit user preference; commits were rewritten + force-pushed to strip trailers.

## 2026-07-16 · Generic skill semantics replace hardcoded lookups
`skill_semantics_service.py` classifies categories by nearest-centroid over `TECH_TAXONOMY` embeddings and estimates transfer by cosine similarity + same-category boost. **Why:** the mandate is "algorithms not examples" — the old `CATEGORY_TAXONOMY` dict and `if req in [...] and has_python` chains didn't generalize to unseen skills.

## 2026-07-16 · Two-agent runtime, LangGraph sequential
Exactly two runtime agents (Candidate Intelligence → Hiring Decision), sequential graph, no branching. **Why:** mirrors real hiring (gather evidence, then decide), keeps every decision explainable; extra agents add complexity without value.
