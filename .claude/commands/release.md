Publish a verified checkpoint to GitHub (run in the main worktree only, after `/merge` + `/verify-all` are GREEN).

Preconditions (must all hold):
- On `main` in the main worktree.
- Full verification is GREEN (backend pytest + frontend build + integration).
- Working tree clean, history linear (fast-forward push).

Steps:
```
git log --oneline origin/main..main      # review exactly what will publish
git push origin main                       # fast-forward only
```

Rules:
- Push **only** verified checkpoints. GitHub holds green states only — never a red or half-finished slice.
- Never force-push a shared checkpoint unless explicitly asked.
- **No Claude/AI attribution** in any commit or PR (no `Co-Authored-By`, no "Generated with Claude Code"). Author is the user only.
- After publishing, ensure dev branches are synced forward (`git merge main` in each worktree).
