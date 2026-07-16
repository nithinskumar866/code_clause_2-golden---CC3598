Merge a completed vertical slice into `main` (run in the main / integration worktree only).

1. Confirm you are in the main worktree on `main`: `git worktree list` and `git branch --show-current`.
2. Merge the ready dev branches (backend first — it lands the contract):
```
git merge --no-ff dev/backend      # if a backend slice is ready
git merge --no-ff dev/frontend     # if a frontend slice is ready
```
   Backend/frontend edit disjoint trees, so conflicts should be rare. If one occurs, resolve at the API-contract boundary (schemas ↔ types) and re-run verification.
3. Run the full verification gate (`/verify-all`): backend pytest + frontend build + end-to-end integration.
4. If GREEN, this is a candidate checkpoint — see `/release` to push.
5. After a successful checkpoint, sync the dev branches forward: in each worktree run `git merge main`.

Never merge a slice whose own tests/build are red. No Claude attribution on merge commits.
