# AGENTS.md

Instructions for AI coding agents (Claude Code, Codex, Copilot, Cursor, etc.) working in this repository.

## Rules

### 1. Never commit or push
- Do **not** run `git commit`, `git push`, `git merge`, `git rebase`, `git tag`, or `git reset`, and do not amend history.
- Do **not** create branches or pull requests.
- Leave every change uncommitted in the working tree so the user can review it and commit it themselves.
- These rules hold even when a task seems to call for a commit. Tell the user the work is ready to commit and stop there.

### 2. Always follow the user's direction
- Do what the user asks: no more, no less. Don't add features, refactors, or changes they didn't request.
- If a request is unclear, ask before acting instead of guessing.
- If you disagree with an approach, say so briefly, then do it the user's way unless they change their mind.
- Ask before deleting files, overwriting data (especially `srilanka_weekly_data.csv`), or installing dependencies.
- Report results honestly: say what changed, what was skipped, and what failed.
