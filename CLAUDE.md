# Instructions for AI coding agents

## Pull requests and the AI reviewer

Runqueue's reviewer agent reviews a pull request every time a commit is pushed to it, as long as the PR is open and not a draft. Each review is a paid model call (from a few cents to about $0.50 for a large PR), and the 23ventures org has a monthly model budget. When the budget runs out, automatic reviews stop for every project until it is raised or the month resets. Draft PRs are never reviewed, so:

- **Open every PR as a draft:** `gh pr create --draft`. Keep the `pr0: AXY-<n>` line in the description. The item stays in progress while the PR is a draft.
- **Mark it ready only when the work is finished:** every acceptance criterion is met, tests, lint and type checks pass locally, and CI is green. Then run `gh pr ready <number>`. This moves the item to review and starts one review.
- **After a review, push once per round.** Fix all the findings from one review, run the checks again, and push them together. Don't push one commit per fix.
- **Don't push work in progress to a PR that is ready.** If it needs substantial rework, run `gh pr ready --undo <number>` to turn it back into a draft, then mark it ready again when the work is done.
- **Ask for a review again** (`/rereview`, or "Review again" in Runqueue) only when no review ran on the latest commit. Answering the reviewer's question in the PR conversation already starts a new review on its own.
