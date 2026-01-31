Add, commit, and push all current changes. If "$ARGUMENTS" contains "release", also create and push a version tag.

## Steps

1. Run `git status` and `git diff --staged` and `git diff` to understand what changed.
2. Stage the relevant files (prefer specific files over `git add -A`).
3. Write a commit message following Conventional Commits with required scope (`type(scope): description`). Add `Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>`.
4. Commit and push.

## If releasing (`$ARGUMENTS` contains "release")

After the push:

5. Look at the latest git tags (`git tag -l 'v*' --sort=-v:refname | head -5`) to determine the current version.
6. Determine the next version based on the changes:
   - **patch** bump for fixes and docs
   - **minor** bump for new features
   - **major** bump for breaking changes
7. Ask the user to confirm the version before proceeding.
8. Create the tag: `git tag v<version>`
9. Push the tag: `git push origin v<version>`
10. Remind the user that CI will handle PyPI publish and GitHub Release creation.
