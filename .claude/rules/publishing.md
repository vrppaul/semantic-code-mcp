---
paths:
  - ".github/workflows/*.yml"
  - "pyproject.toml"
---

# Publishing & Releases

## How It Works

- **Version source**: git tags via `hatch-vcs` — no hardcoded version in `pyproject.toml`
- **CI workflow**: `.github/workflows/publish.yml` — triggers on `v*` tag push and first runs the whole of `.github/workflows/ci.yml`
- **Pipeline**: tag push → CI (gates, tests, build) → the `pypi` environment → `uv build` → `uv publish` (PyPI trusted publishers / OIDC) → `gh release create` with auto-generated notes
- **Fresh lock**: a release also needs `fresh-lock`, `scripts/check_fresh_lock.py`, which refuses a `uv.lock` whose runtime packages a fresh resolve would not take; dev tools, which no user installs, are not compared
- **Weekly**: `.github/workflows/fresh-resolve.yml` moves the lock to the newest versions within the bounds, checks that no dev package holds a runtime one back, and runs the suite, as `uvx` installs them
- **Owner approval**: the `pypi` environment stops for the owner only once it has a required reviewer (Settings → Environments → pypi). Not set as of 2026-10-25: until it is, a green CI publishes at once
- **PyPI auth**: trusted publishers (OIDC), no API token stored — configured on PyPI to trust the `.github/workflows/publish.yml` workflow in the `pypi` GitHub environment

## Release Process

1. Move the lock to the newest versions within the bounds (`uv lock --upgrade`), run the suite and both evals on it; commit all changes to `master`; mark the release in `CHANGELOG.md` (`[Unreleased]` → `[x.y.z] - date`)
2. Just before tagging, `uv run --no-project python scripts/check_fresh_lock.py` exits 0; else back to 1
3. `git tag v<major>.<minor>.<patch>` (semver)
4. `git push origin v<major>.<minor>.<patch>`
5. CI runs the gates and tests, then — after the owner's approval, once the environment requires it — builds, publishes and creates the GitHub Release
6. If `fresh-lock` refuses the tag (a runtime package released between steps 2 and 4), nothing reached PyPI and the version is still free: delete the tag (`git tag -d v<version>`, `git push --delete origin v<version>`) and go back to 1

## Key Constraints

- **Never** hardcode a version in `pyproject.toml` — `hatch-vcs` reads it from the git tag
- **Never** run `uv publish` manually — always go through CI
- **Never** create tags without pushing the underlying commits first
- PyPI rejects duplicate versions — a tag can only be used once
- The `fallback-version` (`0.0.0`) is used only when building outside a git checkout; a checkout without tags builds `0.0.1.devN`

## Install

One command on every platform, with no torch and no extra index (about 0.6 GB):

```bash
uvx semantic-code-mcp
```
