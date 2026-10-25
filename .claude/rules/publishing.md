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
- **Owner approval**: the `pypi` environment stops for the owner only once it has a required reviewer (Settings → Environments → pypi). Not set as of 2026-10-25: until it is, a green CI publishes at once
- **PyPI auth**: trusted publishers (OIDC), no API token stored — configured on PyPI to trust the `.github/workflows/publish.yml` workflow in the `pypi` GitHub environment

## Release Process

1. Commit all changes to `master`; mark the release in `CHANGELOG.md` (`[Unreleased]` → `[x.y.z] - date`)
2. `git tag v<major>.<minor>.<patch>` (semver)
3. `git push origin v<major>.<minor>.<patch>`
4. CI runs the gates and tests, then — after the owner's approval, once the environment requires it — builds, publishes and creates the GitHub Release

## Key Constraints

- **Never** hardcode a version in `pyproject.toml` — `hatch-vcs` reads it from the git tag
- **Never** run `uv publish` manually — always go through CI
- **Never** create tags without pushing the underlying commits first
- PyPI rejects duplicate versions — a tag can only be used once
- The `fallback-version` (`0.0.0`) is used only when building outside a git checkout; a checkout without tags builds `0.0.1.devN`

## Platform-Specific Install

macOS and Windows take PyPI's torch, already CPU-only (~1.7GB); Linux takes the CPU-only index, which avoids the CUDA build (~3.5GB → ~1.7GB):

```bash
uvx semantic-code-mcp                                                          # macOS, Windows
uvx --index pytorch-cpu=https://download.pytorch.org/whl/cpu semantic-code-mcp  # Linux
```
