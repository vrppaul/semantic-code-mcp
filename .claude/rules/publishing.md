# Publishing & Releases

## How It Works

- **Version source**: git tags via `hatch-vcs` — no hardcoded version in `pyproject.toml`
- **CI workflow**: `.github/workflows/publish.yml` — triggers on `v*` tag push
- **Pipeline**: tag push → `uv build` → `uv publish` (PyPI trusted publishers / OIDC) → `gh release create` with auto-generated notes
- **PyPI auth**: trusted publishers (OIDC), no API token stored — configured on PyPI to trust the `publish.yml` workflow in the `pypi` GitHub environment

## Release Process

1. Commit all changes to `master`
2. `git tag v<major>.<minor>.<patch>` (semver)
3. `git push origin v<major>.<minor>.<patch>`
4. CI handles build, PyPI publish, and GitHub Release creation

## Key Constraints

- **Never** hardcode a version in `pyproject.toml` — `hatch-vcs` reads it from the git tag
- **Never** run `uv publish` manually — always go through CI
- **Never** create tags without pushing the underlying commits first
- PyPI rejects duplicate versions — a tag can only be used once
- The `fallback-version` (`0.0.0`) is only used when no tags exist (e.g., dev builds)

## Platform-Specific Install

- **macOS/Windows**: `uvx semantic-code-mcp` — PyPI torch is already CPU-only (~1.7GB)
- **Linux**: `uvx --index pytorch-cpu=https://download.pytorch.org/whl/cpu semantic-code-mcp` — avoids CUDA-bundled torch (~3.5GB → ~1.7GB)
