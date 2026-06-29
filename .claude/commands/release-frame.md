Perform a semver release of robometric-frame. Follow these steps exactly.

## Step 1 — Determine the new version

If the user passed an argument (e.g. `/release minor` or `/release 0.3.0`), use it.
Otherwise ask: "What kind of release? (major / minor / patch)" and wait for a response before continuing.

Read the current version from `pyproject.toml` (the `version = "..."` line under `[project]`).

Compute the new version following semver rules:
- `patch`: increment the patch digit (0.1.0 → 0.1.1)
- `minor`: increment minor, reset patch (0.1.0 → 0.2.0)
- `major`: increment major, reset minor and patch (0.1.0 → 1.0.0)
- explicit version string (e.g. `0.3.0`): use it directly, validate it matches `X.Y.Z`

## Step 2 — Update version in source files

Edit **`pyproject.toml`**: change `version = "OLD"` to `version = "NEW"` under `[project]`.

Edit **`src/robometric_frame/__init__.py`**: change the fallback string `__version__ = "OLD"` to `__version__ = "NEW"` (the line inside the `except` block).

## Step 3 — Update CHANGELOG.md

Run `git log` to find commits since the last tag:
```
git log $(git describe --tags --abbrev=0)..HEAD --oneline --no-decorate
```

If no prior tag exists, use all commits: `git log --oneline --no-decorate`.

Create or update `CHANGELOG.md` at the repo root. Use the Keep a Changelog format (https://keepachangelog.com). Prepend a new section for this release above any existing content:

```markdown
# Changelog

## [NEW_VERSION] - YYYY-MM-DD

### Added
- <commits that add features>

### Fixed
- <commits that fix bugs>

### Changed
- <other commits>
```

Group commits by type using keywords in commit messages (add/feat → Added, fix → Fixed, everything else → Changed). Use today's date. Keep existing sections below the new one unchanged.

## Step 4 — Summarize and confirm

Show the user:
- Old version → New version
- Files that will change: `pyproject.toml`, `src/robometric_frame/__init__.py`, `CHANGELOG.md`
- The new CHANGELOG section

Ask: "Looks good? Type 'yes' to commit, or 'no' to abort." Wait for confirmation before proceeding.

## Step 5 — Commit and tag

Stage and commit only these three files:
```
git add pyproject.toml src/robometric_frame/__init__.py CHANGELOG.md
git commit -m "Release v{NEW_VERSION}"
```

Then create an annotated tag:
```
git tag -a "v{NEW_VERSION}" -m "Release v{NEW_VERSION}"
```

Report the commit hash and tag. Remind the user to push with:
```
git push && git push --tags
```
Do NOT push automatically.
