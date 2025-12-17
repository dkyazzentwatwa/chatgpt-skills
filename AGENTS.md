# Repository Guidelines

## Project Structure & Module Organization
- Root directories are individual skills (e.g., `qr-code-generator/`) with required `SKILL.md` frontmatter (`name`, `description`) and optional `scripts/`, `references/`, `assets/`.
- Repo docs: `README.md` (overview), `CLAUDE.md` (agent workflow guidance), `RESEARCH_NOTES.md` (architecture). Update listings in README and CLAUDE when adding a skill.
- `skill-creator/` holds scaffolding and packaging utilities (`scripts/init_skill.py`, `quick_validate.py`, `package_skill.py`); prefer them over manual setup.
- Utility audits at the root: `audit_all_skills.py`, `verify_docs.py` keep structure and documentation in sync.

## Build, Test, and Development Commands
- `python3 skill-creator/scripts/init_skill.py my-new-skill --path .` scaffold a compliant skill folder.
- `python3 skill-creator/scripts/quick_validate.py path/to/skill` quick check of frontmatter naming and description.
- `python3 audit_all_skills.py` full repository audit (structure, shebangs, requirements hints).
- `python3 verify_docs.py` ensure README/CLAUDE skill lists match actual folders.
- `python3 skill-creator/scripts/package_skill.py path/to/skill [./dist]` validate then zip a skill for distribution.

## Coding Style & Naming Conventions
- Skill folder and `name` fields use hyphen-case lowercase (`data-storyteller`); keep `description` specific, <200 chars, and action-oriented.
- Write SKILL.md instructions as concise, imperative steps; move long reference material to `references/`.
- Python scripts: include `#!/usr/bin/env python3`, prefer argparse CLIs, 4-space indentation, dependencies pinned in `scripts/requirements.txt`, and brief usage notes in docstrings.

## Testing Guidelines
- Treat validations as tests: run quick validate on the target skill, then `audit_all_skills.py` and `verify_docs.py` before submitting.
- For new scripts, add simple input/output examples or smoke-test commands in SKILL.md so behavior is clear; expose deterministic CLI entrypoints.

## Commit & Pull Request Guidelines
- Git history uses short imperative summaries (e.g., `add new skills`, `update docs`); follow that tone.
- PRs should note new/changed skills, scripts touched, validation results, dependency additions, and links to related issues. Include screenshots only when assets/templates change.

## Security & Assets
- Do not commit secrets or licensed data; keep assets lightweight and justify large binaries in SKILL.md.
- Pin third-party versions and avoid unnecessary dependencies in skill scripts.
