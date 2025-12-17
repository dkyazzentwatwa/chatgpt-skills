# Public Test Transparency

This folder is for publishing test status so users can see that the skills library is continuously checked.

## How to produce a public report
1) Run the repository tests from the root:
```bash
python -m unittest discover tests
```
2) Save the output for sharing:
```bash
python -m unittest discover tests | tee public/test-report.txt
```
3) Commit `public/test-report.txt` (or date-stamped copies) when you want to make results visible.

## What the tests cover today
- Structural checks: every skill folder has `SKILL.md` with hyphen-case `name`/`description` aligned to the directory naming.
- Baseline docs: `README.md` and `CLAUDE.md` exist and are non-empty.

## Notes
- Keep reports lightweight and text-based; regenerate after adding or updating skills.
- If tests are expanded (e.g., linting, packaging dry-runs), mention the new coverage at the top of the report.
