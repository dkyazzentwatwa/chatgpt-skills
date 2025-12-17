# Repository Tests

This suite sanity-checks the ChatGPT Skills library. Tests are Python `unittest` cases to avoid extra dependencies.

## What the tests cover
- Each top-level skill directory (hyphen-case) includes a `SKILL.md` file.
- `SKILL.md` frontmatter has `name` and `description`, and the `name` is hyphen-case and aligns with the folder name.
- Root docs `README.md` and `CLAUDE.md` exist and are non-empty.

## Running the tests
```bash
python -m unittest discover tests
```

Run from the repository root. All tests should pass before publishing new skills or documentation.

## Extending
- Add new checks as additional test methods in `tests/test_skills.py`.
- Keep tests dependency-free; if you need third-party tools, declare them explicitly before use.
