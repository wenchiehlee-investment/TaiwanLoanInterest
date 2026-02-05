# Repository Guidelines

## Project Structure & Module Organization
This repository currently focuses on project documentation for a Taiwan loan interest data collection system.

- `README.md`: primary project description, data sources, and integration context.
- `CLAUDE.md`: contributor guidance for AI assistants working on the project.
- `LICENSE`: MIT license.

As code is added, keep source modules grouped by function (for example, `src/collectors/`, `src/parsers/`, `src/exporters/`) and place any sample data under `data/` or `fixtures/` so it is clearly separated from code.

## Build, Test, and Development Commands
No build or test commands are defined yet. When adding automation, document the commands here and keep them minimal. Example placeholders:

- `make fetch`: download latest source files.
- `pytest`: run unit tests for parsers.
- `python -m src.collectors.cbc`: run a single collector locally.

## Coding Style & Naming Conventions
There is no established code style in this repository yet. When you introduce code:

- Prefer 2 or 4 spaces for indentation and keep it consistent within a language.
- Use `snake_case` for Python functions/modules and `PascalCase` for class names.
- Keep data files named by source and date, for example `cbc_5newloan_2025-01.csv`.
- Add formatting or linting tools (e.g., `ruff`, `black`, `pytest`) only when they are used consistently.

## Testing Guidelines
No test framework is configured. If tests are added:

- Name tests by target behavior (for example, `test_parse_5newloan.py`).
- Keep fixtures small and stored under `tests/fixtures/`.
- Include a short note in `README.md` about how to run the suite.

## Commit & Pull Request Guidelines
Recent commits use short, imperative messages (for example, “Add weighted average interest rate data source (AVERAGEIR)”). Follow this style and keep the subject under ~72 characters.

Pull requests should include:

- A concise summary of what changed and why.
- Links to related issues or data source pages, when relevant.
- Notes on data schema changes or output format updates.

## Configuration & Data Notes
Data outputs should be in CSV or JSON to support integration with GoodInfo.Analyzer. Preserve historical time series and avoid overwriting raw source files; prefer versioned or date-stamped outputs.
