# Repository Guidelines

## Project Structure & Module Organization

This is a flat Python project; run commands from the repository root. Core game rules live in `engine/game.py`, observations in `encoding/`, and the Gymnasium wrapper in `envs/`. Agent implementations belong in `agents/`; training, tournament evaluation, and exploitability tools belong in `training/`. Statistical analysis code is under `analysis/`, while entry-point utilities are in `scripts/`. Keep YAML settings in `configs/`, tests in `tests/`, generated JSON in `results/`, figures in `plots/`, and experiment artifacts in `runs/<run_name>/`.

The intended dependency direction is `engine` → `encoding` → `envs`/`agents` → `training`/`analysis`. Do not expose hidden game state to agents; search code must use `PistiGame.determinize()`.

## Build, Test, and Development Commands

```bash
python3 -m venv venv
venv/bin/pip install -e ".[dev]"          # editable install plus developer tools
venv/bin/python -m pytest                  # run the complete test suite
venv/bin/python -m pytest tests/test_game.py -q
venv/bin/black --check . && venv/bin/flake8 .  # verify formatting and lint
venv/bin/python -m training.train --config configs/default.yaml
venv/bin/python -m training.evaluate --agents greedy hunter --n-deals 300
```

Use `black .` to apply formatting. Training can be expensive; use a small timestep override for smoke tests.

## Coding Style & Naming Conventions

Target Python 3.8+, four-space indentation, and a 100-character line limit. Use `snake_case` for modules, functions, variables, and config keys; use `PascalCase` for classes and `UPPER_CASE` for constants. Add type hints where practical and keep public behavior documented. Preserve the environment invariant that episode return equals the scaled final score differential.

## Testing Guidelines

Tests use pytest and follow `tests/test_*.py` with functions named `test_*`. Add focused rule scenarios and invariant tests for engine changes. Changes to rewards, observations, determinization, or match pairing require regression coverage. Statistical claims must use mirrored deals and report paired 95% confidence intervals.

## Commit & Pull Request Guidelines

Recent commits use concise, imperative subjects such as `Add browser GUI` and `Fix null-byte corruption`, often followed by the result or rationale. Keep each commit focused. Pull requests should explain the change, affected configs or artifacts, and validation commands; link relevant issues. Include plots or screenshots for analysis and GUI changes, and avoid committing large checkpoints unless they are intentional research artifacts.
