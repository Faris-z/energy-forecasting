# Contributing to energy-forecasting

Thank you for considering a contribution! This document explains the development workflow, code standards, and how to submit changes.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Branching Strategy](#branching-strategy)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Adding a New Model](#adding-a-new-model)
- [Reporting Bugs](#reporting-bugs)

---

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/energy-forecasting.git
   cd energy-forecasting
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   make install
   ```

---

## Development Setup

Verify your environment:
```bash
make test      # all tests pass
make lint      # zero flake8 errors
make format    # code auto-formatted
```

---

## Branching Strategy

| Branch      | Purpose                                        |
|-------------|------------------------------------------------|
| `main`      | Stable, released code                          |
| `develop`   | Integration branch for upcoming releases       |
| `feature/*` | New features (branch off `develop`)            |
| `fix/*`     | Bug fixes (branch off `develop` or `main`)     |
| `chore/*`   | Dependency updates, CI changes, refactoring    |

Branch naming convention: `feature/add-transformer-model`, `fix/lstm-nan-bug`.

---

## Code Standards

### Type Hints
Every function must have complete type annotations for all parameters and return values.

### Docstrings
Use NumPy-style docstrings for all public functions, classes, and methods:
```python
def my_function(x: int, y: float) -> str:
    """Short one-line summary.

    Longer description if needed.

    Parameters
    ----------
    x : int
        Description of x.
    y : float
        Description of y.

    Returns
    -------
    str
        Description of return value.
    """
```

### Logging
Use `logging.getLogger(__name__)` — **never** `print()`. Log at the appropriate level:
- `DEBUG` — verbose diagnostic output
- `INFO` — normal operational messages
- `WARNING` — recoverable issues
- `ERROR` — failures that don't crash the program
- `CRITICAL` — fatal errors

### No Hardcoded Values
All hyperparameters, paths, and thresholds must live in `config/config.yaml`. Access them via the `config` dict in function arguments.

### Formatting
Run before every commit:
```bash
make format
make lint
```

The CI pipeline will reject PRs that fail either check.

---

## Testing

### Running tests
```bash
make test                        # full suite with coverage
pytest tests/test_features.py   # single file
pytest -k "test_lag"            # pattern match
```

### Writing tests
- Place tests in `tests/` with the prefix `test_`.
- One test class per module (e.g., `TestLagFeatures` for `add_lag_features`).
- Tests must be deterministic — fix random seeds with `np.random.default_rng(seed)`.
- Do not rely on external network access or a pre-existing dataset.
- Minimum coverage target: **80%** for new modules.

---

## Submitting a Pull Request

1. Ensure all tests pass: `make test`
2. Ensure linting passes: `make lint`
3. Update `README.md` if you change public behaviour.
4. Open a PR against `develop` with:
   - A clear title describing the change
   - A summary of what was changed and why
   - Benchmark or metric comparison if you modified a model

PRs are reviewed within **3 business days**. Two approvals are required to merge into `main`.

---

## Adding a New Model

1. Create `src/models/my_model.py` following the pattern of `tree_models.py`:
   - Implement `fit(X_train, y_train, ...)` and `predict(X)` methods
   - Add a `from_config(config)` class method
   - Use `logging.getLogger(__name__)` throughout

2. Add hyperparameters to `config/config.yaml` under a new key.

3. Register the model in `src/models/__init__.py`.

4. Plug it into `train.py` (follow the XGBoost block as a template).

5. Add ensemble weight in `config.yaml → ensemble.weights`.

6. Write unit tests in `tests/test_<model_name>.py`.

---

## Reporting Bugs

Open an issue with:
- A **minimal reproducible example**
- Python version (`python --version`)
- OS and architecture
- Full error traceback

Thank you for helping improve this project!
