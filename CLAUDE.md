# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spotify Confidence is a Python library for A/B test analysis. It provides convenience wrappers around statsmodel's functions for computing p-values and confidence intervals. The library supports both frequentist (Z-test, Student's T-test, Chi-squared) and Bayesian (BetaBinomial) statistical methods, with features for variance reduction, sequential testing, and sample size calculations.

## Development Commands

### Setup
```bash
# Install with development dependencies (including tox-uv)
uv pip install -e ".[dev]"
```

### Testing
```bash
# Run all tests with coverage
uv run pytest

# Run tests without coverage reports
uv run pytest --no-cov

# Run specific test file
uv run pytest tests/frequentist/test_z_test.py

# Run specific test
uv run pytest tests/frequentist/test_z_test.py::test_name

# Run all tests across Python versions
uv run tox
```

### Code Quality
```bash
# Format code with black (line length: 119)
uv run black spotify_confidence tests

# Check formatting without making changes
uv run black --check --diff spotify_confidence tests

# Lint with flake8 (max line length: 120)
uv run flake8 spotify_confidence tests

# Run all quality checks (as done in CI)
uv run black --check --diff spotify_confidence tests && uv run flake8 spotify_confidence tests && uv run pytest
```

### Build
```bash
# Build distribution packages
uv run python -m build
```

## Architecture

### Core Design Pattern

The library follows an object-oriented design with separation of concerns:

1. **Statistical Test Classes**: High-level APIs (`ZTest`, `StudentsTTest`, `ChiSquared`, `BetaBinomial`, `ZTestLinreg`)
2. **Experiment Class**: Base class containing shared analysis methods for frequentist tests
3. **Computer Classes**: Perform the actual statistical computations
4. **Grapher Classes**: Generate visualizations using Chartify

All main test classes inherit from abstract base classes in `spotify_confidence/analysis/abstract_base_classes/`:
- `ConfidenceABC`: Base for all statistical test classes
- `ConfidenceComputerABC`: Base for computation logic
- `ConfidenceGrapherABC`: Base for visualization logic

### Module Structure

```
spotify_confidence/
├── analysis/
│   ├── abstract_base_classes/    # ABC definitions for the framework
│   ├── frequentist/               # Frequentist statistical methods
│   │   ├── confidence_computers/  # Statistical computation logic
│   │   ├── experiment.py          # Base class for frequentist tests
│   │   ├── z_test.py              # Z-test implementation
│   │   ├── t_test.py              # Student's T-test implementation
│   │   ├── chi_squared.py         # Chi-squared test
│   │   ├── z_test_linreg.py       # Z-test with linear regression variance reduction
│   │   ├── sequential_bound_solver.py  # Group sequential testing
│   │   ├── multiple_comparison.py # Multiple testing correction
│   │   └── sample_size_calculator.py
│   ├── bayesian/                  # Bayesian methods
│   │   └── bayesian_models.py     # BetaBinomial implementation
│   ├── constants.py               # Shared constants
│   └── confidence_utils.py        # Shared utility functions
├── samplesize/                    # Sample size calculations
├── examples.py                    # Example data generators
├── chartgrid.py                   # Chart grid utilities
└── options.py                     # Global configuration
```

### Key Classes and Their Relationships

- **Experiment** (in `frequentist/experiment.py`): The core base class for frequentist tests. Provides methods like:
  - `summary()`: Overall metric summaries
  - `difference()`: Pairwise comparisons
  - `multiple_difference()`: Multiple comparisons with correction
  - `difference_plot()`, `summary_plot()`, etc.: Visualization methods
  - `sample_size()`: Required sample size calculations
  - `statistical_power()`: Power analysis

- **ZTest, StudentsTTest, ChiSquared**: Thin wrappers that initialize `Experiment` with the appropriate computer and method

- **Computer Classes** (in `frequentist/confidence_computers/`): Handle the statistical calculations
  - `ZTestComputer`, `TTestComputer`, `ChiSquaredComputer`: Specific computation implementations
  - All inherit from `ConfidenceComputerABC`

- **ChartifyGrapher**: Implements visualization using the Chartify library

### Data Model

The library works with DataFrames containing sufficient statistics:
- `numerator_column`: Sum or count (e.g., sum of conversions)
- `denominator_column`: Total observations (e.g., total users)
- `numerator_sum_squares_column`: Sum of squares (optional, for variance calculations)
- `categorical_group_columns`: Treatment/control groups and other dimensions
- `ordinal_group_column`: Time-based grouping for sequential analysis

### Important Conventions

1. **Method Column**: Tests add a `METHOD_COLUMN_NAME` to data indicating the test type (e.g., "z-test", "t-test")

2. **Multiple Comparison Correction**: Supported methods defined in `constants.py`:
   - Standard: bonferroni, holm, hommel, sidak, FDR methods
   - SPOT-1 variants: Custom Spotify methods for specific use cases

3. **Non-Inferiority Margins (NIMs)**: Can be specified as absolute values or relative percentages

4. **Sequential Testing**: The `sequential_bound_solver.py` module implements group sequential designs with spending functions

5. **Variance Reduction**: `ZTestLinreg` uses pre-exposure data to fit a linear model and reduce variance (CUPED method)

## Testing Guidelines

- Tests are organized to mirror the source structure under `tests/`
- Use pytest fixtures for common test data
- Tests check both DataFrame outputs and chart generation
- Coverage target is configured in `pyproject.toml`

## Python Version Support

Supports Python 3.9, 3.10, 3.11, and 3.12. The `tox.ini` includes a `py39-min` environment that tests with minimum dependency versions.

The project uses `tox-uv` to leverage uv's fast package installation and environment management in tox, significantly speeding up multi-environment testing. The GitHub Actions CI workflow also uses uv for faster dependency installation.

## Code Style

- Black formatting with 119 character line length
- Flake8 linting with max line length 120
- Ignored flake8 rules: E203, E231, W503
- Excluded from linting: `.venv`, `.tox`, `dist`, `build`, `scratch.py`, `confidence_dev`
