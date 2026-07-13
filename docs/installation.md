# Installation

## Requirements

- **Python 3.11 or newer** (the codebase uses 3.11+ typing features).
- A working C toolchain is **not** required — all dependencies ship wheels.

Runtime dependencies (installed automatically): `numpy`, `pandas`, `pyarrow`, `scipy`. The HTML report
additionally needs `plotly` and `jinja2` (the `report` extra).

## Install from a clone (recommended for v1.0.0)

`biasguard` is not yet on PyPI, so install it editable from a clone:

```bash
git clone https://github.com/Giantars/biasguard.git
cd biasguard

python -m venv .venv
# Windows (PowerShell):
. .venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -e ".[dev]"     # editable install with the dev + report extras
```

The `dev` extra pulls in the full quality toolchain (`pytest`, `pytest-cov`, `ruff`, `black`, `mypy`,
`pandas-stubs`) plus `plotly`/`jinja2`. If you only want to *run* backtests and reports, use:

```bash
pip install -e ".[report]"  # runtime + HTML report, no dev tools
```

## Verify the install

```bash
python -c "import biasguard; print(biasguard.__version__)"   # 1.0.0
pytest -q                                                    # the full suite should pass
```

Or run a bundled example end to end:

```bash
python examples/08_educational_strategies.py
```

## Optional extras

| Extra | Adds | When you need it |
| --- | --- | --- |
| *(none)* | numpy / pandas / pyarrow / scipy | Running backtests + integrity checks |
| `report` | plotly, jinja2 | Generating the HTML report |
| `dev` | pytest, ruff, black, mypy, stubs, + `report` | Contributing / running the gate |

## Editor setup

The package ships `py.typed`, so editors and `mypy` see full type information out of the box. The repo is
formatted with **black** (line length 100) and linted with **ruff**; enabling "format on save" with black
keeps diffs clean. See [contributing.md](contributing.md) for the exact commands CI runs.
