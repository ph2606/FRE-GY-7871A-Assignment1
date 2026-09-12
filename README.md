# FRE-GY 7871A - Assignment 1

Panagiotis Housos | ph2606 | Fall 2026

[View the executed notebook](assignment1.ipynb) for the analysis of negative
sentiment and uncertainty in 2021-2025 ARK-company filings. It contains Tables
1-6, Figure 1, methods, interpretation, and disclosed sensitivity checks.
The final sample contains 1,134 filings from 72 firms.

This repository contains the notebook with saved outputs, analysis and download
code, supporting tests and environment files, and [AI_USE.md](AI_USE.md).
**No data files are committed.** All downloaded and derived data stay in ignored
`data/` and `outputs/` directories, created automatically when needed.
Upload the separate PDF report to Brightspace and include the repository URL.

## Reproduce the notebook

Use Python 3.12. Run these PowerShell commands from the repository root after
cloning [this repository](https://github.com/ph2606/FRE-GY-7871A-Assignment1):

```powershell
py -3.12 -m venv .venv
$assignmentPython = ".\.venv\Scripts\python.exe"
& $assignmentPython -m pip install -r requirements-lock.txt
& $assignmentPython -m ipykernel install --sys-prefix --name assignment1

$env:SEC_USER_AGENT = "Your Full Name your.netid@nyu.edu"
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

& $assignmentPython -m pytest tests -q
& $assignmentPython scripts/00_get_lexicons.py
& $assignmentPython scripts/01_build_universe.py
& $assignmentPython scripts/02_download_filings.py --limit 3
& $assignmentPython scripts/02_download_filings.py
& $assignmentPython scripts/03_get_market_data.py
& $assignmentPython -m jupyter nbconvert --to notebook --execute --inplace assignment1.ipynb --ExecutePreprocessor.kernel_name=assignment1 --ExecutePreprocessor.timeout=3600
```

Replace the SEC contact placeholder with your real name and email. Downloads
are cached and resumable. Script 01 restores the assigned holdings snapshot
from pinned starter commit `532c65cf91cdf62a8d37c9bbe6ff0961c152d756`; it does
not substitute current holdings. Reviewed mapping decisions and source links
are in [scripts/01_build_universe.py](scripts/01_build_universe.py).

The three-company trial checks acquisition before the full download and uses
separate trial files, as requested in the starter instructions.

`requirements-lock.txt` pins the tested environment; `requirements.txt` lists
direct dependencies. `python scripts/04_analyze.py --check` checks local input
readiness. Running that script without `--check` executes the core analysis
without opening Jupyter. The notebook also runs the disclosed sensitivities.

The notebook explains the active dictionary's 2,345 negative words versus the
starter's 2,355 nonzero flags, reconciles the holdings and filing counts, and
defines the exact scoring, event windows, controls, and inference. These are
retrospective associations using full-sample document frequencies. All 81
automated tests pass; saved notebook cells contain no execution errors.
