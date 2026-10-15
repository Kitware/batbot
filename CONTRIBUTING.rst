Contributing to BatBot and BatAI
================================

Welcome! Small fixes, questions, reproducible experiments, documentation,
accessibility improvements, and early prototypes are all valuable contributions.
You do not need a finished model or a large dataset to start collaborating.

Code belongs in open source; monitoring data belongs in NABat
-------------------------------------------------------------

BatBot is part of an ecosystem supporting bat-acoustic research and monitoring:

* `Kitware BatBot <https://github.com/Kitware/batbot>`_ provides reusable
  spectrogram processing, inference APIs, command-line tools, and tutorials.
* `Kitware BatAI <https://github.com/Kitware/batai>`_ is the related open-source
  bat-detection and classification research project. Code contributions are
  encouraged in **both BatBot and BatAI**. Discuss cross-project changes in
  linked issues and keep reusable inference behavior in BatBot.
* The `USGS North American Bat Monitoring Program (NABat)
  <https://www.usgs.gov/centers/fort-collins-science-center/science/north-american-bat-monitoring-program-nabat>`_
  supports coordinated bat monitoring. **Contribute monitoring data to the
  NABat database through the NABat Partner Portal, not through GitHub pull
  requests or issues in BatBot or BatAI.** Start with the
  `NABat data-upload guidance <https://www.nabatmonitoring.org/upload-data>`_
  and the `Partner Portal <https://sciencebase.usgs.gov/nabat/>`_ for current
  account, project, metadata, and submission requirements.

A notebook's JSON/CSV files are research exports, **not a NABat submission
format or a direct database integration**. Follow current NABat templates and
quality-control guidance, retain provenance and manual-review status, and
coordinate with your NABat project lead. Automated labels are suggestions,
not verified observations or evidence of NABat approval of this model.

Do not put field recordings, site coordinates, sensitive species locations,
credentials, private datasets, or identifiable notebook outputs in Git history.
Prefer synthetic signals and minimal generated images for regression tests.
If a bug genuinely needs real data, coordinate a permitted sharing route with
maintainers; collecting monitoring datasets should still go through NABat.

Start collaborating
-------------------

#. Open a `BatBot issue <https://github.com/Kitware/batbot/issues>`_ for a bug,
   tutorial improvement, or API proposal. For BatAI research changes, use its
   `issue tracker <https://github.com/Kitware/batai/issues>`_. Cross-link related work.
#. Describe the question or hypothesis, expected behavior, and a small,
   non-sensitive reproducer. Include OS, Python, BatBot/ONNX Runtime versions,
   actual execution providers, and relevant error messages.
#. For a larger idea, agree on a small first experiment and success criterion.
   Draft pull requests and notebooks are welcome before an implementation is polished.
#. Keep experiments reproducible: name inputs without sharing private data,
   record model hashes, parameters, environment, and random seeds, and compare
   results against the same baseline. Report unsuccessful experiments too.
#. Invite review from both software and bat-acoustics collaborators. Explain
   scientific assumptions and API tradeoffs; do not treat model scores as truth.

Set up a development environment
--------------------------------

Use 64-bit Python 3.11–3.14. Install Git and Git LFS first. Work in a fork
when you do not have write access, and use a topic branch for each change.
The repository's ``pyproject.toml`` is the source of truth for dependencies
and tool configuration; do not reintroduce duplicated requirements lists.

macOS/Linux:

.. code-block:: bash

   git clone https://github.com/YOUR-ACCOUNT/batbot.git
   cd batbot
   git remote add upstream https://github.com/Kitware/batbot.git
   git lfs install --local
   git lfs pull
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e ".[cpu,test,docs,notebook,dev]"
   python -m pip check
   pre-commit install
   python -m ipykernel install --user --name batbot --display-name "Python (BatBot)"

Windows PowerShell (activation is optional):

.. code-block:: powershell

   git clone https://github.com/YOUR-ACCOUNT/batbot.git
   cd batbot
   git remote add upstream https://github.com/Kitware/batbot.git
   git lfs install --local
   git lfs pull
   py -3 -m venv .venv
   ./.venv/Scripts/python.exe -m pip install --upgrade pip
   ./.venv/Scripts/python.exe -m pip install -e ".[cpu,test,docs,notebook,dev]"
   ./.venv/Scripts/python.exe -m pip check
   ./.venv/Scripts/pre-commit.exe install
   ./.venv/Scripts/python.exe -m ipykernel install --user --name batbot --display-name "Python (BatBot)"

After this, use the virtual environment's Python executable for commands below.
Binary audio/image wheels usually need no system compiler. Minimal Linux
systems may need ``libgomp1``, ``libsndfile1``, and ``zlib1g``; Windows ONNX
Runtime requires the Microsoft Visual C++ runtime. Hadolint is needed for
Dockerfile linting; install it using its official installation instructions.
For Python-only work without Hadolint, temporarily use ``SKIP=hadolint``
(PowerShell: ``$env:SKIP = 'hadolint'``) when running pre-commit.

CPU and NVIDIA ONNX Runtime distributions share one import name. Install only
one in an environment. For CUDA development create a separate environment and
install ``".[gpu,test,docs,notebook,dev]"`` instead of the CPU combination.
The GPU extra uses CUDA 13/cuDNN 9 and still needs a compatible host driver.
See the README for CoreML, Docker, and hardware-runner setup. Do not mistake
an advertised provider for successful GPU kernel execution.

Codespaces and local dev containers
-----------------------------------

Use **Code > Codespaces > Create codespace** on your fork and choose your topic
branch. The checked-in ``.devcontainer`` installs Git LFS, a CPU runtime,
developer tools, notebook dependencies, and a **Python (BatBot)** kernel.
Open ``examples/notebooks/01_classify_your_files.ipynb`` and use **Run All**.
The same configuration works with VS Code's **Reopen in Container** locally.
Codespaces uses CPU by default; do not assume a hosted GPU is available.

If environment configuration changes, rebuild the dev container. Keep test
files in ignored ``data/input/`` and experiments in ``artifacts/notebooks/``.
Cloud runtimes may be temporary or billed; stop unused codespaces and download
results before deleting them. Do not upload sensitive recordings without
permission to use the chosen cloud service.

Prototype, then promote reusable behavior
-----------------------------------------

Start with the notebooks and the developer architecture guide. Change one
parameter at a time, use a small representative subset, and preserve a baseline
report. The detailed notebook uses a short inspection clip so debugging does
not create thousands of images or exhaust notebook memory.

Keep exploratory code in a notebook or an example helper first. Once useful,
extract reusable behavior into a focused function, define input/output and
failure semantics, add tests, and document the public API. Avoid implementing
a second production inference pipeline in a notebook. Shared tutorial helpers
live in ``examples/notebooks/notebook_utils.py`` and are not a stable API.

Keep model preprocessing compatible with the trained model: uint8 BGR images,
224-by-224 windows, horizontal scaling, window stride, and class order matter.
The current WAV classifier uses full, uncompressed spectrogram chunks from
``compute(..., fast_mode=False)``. Compressed/masked/debug images are useful
diagnostics, **not interchangeable default model inputs**. Proposed changes
to that choice need explicit evaluation and regression tests.

Prefer one reusable ``classifier.Classifier`` per experiment. In notebooks,
start with ``preprocess_workers=1``; standalone scripts using processes need
an ``if __name__ == '__main__':`` guard. Bound worker counts and file subsets
before scaling up. Training, dataset preparation, and related research may
belong in BatAI; coordinate cross-project interfaces rather than copying code.

Tests and quality checks
------------------------

.. code-block:: bash

   python -m pytest --random-order-seed=1
   python -m pytest tests/test_classifier_runtime.py --no-cov
   python -m pytest tests/test_paths.py --no-cov
   pre-commit run --all-files
   python scripts/check_notebooks.py
   python scripts/check_notebooks.py --execute --output-dir artifacts/notebooks/validation
   python -m sphinx -W --keep-going -b html docs docs/_build/html
   python -m build
   python -m twine check dist/*

Notebook execution uses local fixtures and does not install packages, upload
files, mount Drive, or contact external services. The notebook extra supplies
the kernel/execution dependencies. Submitted notebooks must have cleared
outputs and execution counts. Use ``python scripts/check_notebooks.py --clear``
before committing after an interactive session. Review diffs for embedded data
and secrets even after clearing outputs.

Documentation builds are offline by default after installing dependencies.
Set ``BATBOT_DOCS_ONLINE=1`` to resolve optional Python/NumPy cross-project
references; those builds require access to the upstream documentation sites.

Add focused tests for new behavior and failure cases. Prefer ``tmp_path``,
native ``Path`` objects, Unicode paths, and injected external services. Check
session reuse, stable ordering, malformed files, empty results, and bounded
memory. Use a real bundled-model smoke test alongside mocks when inference
behavior changes. Keep Python 3.11 support, ``pytest==8.4.2``, and the existing
Black/isort/Flake8 tools; mypy and classifier type-check gates are not enabled.

The full suite enforces at least 90% combined statement/branch coverage.
Favor behavioral assertions with independent expected values over execution-only
smoke tests. See ``docs/testing.rst`` for the functional-test inventory, audit
findings, coverage commands, and limits of hardware/scientific validation.

CI tests source and built wheels on macOS/Linux/Windows with Python 3.11–3.14.
Notebook execution runs separately on the minimum and newest supported Python,
and documentation builds treat warnings as errors. Docker checks cover CPU
amd64/arm64 and NVIDIA amd64 installs. Actual CUDA kernel tests require the
configured trusted GPU runner; installation/CPU-fallback checks alone do not
prove GPU acceleration. PyPI publishing waits for wheel validation.
All three workflows can be run manually for validation. Python publishing
requires an explicit release tag and opt-in for manual releases; see the
release-recovery instructions in ``docs/publishing.rst``. Keep shared workflow
permissions, concurrency, shell defaults, checkout options, and step naming
consistent when adding CI jobs.

Pull-request checklist
----------------------

* Explain what changed, why, user-visible/API effects, and how it was tested.
* Link issues and related BatAI work; state known scientific and platform limitations.
* Include tests and update relevant examples, API docs, changelog, and notebook guides.
* Share small aggregate charts or synthetic reproductions, not private monitoring data.
* Keep notebooks deterministic, runnable top-to-bottom, and free of committed outputs.
* Avoid unrelated formatting, dependency churn, model binaries, or generated build products.
* For model changes, document provenance, class ordering, checksums, licensing,
  evaluation design, and compatibility; discuss artifact distribution with maintainers.
* Mark prototypes as draft, list open questions, and invite incremental review.

For release responsibilities and PyPI OIDC setup, see ``docs/publishing.rst``.
