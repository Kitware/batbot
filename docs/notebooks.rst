Interactive notebooks: bring your own recordings
================================================

The tutorials are standalone Jupyter notebooks. Each runs from a clean kernel
using CPU by default, with either your own files or an integrity-checked public
example. No upload, Drive authorization, or external data submission happens
automatically. Read cells before running them and use only data you are
permitted to process in the selected environment.

Choose a notebook
------------------

.. list-table:: Tutorial paths
   :header-rows: 1
   :widths: 30 45 25

   * - Notebook
     - What you get
     - Launch
   * - ``01_classify_your_files.ipynb``
     - WAV/image folders, failures, model-input previews, top-class/count/score charts,
       optional reviewed-label comparison, JSON/CSV/PNG/ZIP export
     - `Colab: classification <https://colab.research.google.com/github/Kitware/batbot/blob/main/examples/notebooks/01_classify_your_files.ipynb>`_
   * - ``02_spectrogram_walkthrough.ipynb``
     - A bounded WAV clip, waveform/STFT/gain/quantization, candidate/contour/mask
       debugging, full/compressed comparisons, ONNX windows and final classification
     - `Colab: walkthrough <https://colab.research.google.com/github/Kitware/batbot/blob/main/examples/notebooks/02_spectrogram_walkthrough.ipynb>`_

Download sources:
:download:`classification notebook <../examples/notebooks/01_classify_your_files.ipynb>`,
:download:`walkthrough notebook <../examples/notebooks/02_spectrogram_walkthrough.ipynb>`.
The shared helper is ``examples/notebooks/notebook_utils.py``; these utilities
are examples, not a stable package API. The runtime clones/uses the repository
so helper code matches the checkout. The model's bundled copy is verified;
a Git LFS pointer or missing bundled model triggers the normal verified mirror.

Before this change is merged, the main-branch Colab links will not yet contain
the notebooks. Open the notebook from its topic branch in Colab's GitHub picker
and change ``SOURCE_REF`` in the setup cell to that branch. For reproducible
published experiments, use a release tag/commit rather than a moving branch.
An existing checkout is never reset or silently switched by notebook setup.

Codespaces: the quickest collaborative workspace
-------------------------------------------------

#. Open your fork or `Kitware/batbot <https://github.com/Kitware/batbot>`_, choose
   the intended branch, and select **Code > Codespaces > Create codespace**.
   Alternatively use `the Codespaces launcher <https://codespaces.new/Kitware/batbot>`_.
#. Wait for ``.devcontainer/post-create.sh`` to install Git LFS, the package,
   CPU inference, developer tools, and the notebook kernel. Check the terminal
   for its Ready message. Rebuild the container after changing its configuration.
#. Open either notebook and select **Python (BatBot)**. Run All once with the
   example before changing settings.
#. Drag WAV/JPG/PNG files into ignored ``data/input/`` using the VS Code file
   explorer. Or set ``DATA_DIR`` to an existing native folder/file path.
#. Rerun from the configuration cell onward. Increase ``MAX_FILES`` gradually.
   Download the generated ZIP using the file explorer before deleting the codespace.

The dev container also works locally with VS Code's Dev Containers extension.
It does not expose a public Jupyter port or disable authentication. Standard
Codespaces hardware should be treated as CPU-only: GitHub notes that GPU machine
types are `not generally available
<https://docs.github.com/en/codespaces/developing-in-a-codespace/getting-started-with-github-codespaces-for-machine-learning>`_.
Normal service usage/storage charges and resource limits still apply.

Google Colab: upload or use a permitted Drive folder
----------------------------------------------------

#. Open a Colab link above, or use **File > Open notebook > GitHub** and locate
   the notebook on the correct branch. Saving a personal copy is optional.
#. Keep ``RUNTIME_PROFILE = "cpu"`` for a small first experiment. Run the setup
   cell and wait for installation into the active kernel.
#. To upload files, set ``UPLOAD_FILES = True`` and run the upload cell. The
   multi-file picker writes into ``DATA_DIR``. Existing names are handled by
   Colab's upload tool; inspect the discovered input table for duplicates.
#. For an existing folder, set ``MOUNT_DRIVE = True`` and, for example,
   ``DATA_DIR = Path("/content/drive/MyDrive/batbot-tests")``. Only authorize
   Drive access if you have permission. For large datasets, consider a small
   representative subset copied into the session rather than processing all Drive data.
#. Keep remaining defaults for an initial run. Read the input/error table,
   view the charts, and set ``DOWNLOAD_RESULTS = True`` in the final cell to
   download the ZIP. Drive uploads and result downloads are separate opt-in actions.

Colab's documented file upload/download helpers are available in its
`I/O tutorial <https://colab.research.google.com/notebooks/io.ipynb>`_. Session
storage and runtime availability are temporary; see the
`Colab FAQ <https://research.google.com/colaboratory/faq.html>`_. Saving the
notebook is not a backup of uploaded files or generated artifacts.

Optional accelerators
~~~~~~~~~~~~~~~~~~~~~~

Select a GPU runtime in Colab **before** setup, then choose a compatible profile:

.. list-table:: Notebook runtime profiles
   :header-rows: 1

   * - Profile
     - Runtime / behavior
   * - ``cpu``
     - Standard ONNX Runtime; fast setup; no accelerator allocation required
   * - ``auto``
     - Inspect ``nvidia-smi`` driver CUDA support; choose CUDA 13, CUDA 12, or CPU
   * - ``cuda12``
     - ``onnxruntime-gpu[cuda,cudnn]==1.26.0`` for CUDA 12-era drivers/environments
   * - ``cuda13``
     - ONNX Runtime 1.31 series plus matching CUDA 13/cuDNN runtime packages
   * - ``coreml``
     - Native macOS CoreML; not available in Linux Codespaces/Colab containers

CUDA runtime packages are large. The NVIDIA driver and supported GPU still
come from the host. Colab hardware and driver versions can vary; do not assume
that the newest CUDA 13 wheel matches an allocated runtime. For an existing
PyTorch installation use the upstream
`ONNX Runtime compatibility guide <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html>`_.
The CUDA 12 notebook profile installs core BatBot plus a compatible runtime,
not BatBot's CUDA 13 ``gpu`` extra. It does not change package-wide defaults.

CPU/GPU ONNX distributions share a Python import. The helper refuses to
uninstall an existing opposite runtime automatically. Use a fresh Colab runtime
or a separate local environment when changing profiles, and restart the kernel
after explicitly replacing a runtime. Explicit providers must initialize;
inspect actual ``get_providers()`` output, not just advertised availability.
Neither these notebook checks nor CPU fallback demonstrate CUDA kernel execution;
the configured GPU Docker CI job performs that stronger check.

Local Jupyter
---------------

In an activated Python 3.11+ environment:

.. code-block:: bash

   python -m pip install -e ".[cpu,notebook]"
   python -m ipykernel install --user --name batbot --display-name "Python (BatBot)"
   jupyter lab

Open a notebook and select that kernel. On Windows use the environment's
``Scripts/python.exe`` and ``Scripts/jupyter-lab.exe`` if not activated.
Native paths with spaces/Unicode and ``pathlib.Path`` work. Quote paths in
shell commands; avoid mixing Windows and Unix path syntaxes in one runtime.

How to read outputs
---------------------

* ``results.json``: original result schema, failures, summary, model hash, and runtime provenance.
* ``results.csv``: one row per input, including failed files and error messages.
* ``class_scores.csv``: per-class scores for successful inputs; no failure is silently converted to noise.
* PNG charts: predicted file counts, score distributions/heatmaps, top classes;
  walkthrough stages, selected debugging images, and input-window scores.
* Retained processing images/metadata: full model inputs plus compressed/debug
  views for inspection. Walkthrough archives include the short inspection WAV.
* ``experiment_settings.json`` / ``inspection_settings.json``: configuration
  and the source/clip interval needed to repeat the experiment.

Each run gets a new folder under ignored ``artifacts/notebooks/``. The quick
classification archive excludes original input recordings; the walkthrough
archive includes its generated clip. Both may expose sensitive paths/derived
data. Review before sharing, download deliberately, and clear source-notebook
outputs before contributing code.

Species summaries count **classified input files**, not bat populations or
individual calls. Scores are not calibrated probabilities. Reviewed folder
labels are optional; failed files are excluded from exploratory accuracy and
must be reported separately. Do not use a public demonstration recording as
ground truth or infer NABat validation from a model output.

Code and data have different contribution routes
-------------------------------------------------

Bring prototypes, improvements, and reproducible code to
`BatBot <https://github.com/Kitware/batbot>`_ and
`Kitware BatAI <https://github.com/Kitware/batai>`_. Contribute monitoring data
to the `USGS NABat program
<https://www.usgs.gov/centers/fort-collins-science-center/science/north-american-bat-monitoring-program-nabat>`_
database through its `Partner Portal <https://sciencebase.usgs.gov/nabat/>`_,
using `current NABat upload guidance <https://www.nabatmonitoring.org/upload-data>`_.
These notebook exports are not portal templates and no automatic submission is
implemented. Follow project permissions, metadata standards, manual review,
and quality-control requirements rather than attaching datasets to GitHub.

Troubleshooting and extending a prototype
------------------------------------------

* **Missing helper/notebook on main:** choose the PR branch in both Colab and
  ``SOURCE_REF``, or use the local checkout. Existing checkouts are not overwritten.
* **Wrong Python/kernel:** check ``sys.executable`` and select the environment
  where ``.[cpu,notebook]`` was installed. Setup requires Python 3.11 or newer.
* **Missing model or LFS pointer:** run ``git lfs pull`` in a checkout, or allow
  BatBot's verified mirror download. Do not bypass checksum validation.
* **No files discovered:** inspect ``DATA_DIR`` and supported suffixes. The
  default may use the public example; disable ``USE_EXAMPLE_IF_EMPTY`` to make
  missing user data an explicit error.
* **CUDA initialization/ABI problems:** inspect driver/runtime versions, avoid
  installing both ONNX distributions, choose the matching profile, and restart
  a fresh kernel. CPU is a useful compatibility baseline.
* **Memory/time pressure:** reduce ``MAX_FILES``, ``CLIP_SECONDS``, inference
  batch size, and workers. A notebook result table still grows with dataset size;
  use the CLI/bulk API and an appropriate machine for large studies.
* **No contours/debug views:** inspect sample rate and signal quality, then try
  another short interval. Missing contours do not establish a noise label.
* **Comparing processing variants:** keep source data/model fixed and change
  one documented parameter; do not substitute compressed/masked images into
  the default classifier without evaluating that change.

Validation commands and review expectations are in the developer guide.
Notebook execution checks run against local WAV/JPG fixtures without installing
packages, authorizing cloud services, or fetching data; actual Colab/Codespaces
UI interactions and NVIDIA hardware still need an appropriate external runtime.
