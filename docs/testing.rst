Functional testing and coverage
===============================

The full suite enforces **at least 90% combined statement/branch coverage**
for ``batbot``. Branch measurement stays enabled and the measured package
scope is unchanged. Coverage is a regression guard, not evidence that model
predictions or every processing parameter have been scientifically validated.

Run and inspect
---------------

Install Git LFS and the development environment described in the contributor
guide. The standard CPU/test installation is sufficient for functional tests:

.. code-block:: bash

   python -m pip install -e ".[cpu,test]"
   python -m pytest --random-order-seed=1 --cov-report=term-missing --cov-report=xml
   python -m coverage report -m

Open ``coverage/html/index.html`` for the interactive report. To iterate on
selected tests without applying a whole-package coverage requirement:

.. code-block:: bash

   python -m pytest tests/test_spectrogram_signals.py --no-cov
   python -m pytest tests/test_classification_functional.py --no-cov

Use an ordinary local terminal/container with multiprocessing permissions;
the real spawned-worker tests need local process/semaphore support. Keep
``MPLBACKEND=Agg`` on headless machines. Run a different random-order seed
before submitting changes, especially those involving shared sessions or caches.

What the functional tests verify
--------------------------------

.. list-table:: Behavioral contracts
   :header-rows: 1
   :widths: 25 45 30

   * - Area
     - Expected behavior
     - Tests
   * - Signal processing
     - Known tone frequencies, Nyquist suppression, per-frequency gain,
       quantization, window geometry, candidate thresholds, contour masks,
       harmonic/echo direction, and significance thresholds
     - ``test_spectrogram_signals.py``
   * - WAV artifacts and debug outputs
     - Readable images, dimensions/dtypes/intensity range, persisted metadata,
       bounded segment/curve coordinates, original-sample-rate output above
       and below 250 kHz, preserved source bytes, and scoped debug cleanup
     - ``test_spectrogram_functional.py``
   * - Large spectrograms
     - Ordered 50,000-column image chunks whose widths sum to the full
       metadata width, with valid call/frequency metadata
     - ``test_spectrogram.py``
   * - Classifier transforms and aggregation
     - BGR pixel ordering, exact overlapping windows and padding, unequal
       inference-batch means weighted by window count, normalized finite
       real-model scores, ranked top-k, and session/provider behavior
     - ``test_classifier.py`` and classifier unit/runtime tests
   * - End-to-end classification
     - All four CLI routes execute real CPU ONNX inference, preserve legacy
       duplicates, retain full uncompressed model inputs, isolate corrupt
       WAV/image failures, serialize JSON, and compute consistent summaries
     - ``test_classification_functional.py``
   * - Concurrency
     - Real spawned preprocessing matches serial inference; synthetic
       synchronization tests verify concurrent work, ordering, bounded
       temporary storage, cleanup, and progress accounting
     - ``test_classifier_batch.py`` and runtime/API tests
   * - Evaluation visualization
     - Genus swaps preserve labels, score columns, targets, and predictions;
       shading counts cross-genus errors using the display aliases
     - ``test_classifier_performance.py``
   * - Example and notebooks
     - Example outputs are isolated under temporary paths; notebook CI
       executes both tutorials and checks predictions, figures, and exports
     - ``test_batbot.py`` and ``scripts/check_notebooks.py``

Audit findings and test design
------------------------------

The original spectrogram and example tests only called the implementation.
They now validate produced data and artifacts, and the example test no longer
writes into the source checkout. Shape-only image tests now have pixel-level
companions. A constant-output inference fixture could not detect incorrect
averaging: its replacement uses five distinct window scores in batches of
2, 2, and 1 and asserts the independently calculated weighted mean.

Three targeted, in-memory fault-injection probes also verified that the new
assertions fail for unweighted batch averaging, accidental BGR-to-RGB channel
conversion, and omitted genus-order swaps. These probes did not modify any
production files. They check sensitivity to those specific regressions, not
an exhaustive mutation score.

Existing mocked CLI/provider tests remain useful for option forwarding,
deterministic failures, and hardware-unavailable paths. They are complemented
by real-file/real-model functional tests; a mock's return value alone is not
evidence that the pipeline works. Test real outcomes at those boundaries.

Keep expectations independent of the function under test. Use explicit
numerical examples, tolerances justified by quantization/resampling, known
input metadata, and cross-artifact consistency checks. Assert contents, error
isolation, ordering, cleanup, and unchanged inputs, not only file existence,
successful return codes, or the number of mock calls. Keep expensive debug
tests bounded using a short local recording; do not download fixtures or add
private recordings to exercise branches.

Remaining validation boundaries
-------------------------------

* CPU functional tests do not prove CUDA or CoreML kernel execution. Provider
  mocks validate policy; actual acceleration requires the configured NVIDIA
  hardware runner or a native macOS CoreML run. Installation smoke checks
  remain separate packaging checks, not substitutes for these tests.
* GitHub's OS/Python grid validates platform behavior when CI runs. A passing
  local suite does not establish that every remote runner has passed.
* Tutorial execution uses local fixtures and CPU. Actual Colab authorization,
  uploads/Drive access, Codespaces provisioning, and cloud GPU allocations
  require the corresponding external environments.
* These tests do not establish species accuracy, probability calibration,
  biological detection thresholds, or monitoring-program acceptance. Evaluate
  those on independently reviewed, held-out data using the performance example.
  Code contributions belong in BatBot/BatAI; monitoring data belongs in NABat.
