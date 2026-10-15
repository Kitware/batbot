Architecture and prototyping map
================================

Use this map to locate a change before editing the pipeline. BatBot's small
public API wraps lower-level processing functions; notebook experiments should
call those implementations instead of maintaining independent approximations.

.. list-table:: Ownership and extension points
   :header-rows: 1
   :widths: 30 35 35

   * - Area
     - Responsibility
     - Typical change and validation
   * - ``batbot/api.py``
     - Lazy high-level imports; preprocessing and legacy batch adapters
     - Preserve signatures, ordering, lightweight import, and per-file failures
   * - ``batbot/spectrogram/__init__.py``
     - Waveforms, STFT, normalization, candidate/contour extraction, images and metadata
     - Compare real and synthetic WAVs; inspect debug outputs and timing metadata
   * - ``classifier/config.py`` and ``model.py``
     - Immutable label/model configuration; verified resources and Pooch mirror/cache
     - Validate label order, SHA-256, local/mirror parity, offline bundled use
   * - ``classifier/dataloader.py``
     - Unicode-safe BGR image loading and uint8 window transforms
     - Match training geometry, padding, strides, color ordering, dtype
   * - ``classifier/inference.py``
     - Provider setup, SessionOptions, reused sessions, ordered threaded inference
     - Real model smoke plus provider/fallback/threading regressions
   * - ``classifier/bulk.py``
     - File discovery, bounded WAV worker batches, aggregation, failures, summaries
     - Test mixed inputs, worker isolation, cleanup, counts, and duplicate names
   * - ``batbot/_paths.py`` and ``batbot_cli.py``
     - Native paths/cache roots; Click commands and JSON/progress boundaries
     - Windows drives/UNC, Unicode, environment expansion, JSON stdout
   * - ``examples/notebooks`` and ``examples/plot_classifier_performance.py``
     - Interactive exploration and labeled evaluation, not a second production API
     - Execute clean notebooks; verify exported reports and scientific interpretation

What a WAV classification actually does
---------------------------------------

.. code-block:: text

   WAV (original file retained)
     -> mono waveform -> resample to 250 kHz
     -> STFT: 512-point FFT, 256-sample Blackman-Harris window, 16-sample hop
     -> select approximately 5–120 kHz; blank nonphysical bands above input Nyquist
     -> subtract per-frequency median, apply gain/autogain, clip, quantize, flip
     -> compute(fast_mode=False)
          |-> full uncompressed JPEG chunks ------------> classifier input
          |-> compressed chirp images, masks, metadata -> inspection/prototyping
          `-> debug=True: TIFF/PNG diagnostics ---------> inspection only
     -> resize each input to height 224, horizontal scale 0.5
     -> 224x224 BGR uint8 windows, stride 100; pad narrow images
     -> ONNX inference in bounded window batches
     -> mean scores per image -> window-count-weighted scores per WAV
     -> top label, top-k, all scores, window count -> file-level summary

The default gain is currently 120 dB before optional median-absolute-deviation
autogain. Consult the implementation for exact frequency-bin edges and
thresholds. The notebooks show actual arrays and output files. Resampling a
low-rate recording does not recover absent ultrasonic frequencies.

``fast_mode=True`` is a separate, reduced preprocessing mode: it uses 8-bit
data and different candidate windows and omits full segmentation. It is not
the current WAV classifier default. Do not change training-sensitive transforms
or switch full/compressed images simply because a preview looks clearer.

A full spectrogram's horizontal axis retains acquisition time; compressed
images concatenate selected ranges and no longer have a continuous time axis.
Classifier windows overlap and are not one-to-one with calls or individual bats.
``species_counts`` counts successful input-file predictions, excluding ``NOISE``;
it is not a population estimate. A score is not a calibrated probability.

Public APIs and error boundaries
--------------------------------

.. code-block:: python

   from pathlib import Path
   from batbot import classifier

   runner = classifier.Classifier(providers=["CPUExecutionProvider"], num_workers=1)
   images = runner.classify([Path("data/call.jpg")], progress=False)
   recording = runner.classify_wav(Path("data/recording.wav"), progress=False)
   report = runner.classify_bulk(
       [Path("data/input")], input_type="auto",
       preprocess_workers=1, progress=True,
       spectrogram_output=Path("artifacts/spectrograms"),
   )
   print(report["summary"])

Successful result dictionaries include ``path``, ``label``, ``confidence``,
``window_count``, ``top``, and ``scores``. WAV results can include retained
``spectrogram_paths``. Bulk failures contain ``path`` and ``error``. Validate
input discovery and failed-file counts as well as successful predictions.
Runtime/provider initialization errors are meaningful configuration failures;
do not catch them and quietly label all affected files as biological noise.

``pre -> predict -> post`` remains available for lower-level experiments.
Inference workers share one validated session and preserve ordering. WAV
preprocessing uses spawned processes, staged in bounded batches, so native
threads/GPU state are not inherited. Thread count, inference workers,
preprocessing workers, and batch size affect different resources; benchmark
them separately. Never start unbounded processes from notebook cells.

Reproducible model experiments
------------------------------

Record model filename/hash, class order, software versions, active execution
providers, parameters, dataset revision and permissions, and manual-label
provenance. Split evaluation data at an appropriate recording/site/night or
project level to avoid leakage from overlapping calls. Keep the same baseline
and summarize failures and class support, not only aggregate accuracy.

The labeled performance script accepts ``LABEL/*.wav`` and ``LABEL/*.jpg``
or ``*.jpeg`` and applies genus ordering to plot shading. It does not relabel
model outputs. Use held-out, reviewed labels; a folder name alone does not
establish trustworthy ground truth. Discuss research/training extensions in
`BatAI <https://github.com/Kitware/batai>`_, and contribute monitoring data via
the `NABat database <https://www.nabatmonitoring.org/upload-data>`_.
