Changelog
=========

Version 0.2.0
-------------

* Add MobileNet ONNX species classification for spectrogram images and WAV files.
* Add ``classify``, ``classify-wav``, and ``classify-bulk`` CLI commands.
* Add mirrored model retrieval, bulk species summaries, and a performance plotting example.
* Integrate parallel WAV spectrogram preprocessing with bounded batch inference.
* Show preprocessing and classification progress by default, with ``--no-progress`` support.
* Add explicit CPU/CUDA selection, CUDA library preloading, and GPU installation guidance.
* Validate ONNX session options in normal and reduced-thread modes, including fresh wheel installs.
* Support native macOS, Linux, and Windows paths, including Unicode image filenames and cache overrides.
* Use spawned preprocessing processes consistently to avoid inheriting native threads or GPU state.
* Separate CPU and NVIDIA runtime installation extras and add explicit Apple CoreML selection.
* Test source and wheel installs on all three platforms with Python 3.11 through 3.14.
* Modernize non-root Docker CPU (x86-64/ARM64) and NVIDIA (x86-64) images and add container smoke tests.
* Add an opt-in trusted GPU runner check that verifies actual CUDA kernel execution.

TODO
----

* Make PyPI
* Make Readthedocs
* Make Huggingface
* Reset git history
* Fix CI/CD
* Add example Colab notebooks
* Port over spectrogram and model inference code from BatAI
* Update tests
* Update documentation

Version 0.0.1
-------------

New Version
