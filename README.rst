==============
Kitware BatBot
==============

|Tests| |Codecov| |Wheel| |Docker| |ReadTheDocs| |Huggingface|

.. contents:: Quick Links
    :backlinks: none

.. sectnum::

Development Environment
-----------------------

.. code-block:: bash
    # Find repo on host machine
    cd ~/code/batbot

    # Build Docker image
    docker build -t kitware/batbot:latest .

    # Start Docker container using image
    docker run \
       -it \
       --rm \
       --entrypoint bash \
       --name batbot \
       -v $(pwd):/code \
       kitware/batbot:latest

    ########################
    # Inside the container #
    ########################

    # Activate Python environment
    source /venv/bin/activate

    # Install local version
    pip install -e .

    # Run batbot
    batbot --help

Spectrogram Extraction
----------------------

Here are the steps for extracting the compressed spectrogram:

* Create the STFT

  * Load the original waveform at the original sample rate
  * Resample waveform to 250kHz
  * Convert to a STFT spectrogram (fft=512, method=blackmanharris, window=256, hop=16)
  * Convert complex power STFT to amplitude STFT (dB)

* Normalize the STFT

  * Trim STFT to minimum and maximum frequencies (5kHz to 120kHz)
  * Subtract the per-freqency median dB (reduce any spectral bias / shift)
  * Set global dynamic range to -80 dB from the global maximum amplitude
  * Calculate the global median non-minimum dB (greater than -80dB)
  * Calculate the median absolute deviation (MAD)
  * Autogain the dynamic range to (5 * MAD) below the global amplitude median, if necessary

* Quantize the STFT

  * Quantize the floating-point amplitude STFT to a 16-bit integer representation spanning the full dynamic range (65,536 bins)
  * Vertically flip the spectrogram (low frequencies on bottom) and convert to a C-contiguous array

* Find Candidate Chirps

  * Create a 12ms sliding window with a 3ms stride
  * Keep the time windows that show a substantial right-skew across 10% of the frequency range
  * Add any user-provided time windows (annotations) to the found candidates windows
  * Merge any overlapping time windows into a set of contiguous time ranges
  * Tighten the candidate time ranges (and separate as needed) by repeating the same skew-based filter with a smaller sliding window and stride

* Extract Chirp Metrics

  * *for each candidate chirp*
  * *Start*: First, find the peak amplitude location.
  * Step 1 - Normalize the chirp to the full 16-bit range.  Calculate a histogram and identify the most common dB and standard deviation.  Scale the amplitude values using an inverted PDF, weighting each value by its inverse probability of being noise (values below the most common dB are set to zero)
  * Step 2 - Apply a median filter and re-normalize
  * Step 3 - Apply a morphological open operation
  * Step 4 - Blur the chirp (k=5) and re-normalize
  * Step 5 - Find contours using the "marching squares" algorithm and select the one that contains the peak amplitude.  Extract the convex hull of the contour and smooth the resulting outline
  * Step 6 - Extract a segmentation mask for the contour
  * Step 7 - Locate the harmonic (doubling the frequency) and echo (right edge of the contour to the end of the chirp time range) regions.  Remove any overlapping noise from the chirp contour.
  * Step 8 - Locate the start, end, and characteristic frequency points (peak amplitude) and calculate an optimization cost grid for the contour using the masked amplitudes.
  * Step 9 - Solve a minimum distance optimization using A* that also maximizes the amplutide values from start to end points.
  * Step 10 - Smooth the contour path, extract the contour's slope, then identify the knee, heel, and other defining attributes.
  * *End*: Finally, if any of the above steps fails, or the chirp's attributes do not make semantic sense, then skip the candidate chirp.

* Create Output

  * Collect all valid chirps regions and metadata, create a compressed spectrogram
  * Write the 16-bit spectrogram as a series of 8-bit JPEGs image chunks (max width per chunk 50k pixels)
  * Write the file and chirp metadata to a JSON file.

How to Install
--------------

.. code-block:: bash

    pip install batbot

or, from source:

.. code-block:: bash

   git clone https://github.com/Kitware/batbot
   cd batbot
   pip install -e .

To then add GPU acceleration, you need to replace `onnxruntime` with `onnxruntime-gpu`:

.. code-block:: bash

   pip uninstall -y onnxruntime
   pip install onnxruntime-gpu

How to Run
----------

You can run the Gradio demo with:

.. code-block:: bash

   python app.py

To run with Docker:

.. code-block:: bash

   cd batbot
   docker run \
     -it \
     --entrypoint bash \
     --rm \
     --name batbot \
     -v $(pwd):/code \
     kitware/batbot:latest

or to run the Gradio app:

.. code-block:: bash

   docker run \
     -it \
     --rm \
     -p 7860:7860 \
     --gpus all \
     --name batbot \
     kitware/batbot:latest \
     python3 app.py

To run with Docker Compose:

.. code-block:: yaml

    version: "3"

    services:
      batbot:
        image: kitware/batbot:latest
        command: python3 app.py
        ports:
          - "7860:7860"
        environment:
          CLASSIFIER_BATCH_SIZE: 512
        restart: unless-stopped
        deploy:
          resources:
            reservations:
              devices:
                - driver: nvidia
                  device_ids: ["all"]
                  capabilities: [gpu]

and run ``docker compose up -d``.

How to Build and Deploy
-----------------------

Docker Hub
==========

The application can also be built into a Docker image and is hosted on Docker Hub as ``kitware/batbot:latest``.  Any time the ``main`` branch is updated or a tagged release is made (see the PyPI instructions below), an automated GitHub CD action will build and deploy the newest image to Docker Hub automatically.

To do this manually, use the code below:

.. code-block:: bash

    docker login

    export DOCKER_BUILDKIT=1
    export DOCKER_CLI_EXPERIMENTAL=enabled
    docker buildx create --name multi-arch-builder --use

    docker buildx build \
        -t kitware/batbot:latest \
        --platform linux/amd64 \
        --push \
        .

PyPI
====

To upload the latest BatBot version to the Python Package Index (PyPI), follow the steps below:

#. Edit ``batbot/__init__.py:65`` and set ``VERSION`` to the desired version

    .. code-block:: python

        VERSION = 'X.Y.Z'


#. Push any changes and version update to the ``main`` branch on GitHub and wait for CI tests to pass

    .. code-block:: bash

        git pull origin main
        git commit -am "Release for Version X.Y.Z"
        git push origin main


#. Tag the ``main`` branch as a new release using the `SemVer pattern <https://semver.org/>`_ (e.g., ``vX.Y.Z``)

    .. code-block:: bash

        git pull origin main
        git tag vX.Y.Z
        git push origin vX.Y.Z


#. Wait for the automated GitHub CD actions to build and push to `PyPI <https://pypi.org/project/batbot/>`_ and `Docker Hub <https://hub.docker.com/r/kitware/batbot>`_.

Tests and Coverage
------------------

You can run the automated tests in the ``tests/`` folder by running:

.. code-block:: bash

    pip install -r requirements/optional.txt
    pytest

You may also get a coverage percentage by running:

.. code-block:: bash

    coverage html

and open the `coverage/html/index.html` file in your browser.

Building Documentation
----------------------

There is Sphinx documentation in the ``docs/`` folder, which can be built by running:

.. code-block:: bash

    cd docs/
    pip install -r requirements/optional.txt
    sphinx-build -M html . build/

Logging
-------

The script uses Python's built-in logging functionality called ``logging``.  All print functions are replaced with ``log.info()``, which sends the output to two places:

#. the terminal window, and
#. the file `batbot.log`

Code Formatting
---------------

It's recommended that you use ``pre-commit`` to ensure linting procedures are run
on any code you write.  See `pre-commit.com <https://pre-commit.com/>`_ for more information.

Reference `pre-commit's installation instructions <https://pre-commit.com/#install>`_ for software installation on your OS/platform. After you have the software installed, run ``pre-commit install`` on the command line. Now every time you commit to this project's code base the linter procedures will automatically run over the changed files.  To run pre-commit on files preemtively from the command line use:

.. code-block:: bash

    pip install -r requirements/optional.txt
    pre-commit run --all-files

The code base has been formatted by `Black <https://black.readthedocs.io/en/stable/>`_.  Furthermore, try to conform to ``PEP8``.  You should set up your preferred editor to use ``flake8`` as its Python linter, but pre-commit will ensure compliance before a git commit is completed.  This will use the ``flake8`` configuration within ``setup.cfg``, which ignores several errors and stylistic considerations.  See the ``setup.cfg`` file for a full and accurate listing of stylistic codes to ignore.


.. |Tests| image:: https://github.com/Kitware/batbot/actions/workflows/testing.yaml/badge.svg?branch=main
    :target: https://github.com/Kitware/batbot/actions/workflows/testing.yaml
    :alt: GitHub CI

.. |Codecov| image:: https://codecov.io/gh/Kitware/batbot/branch/main/graph/badge.svg?token=FR6ITMWQNI
    :target: https://app.codecov.io/gh/Kitware/batbot
    :alt: Codecov

.. |Wheel| image:: https://github.com/Kitware/batbot/actions/workflows/python-publish.yaml/badge.svg
    :target: https://github.com/Kitware/batbot/actions/workflows/python-publish.yaml
    :alt: Python Wheel

.. |Docker| image:: https://img.shields.io/docker/image-size/kitware/batbot/latest
    :target: https://hub.docker.com/r/kitware/batbot
    :alt: Docker

.. |ReadTheDocs| image:: https://readthedocs.org/projects/batbot/badge/?version=latest
    :target: https://kitware-batbot.readthedocs.io/en/latest/?badge=latest
    :alt: ReadTheDocs

.. |Huggingface| image:: https://img.shields.io/badge/HuggingFace-running-success
    :target: https://huggingface.co/spaces/Kitware/batbot
    :alt: Huggingface
