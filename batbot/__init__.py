"""
The above components must be run in the correct order, but BatBot also offers a processing pipeline.

The machine learning (ML) model can be pre-downloaded and fetched by a single call to
:func:`batbot.fetch` and the unified pipeline can be run by the function :func:`batbot.pipeline`.
Below is example code for how these components interact.

Furthermore, there is an application demo file (``app.py``) that shows how the entire pipeline can
be run on WAV files.

.. code-block:: python

    # Get WAV filepath
    filepath = '/path/to/file.wav'

    # Run tiling
    output_paths, metadata_path, metadata = spectrogram.compute(filepath)
"""

from os.path import exists, join
from pathlib import Path

import pooch

from batbot import utils

log = utils.init_logging()
QUIET = not utils.VERBOSE


from batbot import spectrogram  # NOQA

VERSION = '0.1.0'
version = VERSION
__version__ = VERSION

PWD = Path(__file__).absolute().parent.parent


def fetch(pull=False, config=None):
    """
    Fetch the Classifier ONNX model file from a CDN if it does not exist locally.

    This function will throw an AssertionError if the download fails or the
    file otherwise does not exist locally on disk.

    Args:
        pull (bool, optional): If :obj:`True`, force using the downloaded version
            stored in the local system's cache.  Defaults to :obj:`False`.
        config (str or None, optional): the configuration to use.  Defaults to :obj:`None`.

    Returns:
        None

    Raises:
        AssertionError: If the model cannot be fetched.
    """
    raise NotImplementedError


def pipeline(
    filepath,
    config=None,
    # classifier_thresh=classifier.CONFIGS[None]['thresh'],
    clean=True,
):
    """
    Run the ML pipeline on a given WAV filepath and return the classification results

    The final output is a list of time windows where a bat exists.
    Each dictionary has a structure with the following keys:

        ::

            {
                'l': class_label (str)
                'c': confidence (float)
                'x': x_top_left (float)
                'y': y_top_left (float)
                'w': width (float)
                'h': height (float)
            }

    Args:
        filepath (str): WAV filepath (relative or absolute)
        config (str or None, optional): the configuration to use.  Defaults to :obj:`None`.
        classifier_thresh (float or None, optional): the confidence threshold for the classifier's
            predictions.  Defaults to the default configuration setting.
        clean (bool, optional): a flag to clean up any on-disk spectrograms that were generated.
            Defaults to :obj:`True`.

    Returns:
        tuple ( float, list ( dict ) ): classifier score, list of time windows
    """
    # Generate spectrogram
    output_paths, metadata_path, metadata = spectrogram.compute(filepath)

    return output_paths, metadata_path


def batch(
    filepaths,
    config=None,
    # classifier_thresh=classifier.CONFIGS[None]['thresh'],
    clean=True,
):
    """
    Run the ML pipeline on a given batch of WAV filepaths and return the detections
    in a corresponding list.  The output is a list of outputs matching the output of
    :func:`batbot.pipeline`, except the processing is done in batch and is much faster.

    The final output is a list of lists of dictionaries, each representing a
    single detection.  Each dictionary has a structure with the following keys:

        ::

            {
                'l': class_label (str)
                'c': confidence (float)
                'x': x_top_left (float)
                'y': y_top_left (float)
                'w': width (float)
                'h': height (float)
            }

    Args:
        filepaths (list): list of str WAV filepath (relative or absolute)
        config (str or None, optional): the configuration to use.  Defaults to :obj:`None`.
        classifier_thresh (float or None, optional): the confidence threshold for the Classifier's
            predictions.  Defaults to the default configuration setting.
        clean (bool, optional): a flag to clean up any on-disk spectrograms that were generated.
            Defaults to :obj:`True`.

    Returns:
        tuple ( list ( float ), list ( list ( dict ) ) : corresponding list of classifier scores, corresponding list of lists of predictions
    """
    # Run tiling
    batch = {}
    for filepath in filepaths:
        _, _, metadata = spectrogram.compute(filepath)
        batch[filepath] = metadata

    raise NotImplementedError


def example():
    """
    Run the pipeline on an example WAV with the default configuration
    """
    TEST_WAV = 'example1.wav'
    TEST_WAV_HASH = '391efce5433d1057caddb4ce07b9712c523d6a815e4ee9e64b62973569982925'  # NOQA

    wav_filepath = join(PWD, 'examples', 'example1.wav')

    if not exists(wav_filepath):
        wav_filepath = pooch.retrieve(
            url=f'https://github.com/Kitware/batbot/{TEST_WAV}',
            known_hash=TEST_WAV_HASH,
            progressbar=True,
        )
        assert exists(wav_filepath)

    log.debug(f'Running pipeline on WAV: {wav_filepath}')

    results = pipeline(wav_filepath)

    log.debug(results)
