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

import concurrent.futures
from multiprocessing import Manager
from os.path import basename, exists, join, splitext
from pathlib import Path

import pooch
from tqdm import tqdm

from batbot import utils

log = utils.init_logging()
QUIET = not utils.VERBOSE


from batbot import spectrogram  # NOQA

VERSION = '0.1.3'
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
    out_file_stem=None,
    output_folder=None,
    fast_mode=False,
    force_overwrite=False,
    quiet=False,
    plot_uncompressed_amplitude=False,
    debug=False,
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
    output_paths, compressed_paths, metadata_path, metadata = spectrogram.compute(
        filepath,
        out_file_stem=out_file_stem,
        output_folder=output_folder,
        fast_mode=fast_mode,
        force_overwrite=force_overwrite,
        quiet=quiet,
        plot_uncompressed_amplitude=plot_uncompressed_amplitude,
        debug=debug,
    )

    return output_paths, compressed_paths, metadata_path


def pipeline_multi_wrapper(
    filepaths,
    out_file_stems=None,
    fast_mode=False,
    force_overwrite=False,
    worker_position=None,
    quiet=False,
    tqdm_lock=None,
):
    """Fault-tolerant wrapper for multiple inputs.

    Args:
        filepaths (_type_): _description_
        out_file_stems (_type_, optional): _description_. Defaults to None.
        fast_mode (bool, optional): _description_. Defaults to False.
        force_overwrite (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    if out_file_stems is not None:
        assert len(filepaths) == len(
            out_file_stems
        ), 'Input filepaths and out_file_stems have different length.'
    else:
        out_file_stems = [None] * len(filepaths)

    outputs = {'output_paths': [], 'compressed_paths': [], 'metadata_paths': [], 'failed_files': []}
    # print(filepaths, out_file_stems)
    if tqdm_lock is not None:
        tqdm.set_lock(tqdm_lock)
    for in_file, out_stem in tqdm(
        zip(filepaths, out_file_stems),
        desc='Processing, worker {}'.format(worker_position),
        position=worker_position,
        total=len(filepaths),
        leave=True,
    ):
        try:
            output_paths, compressed_paths, metadata_path = pipeline(
                in_file,
                out_file_stem=out_stem,
                fast_mode=fast_mode,
                force_overwrite=force_overwrite,
                quiet=quiet,
            )
            outputs['output_paths'].extend(output_paths)
            outputs['compressed_paths'].extend(compressed_paths)
            outputs['metadata_paths'].append(metadata_path)
        except Exception as e:
            outputs['failed_files'].append((str(in_file), e))

    return tuple(outputs.values())


def parallel_pipeline(
    in_file_chunks,
    out_stem_chunks=None,
    fast_mode=False,
    force_overwrite=False,
    num_workers=0,
    threaded=False,
    quiet=False,
    desc=None,
):

    if out_stem_chunks is None:
        out_stem_chunks = [None] * len(in_file_chunks)

    if len(in_file_chunks) == 0:
        return None
    else:
        assert len(in_file_chunks) == len(
            out_stem_chunks
        ), 'in_file_chunks and out_stem_chunks must have the same length.'

    if threaded:
        executor_cls = concurrent.futures.ThreadPoolExecutor
    else:
        executor_cls = concurrent.futures.ProcessPoolExecutor

    num_workers = min(len(in_file_chunks), num_workers)

    outputs = {'output_paths': [], 'compressed_paths': [], 'metadata_paths': [], 'failed_files': []}

    lock_manager = Manager()
    tqdm_lock = lock_manager.Lock()

    with tqdm(total=len(in_file_chunks), disable=quiet, desc=desc) as progress:
        with executor_cls(max_workers=num_workers) as executor:

            futures = [
                executor.submit(
                    pipeline_multi_wrapper,
                    filepaths=file_chunk,
                    out_file_stems=out_stem_chunk,
                    fast_mode=fast_mode,
                    force_overwrite=force_overwrite,
                    worker_position=index % num_workers,
                    quiet=quiet,
                    tqdm_lock=tqdm_lock,
                )
                for index, (file_chunk, out_stem_chunk) in enumerate(
                    zip(in_file_chunks, out_stem_chunks)
                )
            ]

            for future in concurrent.futures.as_completed(futures):
                output_paths, compressed_paths, metadata_path, failed_files = future.result()
                outputs['output_paths'].extend(output_paths)
                outputs['compressed_paths'].extend(compressed_paths)
                outputs['metadata_paths'].extend(metadata_path)
                outputs['failed_files'].extend(failed_files)
                progress.update(1)

    return tuple(outputs.values())


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
        _, _, _, metadata = spectrogram.compute(filepath)
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
            url=f'https://raw.githubusercontent.com/Kitware/batbot/main/examples/{TEST_WAV}',
            known_hash=TEST_WAV_HASH,
            progressbar=True,
        )
        assert exists(wav_filepath)

    log.debug(f'Running pipeline on WAV: {wav_filepath}')

    import time

    output_stem = join('output', splitext(basename(wav_filepath))[0])
    start_time = time.time()
    results = pipeline(
        wav_filepath,
        out_file_stem=output_stem,
        fast_mode=False,
        force_overwrite=True,
        plot_uncompressed_amplitude=True,
    )
    stop_time = time.time()
    print('Example pipeline completed in {} seconds.'.format(stop_time - start_time))

    log.debug(results)
