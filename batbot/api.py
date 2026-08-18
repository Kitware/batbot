"""High-level BatBot processing APIs.

Heavy dependencies are imported inside the functions that use them so that a
plain ``import batbot`` remains inexpensive and does not initialize logging or
create files on disk.
"""

from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Sequence
from multiprocessing import Manager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from batbot._config import log

if TYPE_CHECKING:
    from batbot.classifier.types import ClassificationResult

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


def fetch(pull: bool = False, config: str | None = None) -> str:
    """Return the local ONNX classifier model path."""
    from batbot.classifier import fetch as fetch_classifier

    return fetch_classifier(pull=pull, config=config)


def pipeline(
    filepath: str | Path,
    out_file_stem: str | None = None,
    output_folder: str | None = None,
    fast_mode: bool = False,
    force_overwrite: bool = False,
    quiet: bool = False,
    plot_uncompressed_amplitude: bool = False,
    include_original_sr: bool = False,
    time_buffer_ms: float = 1.0,
    debug: bool = False,
) -> tuple[list[str], list[str], str | None]:
    """Generate spectrograms and metadata for one WAV file."""
    from batbot.spectrogram import compute

    output_paths, compressed_paths, metadata_path, _ = compute(
        str(filepath),
        out_file_stem=out_file_stem,
        output_folder=output_folder,
        fast_mode=fast_mode,
        force_overwrite=force_overwrite,
        quiet=quiet,
        plot_uncompressed_amplitude=plot_uncompressed_amplitude,
        include_original_sr=include_original_sr,
        time_buffer_ms=time_buffer_ms,
        debug=debug,
    )
    return output_paths, compressed_paths, metadata_path


def pipeline_multi_wrapper(
    filepaths: Sequence[str],
    out_file_stems: Sequence[str | None] | None = None,
    fast_mode: bool = False,
    force_overwrite: bool = False,
    worker_position: int | None = None,
    quiet: bool = False,
    tqdm_lock: Any = None,
) -> tuple[list[str], list[str], list[str | None], list[tuple[str, Exception]]]:
    """Run :func:`pipeline` for a chunk while retaining per-file failures."""
    from tqdm import tqdm

    if out_file_stems is not None and len(filepaths) != len(out_file_stems):
        raise ValueError('Input filepaths and out_file_stems have different length')
    if out_file_stems is None:
        out_file_stems = [None] * len(filepaths)

    output_paths: list[str] = []
    compressed_paths: list[str] = []
    metadata_paths: list[str | None] = []
    failed_files: list[tuple[str, Exception]] = []

    if tqdm_lock is not None:
        tqdm.set_lock(tqdm_lock)
    for in_file, out_stem in tqdm(
        zip(filepaths, out_file_stems),
        desc=f'Processing, worker {worker_position}',
        position=worker_position,
        total=len(filepaths),
        leave=True,
    ):
        try:
            outputs, compressed, metadata = pipeline(
                in_file,
                out_file_stem=out_stem,
                fast_mode=fast_mode,
                force_overwrite=force_overwrite,
                quiet=quiet,
            )
            output_paths.extend(outputs)
            compressed_paths.extend(compressed)
            metadata_paths.append(metadata)
        except Exception as error:  # pragma: no cover - worker failures depend on input data
            failed_files.append((str(in_file), error))

    return output_paths, compressed_paths, metadata_paths, failed_files


def parallel_pipeline(
    in_file_chunks: Sequence[Sequence[str]],
    out_stem_chunks: Sequence[Sequence[str | None]] | None = None,
    fast_mode: bool = False,
    force_overwrite: bool = False,
    num_workers: int = 0,
    threaded: bool = False,
    quiet: bool = False,
    desc: str | None = None,
) -> tuple[list[str], list[str], list[str | None], list[tuple[str, Exception]]] | None:
    """Run spectrogram processing chunks concurrently."""
    from tqdm import tqdm

    if not in_file_chunks:
        return None
    if out_stem_chunks is None:
        out_stem_chunks = [[None] * len(chunk) for chunk in in_file_chunks]
    if len(in_file_chunks) != len(out_stem_chunks):
        raise ValueError('in_file_chunks and out_stem_chunks must have the same length')

    executor_cls = (
        concurrent.futures.ThreadPoolExecutor
        if threaded
        else concurrent.futures.ProcessPoolExecutor
    )
    num_workers = min(len(in_file_chunks), num_workers)
    if num_workers <= 0:
        raise ValueError('num_workers must be positive')

    output_paths: list[str] = []
    compressed_paths: list[str] = []
    metadata_paths: list[str | None] = []
    failed_files: list[tuple[str, Exception]] = []

    with Manager() as lock_manager:
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
                    outputs, compressed, metadata, failures = future.result()
                    output_paths.extend(outputs)
                    compressed_paths.extend(compressed)
                    metadata_paths.extend(metadata)
                    failed_files.extend(failures)
                    progress.update(1)

    return output_paths, compressed_paths, metadata_paths, failed_files


def batch(
    filepaths: Sequence[str | Path],
    config: str | None = None,
    clean: bool = True,
) -> list[ClassificationResult]:
    """Classify multiple WAV files using one reusable ONNX session.

    ``clean`` remains for API compatibility; temporary spectrograms are always
    removed by the classifier.
    """
    del clean
    from batbot.classifier import Classifier

    classifier = Classifier(config=config)
    return [classifier.classify_wav(filepath) for filepath in filepaths]


def example() -> None:
    """Run the spectrogram pipeline on the packaged example WAV."""
    import pooch

    wav_filepath = PROJECT_ROOT / 'examples' / 'example1.wav'
    if not wav_filepath.exists():
        wav_filepath = Path(
            pooch.retrieve(
                url=(
                    'https://media.githubusercontent.com/media/Kitware/batbot/'
                    'main/examples/example1.wav'
                ),
                known_hash=(
                    'sha256:391efce5433d1057caddb4ce07b9712c523d6a815e4ee9e64b62973569982925'
                ),
                progressbar=True,
            )
        )

    log.debug('Running pipeline on WAV: %s', wav_filepath)
    output_stem = Path('output') / wav_filepath.stem
    start_time = time.time()
    results = pipeline(
        wav_filepath,
        out_file_stem=str(output_stem),
        fast_mode=False,
        force_overwrite=True,
        plot_uncompressed_amplitude=True,
        include_original_sr=True,
        time_buffer_ms=5.0,
    )
    print(f'Example pipeline completed in {time.time() - start_time} seconds.')
    log.debug(results)
