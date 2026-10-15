"""WAV and fault-tolerant bulk classification orchestration."""

from __future__ import annotations

import tempfile
from collections import Counter
from collections.abc import Iterable, MutableMapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path
from typing import Any, TypeGuard

import numpy as np
import tqdm

from batbot._paths import discover_files, native_path
from batbot.classifier.config import (
    DEFAULT_CONFIG,
    SPECTROGRAM_EXTENSIONS,
    WAV_EXTENSIONS,
)
from batbot.classifier.dataloader import BATCH_SIZE
from batbot.classifier.inference import Classifier, _as_filepaths, _format_result
from batbot.classifier.types import (
    BulkClassification,
    ClassificationFailure,
    ClassificationItem,
    ClassificationResult,
    ClassificationSummary,
    ClassifierConfig,
    InputType,
    PathInput,
)


def _is_success(result: ClassificationItem) -> TypeGuard[ClassificationResult]:
    return 'label' in result


def _is_failure(result: ClassificationItem) -> TypeGuard[ClassificationFailure]:
    return 'error' in result


def _aggregate_results(
    filepath: PathInput,
    results: Sequence[ClassificationResult],
    spectrogram_paths: Sequence[PathInput],
    top_k: int = 5,
) -> ClassificationResult:
    if not results:
        raise ValueError(f'No spectrograms were created for {filepath}')

    weights = np.asarray([result['window_count'] for result in results], dtype=np.float64)
    classes = list(results[0]['scores'])
    values = np.asarray(
        [[result['scores'][class_name] for class_name in classes] for result in results]
    )
    scores = {
        class_name: float(score)
        for class_name, score in zip(classes, np.average(values, axis=0, weights=weights))
    }
    output = _format_result(filepath, scores, int(weights.sum()), top_k=top_k)
    output['spectrogram_paths'] = [str(path) for path in spectrogram_paths]
    return output


def classify_wav(
    filepath: PathInput,
    batch_size: int = BATCH_SIZE,
    config: str | ClassifierConfig | None = DEFAULT_CONFIG,
    providers: Sequence[str] | None = None,
    top_k: int = 5,
    output_folder: PathInput | None = None,
    out_file_stem: PathInput | None = None,
    keep_spectrograms: bool = False,
    sessions: MutableMapping[str, Any] | None = None,
    num_workers: int = 1,
    progress: bool = True,
    *,
    _runner: Classifier | None = None,
) -> ClassificationResult:
    """Generate spectrograms for a WAV file and return one prediction."""
    from batbot.spectrogram import compute

    runner = _runner or Classifier(
        config=config,
        batch_size=batch_size,
        providers=providers,
        top_k=top_k,
        sessions=sessions,
        num_workers=num_workers,
    )
    filepath_string = str(native_path(filepath))
    if output_folder is not None or out_file_stem is not None or keep_spectrograms:
        selected_output = './output' if output_folder is None else str(native_path(output_folder))
        output_paths, _, _, _ = compute(
            filepath_string,
            output_folder=selected_output,
            out_file_stem=None if out_file_stem is None else str(native_path(out_file_stem)),
            fast_mode=False,
            force_overwrite=True,
            quiet=True,
        )
        results = runner.classify(output_paths, top_k=top_k, progress=progress)
        return _aggregate_results(filepath_string, results, output_paths, top_k=top_k)

    with tempfile.TemporaryDirectory(prefix='batbot-classifier-') as temp_dir:
        output_paths, _, _, _ = compute(
            filepath_string,
            output_folder=temp_dir,
            fast_mode=False,
            force_overwrite=True,
            quiet=True,
        )
        results = runner.classify(output_paths, top_k=top_k, progress=progress)
        output = _aggregate_results(filepath_string, results, output_paths, top_k=top_k)
        output['spectrogram_paths'] = []
        return output


def discover_inputs(
    inputs: PathInput | Iterable[PathInput],
    input_type: InputType = 'auto',
    recursive: bool = True,
) -> list[str]:
    """Resolve files and directories into deterministic classifier inputs."""
    if input_type not in {'auto', 'spectrogram', 'wav'}:
        raise ValueError('input_type must be auto, spectrogram, or wav')

    extensions = SPECTROGRAM_EXTENSIONS | WAV_EXTENSIONS
    if input_type == 'spectrogram':
        extensions = SPECTROGRAM_EXTENSIONS
    elif input_type == 'wav':
        extensions = WAV_EXTENSIONS

    return discover_files(_as_filepaths(inputs), extensions, recursive=recursive)


def summarize(results: Sequence[ClassificationItem]) -> ClassificationSummary:
    """Generate species counts and basic confidence statistics."""
    successful = [result for result in results if _is_success(result)]
    failures = [result for result in results if _is_failure(result)]
    counts = Counter(result['label'] for result in successful)
    species_counts = {label: count for label, count in counts.items() if label != 'NOISE'}
    confidences = [result['confidence'] for result in successful]
    return {
        'total': len(results),
        'classified': len(successful),
        'failed': len(failures),
        'label_counts': dict(sorted(counts.items())),
        'species_counts': dict(sorted(species_counts.items())),
        'noise_count': counts.get('NOISE', 0),
        'mean_confidence': float(np.mean(confidences)) if confidences else None,
    }


def _prepare_wav(filepath: str, output_folder: str) -> list[str]:
    """Use the spectrogram pipeline in an isolated preprocessing worker."""
    from batbot.api import pipeline

    digest = sha256(str(Path(filepath).resolve()).encode('utf8')).hexdigest()[:10]
    output_stem = Path(output_folder) / f'{Path(filepath).stem}.{digest}'
    outputs, _, _ = pipeline(
        filepath,
        output_folder=output_folder,
        out_file_stem=str(output_stem),
        fast_mode=False,
        force_overwrite=True,
        quiet=True,
    )
    if not outputs:
        raise ValueError(f'No spectrograms were created for {filepath}')
    return outputs


def _classify_spectrograms(
    paths: list[str], runner: Classifier, top_k: int
) -> dict[str, ClassificationItem]:
    """Batch image inference, isolating unreadable inputs when necessary."""
    if not paths:
        return {}
    try:
        return {
            result['path']: result for result in runner.classify(paths, top_k=top_k, progress=False)
        }
    except (OSError, ValueError):
        results: dict[str, ClassificationItem] = {}
        for path in paths:
            try:
                results[path] = runner.classify([path], top_k=top_k, progress=False)[0]
            except Exception as error:
                results[path] = {'path': path, 'error': str(error)}
        return results


def classify_bulk(
    inputs: PathInput | Iterable[PathInput],
    input_type: InputType = 'auto',
    recursive: bool = True,
    batch_size: int = BATCH_SIZE,
    config: str | ClassifierConfig | None = DEFAULT_CONFIG,
    providers: Sequence[str] | None = None,
    top_k: int = 5,
    spectrogram_output: PathInput | None = None,
    num_workers: int = 1,
    preprocess_workers: int = 1,
    progress: bool = True,
    *,
    _runner: Classifier | None = None,
) -> BulkClassification:
    """Preprocess WAVs in bounded batches and reuse one ONNX inference session.

    ``preprocess_workers`` controls separate spectrogram processes, while
    ``num_workers`` controls inference threads. Progress is written to stderr
    and can be disabled independently of BatBot's logging settings.
    """
    if preprocess_workers <= 0:
        raise ValueError('preprocess_workers must be positive')
    paths = discover_inputs(inputs, input_type=input_type, recursive=recursive)
    runner = _runner or Classifier(
        config=config,
        batch_size=batch_size,
        providers=providers,
        top_k=top_k,
        num_workers=num_workers,
    )
    results_by_path: dict[str, ClassificationItem] = {}

    image_paths = [path for path in paths if Path(path).suffix.lower() in SPECTROGRAM_EXTENSIONS]
    wav_paths = [path for path in paths if Path(path).suffix.lower() in WAV_EXTENSIONS]
    chunk_size = max(8, preprocess_workers, runner.num_workers)
    with ExitStack() as stack:
        preprocessing = stack.enter_context(
            tqdm.tqdm(
                total=len(wav_paths),
                desc='Creating spectrograms',
                unit='file',
                disable=not progress or not wav_paths,
                position=0,
            )
        )
        classification = stack.enter_context(
            tqdm.tqdm(
                total=len(paths),
                desc='Classifying files',
                unit='file',
                disable=not progress or not paths,
                position=1 if wav_paths else 0,
            )
        )
        executor = None
        if wav_paths and preprocess_workers > 1:
            # Spawn avoids inheriting ONNX Runtime threads and GPU state.
            executor = stack.enter_context(
                ProcessPoolExecutor(
                    max_workers=preprocess_workers,
                    mp_context=get_context('spawn'),
                )
            )

        for start in range(0, len(image_paths), chunk_size):
            batch_paths = image_paths[start : start + chunk_size]
            results_by_path.update(_classify_spectrograms(batch_paths, runner, top_k))
            classification.update(len(batch_paths))

        for start in range(0, len(wav_paths), chunk_size):
            batch_paths = wav_paths[start : start + chunk_size]
            with ExitStack() as batch_stack:
                output_folder = (
                    str(native_path(spectrogram_output))
                    if spectrogram_output is not None
                    else batch_stack.enter_context(
                        tempfile.TemporaryDirectory(prefix='batbot-classifier-')
                    )
                )
                # Create the shared destination before workers enter the
                # spectrogram pipeline, whose directory creation is not atomic.
                Path(output_folder).mkdir(parents=True, exist_ok=True)
                prepared: dict[str, list[str]] = {}
                if executor is None:
                    for path in batch_paths:
                        try:
                            prepared[path] = _prepare_wav(path, output_folder)
                        except Exception as error:
                            results_by_path[path] = {'path': path, 'error': str(error)}
                        preprocessing.update(1)
                else:
                    futures = {
                        executor.submit(_prepare_wav, path, output_folder): path
                        for path in batch_paths
                    }
                    for future in as_completed(futures):
                        path = futures[future]
                        try:
                            prepared[path] = future.result()
                        except Exception as error:
                            results_by_path[path] = {'path': path, 'error': str(error)}
                        preprocessing.update(1)

                generated = [image for path in batch_paths for image in prepared.get(path, [])]
                predictions = _classify_spectrograms(generated, runner, top_k)
                for path in batch_paths:
                    if path in prepared:
                        images = prepared[path]
                        per_image = [predictions[image] for image in images]
                        failures = [result for result in per_image if _is_failure(result)]
                        if failures:
                            results_by_path[path] = {'path': path, 'error': failures[0]['error']}
                        else:
                            results_by_path[path] = _aggregate_results(
                                path,
                                [result for result in per_image if _is_success(result)],
                                images if spectrogram_output is not None else [],
                                top_k=top_k,
                            )
                    classification.update(1)

    results = [results_by_path[path] for path in paths]
    return {'results': results, 'summary': summarize(results)}
