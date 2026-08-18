"""WAV and fault-tolerant bulk classification orchestration."""

from __future__ import annotations

import tempfile
from collections import Counter
from collections.abc import Iterable, MutableMapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any, TypeGuard

import numpy as np
import tqdm

from batbot._config import QUIET
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
    )
    filepath_string = str(filepath)
    if output_folder is not None or out_file_stem is not None or keep_spectrograms:
        selected_output = './output' if output_folder is None else str(output_folder)
        output_paths, _, _, _ = compute(
            filepath_string,
            output_folder=selected_output,
            out_file_stem=None if out_file_stem is None else str(out_file_stem),
            fast_mode=False,
            force_overwrite=True,
            quiet=True,
        )
        results = runner.classify(output_paths, top_k=top_k)
        return _aggregate_results(filepath_string, results, output_paths, top_k=top_k)

    with tempfile.TemporaryDirectory(prefix='batbot-classifier-') as temp_dir:
        output_paths, _, _, _ = compute(
            filepath_string,
            output_folder=temp_dir,
            fast_mode=False,
            force_overwrite=True,
            quiet=True,
        )
        results = runner.classify(output_paths, top_k=top_k)
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

    discovered = []
    for value in _as_filepaths(inputs):
        path = Path(value)
        if path.is_file():
            if path.suffix.lower() in extensions:
                discovered.append(str(path))
        elif path.is_dir():
            iterator = path.rglob('*') if recursive else path.glob('*')
            discovered.extend(
                str(candidate)
                for candidate in iterator
                if candidate.is_file() and candidate.suffix.lower() in extensions
            )
        else:
            raise FileNotFoundError(f'Input does not exist: {value}')
    return sorted(set(discovered))


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


def classify_bulk(
    inputs: PathInput | Iterable[PathInput],
    input_type: InputType = 'auto',
    recursive: bool = True,
    batch_size: int = BATCH_SIZE,
    config: str | ClassifierConfig | None = DEFAULT_CONFIG,
    providers: Sequence[str] | None = None,
    top_k: int = 5,
    spectrogram_output: PathInput | None = None,
    *,
    _runner: Classifier | None = None,
) -> BulkClassification:
    """Classify a directory tree while reusing a single ONNX session."""
    paths = discover_inputs(inputs, input_type=input_type, recursive=recursive)
    runner = _runner or Classifier(
        config=config,
        batch_size=batch_size,
        providers=providers,
        top_k=top_k,
    )
    results_by_path: dict[str, ClassificationItem] = {}

    image_paths = [path for path in paths if Path(path).suffix.lower() in SPECTROGRAM_EXTENSIONS]
    if image_paths:
        try:
            for result in runner.classify(image_paths, top_k=top_k):
                results_by_path[result['path']] = result
        except (OSError, ValueError):
            for path in image_paths:
                try:
                    results_by_path[path] = runner.classify([path], top_k=top_k)[0]
                except Exception as error:  # pragma: no cover - decoder errors vary by platform
                    results_by_path[path] = {'path': path, 'error': str(error)}

    wav_paths = [path for path in paths if Path(path).suffix.lower() in WAV_EXTENSIONS]
    for path in tqdm.tqdm(wav_paths, disable=QUIET, desc='Classifying WAV files'):
        try:
            out_file_stem = None
            if spectrogram_output is not None:
                digest = sha256(str(Path(path).resolve()).encode('utf8')).hexdigest()[:10]
                output_name = f'{Path(path).stem}.{digest}'
                out_file_stem = str(Path(spectrogram_output) / output_name)
            results_by_path[path] = classify_wav(
                path,
                top_k=top_k,
                output_folder=spectrogram_output,
                out_file_stem=out_file_stem,
                keep_spectrograms=spectrogram_output is not None,
                _runner=runner,
            )
        except Exception as error:  # pragma: no cover - spectrogram errors vary by input
            results_by_path[path] = {'path': path, 'error': str(error)}

    results = [results_by_path[path] for path in paths]
    return {'results': results, 'summary': summarize(results)}
