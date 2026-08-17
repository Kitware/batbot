"""Classify BatBot spectrograms with the bundled MobileNet ONNX model.

The low-level API follows Scoutbot's whole-image classifier pipeline:
:func:`pre` loads and windows images, :func:`predict` performs ONNX inference,
and :func:`post` associates model scores with species labels.  Most callers can
use :func:`classify`, :func:`classify_wav`, or :func:`classify_bulk` directly.
"""

import json
import os
import tempfile
import warnings
from collections import Counter
from hashlib import sha256
from os.path import exists, join
from pathlib import Path

import numpy as np
import pooch
import tqdm

from batbot import QUIET, log
from batbot.classifier.dataloader import (  # NOQA
    BATCH_SIZE,
    INPUT_SIZE,
    ImageFilePathList,
    _init_transforms,
)

PWD = Path(__file__).absolute().parent
DEFAULT_CONFIG = os.getenv('CLASSIFIER_CONFIG', os.getenv('CONFIG', 'mobilenet')).strip().lower()
MODEL_NAME = 'batbot.mobilenet.9dc57ea3.onnx'
MODEL_URL = 'https://data.kitware.com/api/v1/file/6a8377e32688ba21262c3907/download'
MODEL_HASH = '351aa656ce5df717472d3c1cda7cef883b321d3bbb382740e16b3f2a1d74f621'
CLASSES = [
    'ANPA',
    'CORA',
    'COTO',
    'EPFU',
    'EUFL',
    'EUMA',
    'EUPE',
    'IDPH',
    'LABL',
    'LABO',
    'LACI',
    'LAIN',
    'LANO',
    'LASE',
    'LAXA',
    'MYAU',
    'MYCA',
    'MYCI',
    'MYEV',
    'MYGR',
    'MYLE',
    'MYLU',
    'MYSE',
    'MYSO',
    'MYTH',
    'MYVE',
    'MYVO',
    'MYYU',
    'NOISE',
    'NYFE',
    'NYHU',
    'NYMA',
    'PAHE',
    'PESU',
    'TABR',
]

CONFIGS = {
    'mobilenet': {
        'name': MODEL_NAME,
        'path': join(PWD, 'models', 'onnx', MODEL_NAME),
        'url': MODEL_URL,
        'hash': MODEL_HASH,
        'classes': CLASSES,
    }
}
if DEFAULT_CONFIG not in CONFIGS:
    raise ValueError('Unknown classifier configuration: {}'.format(DEFAULT_CONFIG))
CONFIGS[None] = CONFIGS[DEFAULT_CONFIG]

SPECTROGRAM_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
WAV_EXTENSIONS = {'.wav'}


def _resolve_config(config):
    config = DEFAULT_CONFIG if config is None else str(config).strip().lower()
    if config not in CONFIGS:
        raise ValueError(
            'Unknown classifier configuration {!r}; choose from {}'.format(
                config, ', '.join(sorted(key for key in CONFIGS if key is not None))
            )
        )
    return config


def _as_filepaths(inputs):
    if isinstance(inputs, (str, Path)):
        return [str(inputs)]
    return [str(filepath) for filepath in inputs]


def fetch(pull=False, config=DEFAULT_CONFIG):
    """Return a local classifier model, downloading the mirror when needed.

    Args:
        pull (bool): Download and use the cached mirror even when the package
            contains a model.
        config (str or None): Classifier configuration name.

    Returns:
        str: Path to the local ONNX model.
    """
    config = _resolve_config(config)
    model = CONFIGS[config]

    if not pull and exists(model['path']):
        onnx_model = model['path']
    else:
        onnx_model = pooch.retrieve(
            url=model['url'],
            known_hash='sha256:{}'.format(model['hash']),
            fname=model['name'],
            progressbar=not QUIET,
        )
        if not exists(onnx_model):  # nocov - pooch raises first in normal failures
            raise OSError('Classifier model could not be fetched')

    log.debug('Classifier model: {}'.format(onnx_model))
    return onnx_model


def pre(inputs, batch_size=BATCH_SIZE, config=DEFAULT_CONFIG):
    """Load spectrograms and yield their model-ready sliding windows.

    Each yielded array has shape ``(windows, 224, 224, 3)`` and dtype
    ``uint8``.  ``batch_size`` is validated here for API compatibility and is
    used by :func:`predict` to bound each ONNX call.
    """
    config = _resolve_config(config)
    inputs = _as_filepaths(inputs)
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')

    log.debug(
        'Preprocessing {} classifier inputs with inference batches of {}'.format(
            len(inputs), batch_size
        )
    )
    transform = _init_transforms()
    dataset = ImageFilePathList(inputs, transform=transform)
    for (data,) in dataset:
        yield data, config


def _create_session(onnx_model, providers=None):
    try:
        import onnxruntime as ort
    except ImportError as error:  # pragma: no cover - dependency is installed in package tests
        raise ImportError(
            'ONNX inference requires onnxruntime; install batbot with its runtime dependencies'
        ) from error

    if providers is None:
        available = ort.get_available_providers()
        preferred = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        providers = [provider for provider in preferred if provider in available]

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        return ort.InferenceSession(onnx_model, providers=providers)


def _validate_session(session, config):
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) < 1:
        raise ValueError('Classifier model must have one input and at least one output')

    input_shape = inputs[0].shape
    if list(input_shape[1:]) != [INPUT_SIZE, INPUT_SIZE, 3]:
        raise ValueError('Unexpected classifier input shape: {}'.format(input_shape))

    metadata = session.get_modelmeta().custom_metadata_map
    if 'labels' in metadata:
        mapping = json.loads(metadata['labels'])
        labels = [mapping['forward'][str(index)] for index in range(len(mapping['forward']))]
        if labels != CONFIGS[config]['classes']:
            raise ValueError('Classifier labels do not match the selected configuration')


def predict(gen, batch_size=BATCH_SIZE, providers=None, pull=False, sessions=None):
    """Run ONNX inference and average all windows from each spectrogram."""
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')

    if sessions is None:
        sessions = {}
    for windows, config in tqdm.tqdm(gen, disable=QUIET, desc='Classifying spectrograms'):
        config = _resolve_config(config)
        session = sessions.get(config)
        if session is None:
            session = _create_session(fetch(pull=pull, config=config), providers=providers)
            _validate_session(session, config)
            sessions[config] = session

        if len(windows) == 0:
            predictions = np.empty((0, len(CONFIGS[config]['classes'])), dtype=np.float32)
        else:
            input_name = session.get_inputs()[0].name
            outputs = []
            for start in range(0, len(windows), batch_size):
                output = session.run(None, {input_name: windows[start : start + batch_size]})
                outputs.append(output[0])
            predictions = np.vstack(outputs).mean(axis=0, keepdims=True)
        yield predictions, config


def post(gen):
    """Associate raw model scores with labels for each spectrogram."""
    outputs = []
    for predictions, config in gen:
        config = _resolve_config(config)
        classes = CONFIGS[config]['classes']
        for prediction in predictions:
            if len(prediction) != len(classes):
                raise ValueError(
                    'Model returned {} scores for {} labels'.format(len(prediction), len(classes))
                )
            outputs.append(
                {class_name: float(score) for class_name, score in zip(classes, prediction)}
            )
    return outputs


def _format_result(filepath, scores, window_count, top_k=5):
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_k = max(1, min(int(top_k), len(ranked)))
    return {
        'path': str(filepath),
        'label': ranked[0][0],
        'confidence': ranked[0][1],
        'window_count': int(window_count),
        'top': [{'label': label, 'confidence': confidence} for label, confidence in ranked[:top_k]],
        'scores': scores,
    }


def classify(
    inputs,
    batch_size=BATCH_SIZE,
    config=DEFAULT_CONFIG,
    providers=None,
    top_k=5,
    sessions=None,
):
    """Classify one or more spectrogram image files."""
    inputs = _as_filepaths(inputs)
    window_counts = []

    def track_windows():
        for windows, selected_config in pre(inputs, batch_size=batch_size, config=config):
            window_counts.append(len(windows))
            yield windows, selected_config

    scores = post(
        predict(
            track_windows(),
            batch_size=batch_size,
            providers=providers,
            sessions=sessions,
        )
    )
    return [
        _format_result(filepath, output, count, top_k=top_k)
        for filepath, output, count in zip(inputs, scores, window_counts)
    ]


def _aggregate_results(filepath, results, spectrogram_paths, top_k=5):
    if not results:
        raise ValueError('No spectrograms were created for {}'.format(filepath))

    weights = np.asarray([result['window_count'] for result in results], dtype=np.float64)
    classes = list(results[0]['scores'])
    values = np.asarray(
        [[result['scores'][class_name] for class_name in classes] for result in results]
    )
    averaged = np.average(values, axis=0, weights=weights)
    scores = {class_name: float(score) for class_name, score in zip(classes, averaged)}
    output = _format_result(filepath, scores, int(weights.sum()), top_k=top_k)
    output['spectrogram_paths'] = [str(path) for path in spectrogram_paths]
    return output


def classify_wav(
    filepath,
    batch_size=BATCH_SIZE,
    config=DEFAULT_CONFIG,
    providers=None,
    top_k=5,
    output_folder=None,
    out_file_stem=None,
    keep_spectrograms=False,
    sessions=None,
):
    """Generate spectrograms for a WAV file and return one aggregate prediction."""
    from batbot import spectrogram

    filepath = str(filepath)
    if output_folder is not None or out_file_stem is not None or keep_spectrograms:
        output_folder = './output' if output_folder is None else str(output_folder)
        output_paths, _, _, _ = spectrogram.compute(
            filepath,
            output_folder=output_folder,
            out_file_stem=out_file_stem,
            fast_mode=False,
            force_overwrite=True,
            quiet=True,
        )
        results = classify(
            output_paths,
            batch_size=batch_size,
            config=config,
            providers=providers,
            top_k=top_k,
            sessions=sessions,
        )
        return _aggregate_results(filepath, results, output_paths, top_k=top_k)

    with tempfile.TemporaryDirectory(prefix='batbot-classifier-') as temp_dir:
        output_paths, _, _, _ = spectrogram.compute(
            filepath,
            output_folder=temp_dir,
            fast_mode=False,
            force_overwrite=True,
            quiet=True,
        )
        results = classify(
            output_paths,
            batch_size=batch_size,
            config=config,
            providers=providers,
            top_k=top_k,
            sessions=sessions,
        )
        output = _aggregate_results(filepath, results, output_paths, top_k=top_k)
        # Temporary paths cease to be useful as soon as this function returns.
        output['spectrogram_paths'] = []
        return output


def discover_inputs(inputs, input_type='auto', recursive=True):
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
            raise FileNotFoundError('Input does not exist: {}'.format(value))
    return sorted(set(discovered))


def summarize(results):
    """Generate species counts and basic confidence statistics."""
    successful = [result for result in results if 'label' in result]
    failures = [result for result in results if 'error' in result]
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
    inputs,
    input_type='auto',
    recursive=True,
    batch_size=BATCH_SIZE,
    config=DEFAULT_CONFIG,
    providers=None,
    top_k=5,
    spectrogram_output=None,
):
    """Classify a directory tree without retaining all images in memory."""
    paths = discover_inputs(inputs, input_type=input_type, recursive=recursive)
    results_by_path = {}
    sessions = {}

    image_paths = [path for path in paths if Path(path).suffix.lower() in SPECTROGRAM_EXTENSIONS]
    if image_paths:
        try:
            for result in classify(
                image_paths,
                batch_size=batch_size,
                config=config,
                providers=providers,
                top_k=top_k,
                sessions=sessions,
            ):
                results_by_path[result['path']] = result
        except Exception:
            # Retry independently so one unreadable image does not discard a
            # large folder's otherwise valid classifications.
            for path in image_paths:
                try:
                    results_by_path[path] = classify(
                        [path],
                        batch_size=batch_size,
                        config=config,
                        providers=providers,
                        top_k=top_k,
                        sessions=sessions,
                    )[0]
                except Exception as error:  # pragma: no cover - exact decoder errors vary
                    results_by_path[path] = {'path': path, 'error': str(error)}

    wav_paths = [path for path in paths if Path(path).suffix.lower() in WAV_EXTENSIONS]
    for path in tqdm.tqdm(wav_paths, disable=QUIET, desc='Classifying WAV files'):
        try:
            out_file_stem = None
            if spectrogram_output is not None:
                digest = sha256(str(Path(path).resolve()).encode('utf8')).hexdigest()[:10]
                output_name = '{}.{}'.format(Path(path).stem, digest)
                out_file_stem = str(Path(spectrogram_output) / output_name)
            results_by_path[path] = classify_wav(
                path,
                batch_size=batch_size,
                config=config,
                providers=providers,
                top_k=top_k,
                output_folder=spectrogram_output,
                out_file_stem=out_file_stem,
                keep_spectrograms=spectrogram_output is not None,
                sessions=sessions,
            )
        except Exception as error:
            results_by_path[path] = {'path': path, 'error': str(error)}

    results = [results_by_path[path] for path in paths]
    return {'results': results, 'summary': summarize(results)}
