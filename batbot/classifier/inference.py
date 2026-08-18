"""Spectrogram preprocessing and ONNX inference."""

from __future__ import annotations

import json
import os
import warnings
from collections import deque
from collections.abc import Iterable, Iterator, MutableMapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np
import tqdm

from batbot._config import QUIET, log
from batbot.classifier.config import DEFAULT_CONFIG, resolve_config
from batbot.classifier.dataloader import BATCH_SIZE, INPUT_SIZE, ImageFilePathList, _init_transforms
from batbot.classifier.model import fetch
from batbot.classifier.types import (
    BulkClassification,
    ClassificationResult,
    ClassifierConfig,
    InputType,
    PathInput,
)


def _as_filepaths(inputs: PathInput | Iterable[PathInput]) -> list[str]:
    if isinstance(inputs, (str, os.PathLike)):
        return [str(inputs)]
    return [str(filepath) for filepath in inputs]


def pre(
    inputs: PathInput | Iterable[PathInput],
    batch_size: int = BATCH_SIZE,
    config: str | ClassifierConfig | None = DEFAULT_CONFIG,
) -> Iterator[tuple[np.ndarray[Any, Any], str]]:
    """Load spectrograms and yield their model-ready sliding windows."""
    selected = resolve_config(config)
    filepaths = _as_filepaths(inputs)
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')

    log.debug(
        'Preprocessing %d classifier inputs with inference batches of %d',
        len(filepaths),
        batch_size,
    )
    dataset = ImageFilePathList(filepaths, transform=_init_transforms())
    for index in range(len(dataset)):
        (data,) = dataset[index]
        yield data, selected.key


def _create_session(onnx_model: str, providers: Sequence[str] | None = None, reduced=False) -> Any:
    # Official non-Windows builds may otherwise create a persistent telemetry
    # device identifier as soon as ONNX Runtime initializes. Respect an
    # explicit user setting while making private inference the default.
    reduced = os.getenv('BATBOT_REDUCED', reduced) in [True, '1', 'Yes', 'yes', 'YES']

    os.environ.setdefault('ORT_DISABLE_TELEMETRY', '1')
    try:
        import onnxruntime as ort
    except ImportError as error:  # pragma: no cover - a declared runtime dependency
        raise ImportError(
            'ONNX inference requires onnxruntime; install batbot with its runtime dependencies'
        ) from error
    ort.disable_telemetry_events()

    selected_providers = providers
    if selected_providers is None:
        available = ort.get_available_providers()
        preferred = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        selected_providers = [provider for provider in preferred if provider in available]

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        if reduced:
            print('Reducing ONNX inference threads to 2')
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 2
        else:
            opts = None
        return ort.InferenceSession(onnx_model, providers=selected_providers, sess_options=opts)


def _validate_session(session: Any, config: ClassifierConfig) -> None:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or not outputs:
        raise ValueError('Classifier model must have one input and at least one output')

    input_shape = inputs[0].shape
    if list(input_shape[1:]) != [INPUT_SIZE, INPUT_SIZE, 3]:
        raise ValueError(f'Unexpected classifier input shape: {input_shape}')

    metadata = session.get_modelmeta().custom_metadata_map
    if 'labels' in metadata:
        mapping = json.loads(metadata['labels'])
        labels = [mapping['forward'][str(index)] for index in range(len(mapping['forward']))]
        if labels != list(config.classes):
            raise ValueError('Classifier labels do not match the selected configuration')


def _predict_windows(
    windows: np.ndarray[Any, Any],
    session: Any,
    batch_size: int,
) -> np.ndarray[Any, Any]:
    """Run one spectrogram's windows through a reusable ONNX session."""
    input_name = session.get_inputs()[0].name
    outputs = []
    for start in range(0, len(windows), batch_size):
        output = session.run(None, {input_name: windows[start : start + batch_size]})
        outputs.append(output[0])
    if not outputs:
        raise ValueError('Classifier preprocessing produced no image windows')
    return np.vstack(outputs).mean(axis=0, keepdims=True)


def predict(
    gen: Iterable[tuple[np.ndarray[Any, Any], str | ClassifierConfig]],
    batch_size: int = BATCH_SIZE,
    providers: Sequence[str] | None = None,
    pull: bool = False,
    sessions: MutableMapping[str, Any] | None = None,
    total: int | None = None,
    num_workers: int = 1,
) -> Iterator[tuple[np.ndarray[Any, Any], str]]:
    """Run ordered ONNX inference, optionally across multiple worker threads."""
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if num_workers <= 0:
        raise ValueError('num_workers must be positive')

    active_sessions = {} if sessions is None else sessions
    items = tqdm.tqdm(
        gen,
        disable=QUIET,
        desc='Classifying spectrograms',
        total=total,
    )

    def session_for(config_value: str | ClassifierConfig) -> tuple[Any, str]:
        config = resolve_config(config_value)
        session = active_sessions.get(config.key)
        if session is None:
            session = _create_session(fetch(pull=pull, config=config), providers=providers)
            _validate_session(session, config)
            active_sessions[config.key] = session
        return session, config.key

    if num_workers == 1:
        for windows, config_value in items:
            session, config_key = session_for(config_value)
            yield _predict_windows(windows, session, batch_size), config_key
        return

    pending: deque[tuple[Future[np.ndarray[Any, Any]], str]] = deque()
    with ThreadPoolExecutor(
        max_workers=num_workers,
        thread_name_prefix='batbot-onnx',
    ) as executor:
        for windows, config_value in items:
            session, config_key = session_for(config_value)
            pending.append(
                (executor.submit(_predict_windows, windows, session, batch_size), config_key)
            )
            if len(pending) >= num_workers:
                future, completed_config = pending.popleft()
                yield future.result(), completed_config

        while pending:
            future, completed_config = pending.popleft()
            yield future.result(), completed_config


def post(
    gen: Iterable[tuple[np.ndarray[Any, Any], str | ClassifierConfig]],
) -> list[dict[str, float]]:
    """Associate raw model scores with labels for each spectrogram."""
    outputs = []
    for predictions, config_value in gen:
        config = resolve_config(config_value)
        for prediction in predictions:
            if len(prediction) != len(config.classes):
                raise ValueError(
                    f'Model returned {len(prediction)} scores for {len(config.classes)} labels'
                )
            outputs.append(
                {class_name: float(score) for class_name, score in zip(config.classes, prediction)}
            )
    return outputs


def _format_result(
    filepath: PathInput,
    scores: dict[str, float],
    window_count: int,
    top_k: int = 5,
) -> ClassificationResult:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    selected_top_k = max(1, min(int(top_k), len(ranked)))
    return {
        'path': str(filepath),
        'label': ranked[0][0],
        'confidence': ranked[0][1],
        'window_count': int(window_count),
        'top': [
            {'label': label, 'confidence': confidence}
            for label, confidence in ranked[:selected_top_k]
        ],
        'scores': scores,
    }


class Classifier:
    """Reusable classifier that owns a lazily-created ONNX Runtime session."""

    def __init__(
        self,
        config: str | ClassifierConfig | None = DEFAULT_CONFIG,
        batch_size: int = BATCH_SIZE,
        providers: Sequence[str] | None = None,
        top_k: int = 5,
        pull: bool = False,
        sessions: MutableMapping[str, Any] | None = None,
        num_workers: int = 1,
    ) -> None:
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        if top_k <= 0:
            raise ValueError('top_k must be positive')
        if num_workers <= 0:
            raise ValueError('num_workers must be positive')
        self.config = resolve_config(config)
        self.batch_size = batch_size
        self.providers = tuple(providers) if providers is not None else None
        self.top_k = top_k
        self.pull = pull
        self._sessions = {} if sessions is None else sessions
        self.num_workers = num_workers

    @property
    def session(self) -> Any:
        """Return this classifier's validated, reusable ONNX session."""
        session = self._sessions.get(self.config.key)
        if session is None:
            session = _create_session(
                fetch(pull=self.pull, config=self.config),
                providers=self.providers,
            )
            _validate_session(session, self.config)
            self._sessions[self.config.key] = session
        return session

    def classify(
        self,
        inputs: PathInput | Iterable[PathInput],
        top_k: int | None = None,
    ) -> list[ClassificationResult]:
        """Classify one or more spectrogram images."""
        if top_k is not None and top_k <= 0:
            raise ValueError('top_k must be positive')
        filepaths = _as_filepaths(inputs)
        window_counts: list[int] = []

        def track_windows() -> Iterator[tuple[np.ndarray[Any, Any], str]]:
            for windows, selected_config in pre(
                filepaths,
                batch_size=self.batch_size,
                config=self.config,
            ):
                window_counts.append(len(windows))
                yield windows, selected_config

        scores = post(
            predict(
                track_windows(),
                batch_size=self.batch_size,
                providers=self.providers,
                pull=self.pull,
                sessions={self.config.key: self.session},
                total=len(filepaths),
                num_workers=self.num_workers,
            )
        )
        result_top_k = self.top_k if top_k is None else top_k
        return [
            _format_result(filepath, output, count, top_k=result_top_k)
            for filepath, output, count in zip(filepaths, scores, window_counts)
        ]

    def classify_wav(
        self,
        filepath: PathInput,
        top_k: int | None = None,
        output_folder: PathInput | None = None,
        out_file_stem: PathInput | None = None,
        keep_spectrograms: bool = False,
    ) -> ClassificationResult:
        """Generate spectrograms and classify a WAV using this session."""
        from batbot.classifier.bulk import classify_wav

        return classify_wav(
            filepath,
            top_k=self.top_k if top_k is None else top_k,
            output_folder=output_folder,
            out_file_stem=out_file_stem,
            keep_spectrograms=keep_spectrograms,
            _runner=self,
        )

    def classify_bulk(
        self,
        inputs: PathInput | Iterable[PathInput],
        input_type: InputType = 'auto',
        recursive: bool = True,
        top_k: int | None = None,
        spectrogram_output: PathInput | None = None,
    ) -> BulkClassification:
        """Classify a directory tree using this session."""
        from batbot.classifier.bulk import classify_bulk

        return classify_bulk(
            inputs,
            input_type=input_type,
            recursive=recursive,
            top_k=self.top_k if top_k is None else top_k,
            spectrogram_output=spectrogram_output,
            _runner=self,
        )


def classify(
    inputs: PathInput | Iterable[PathInput],
    batch_size: int = BATCH_SIZE,
    config: str | ClassifierConfig | None = DEFAULT_CONFIG,
    providers: Sequence[str] | None = None,
    top_k: int = 5,
    sessions: MutableMapping[str, Any] | None = None,
    num_workers: int = 1,
) -> list[ClassificationResult]:
    """Classify spectrograms with a one-shot or externally shared session."""
    return Classifier(
        config=config,
        batch_size=batch_size,
        providers=providers,
        top_k=top_k,
        sessions=sessions,
        num_workers=num_workers,
    ).classify(inputs)
