"""Classify BatBot spectrograms with the MobileNet ONNX model.

The low-level :func:`pre`, :func:`predict`, and :func:`post` functions retain
the Scoutbot-style pipeline.  :class:`Classifier` is the preferred API for
repeated work because it owns and reuses its ONNX Runtime session.
"""

from batbot.classifier.bulk import classify_bulk, classify_wav, discover_inputs, summarize
from batbot.classifier.config import (
    CLASSES,
    CONFIGS,
    DEFAULT_CONFIG,
    MODEL_HASH,
    MODEL_NAME,
    MODEL_URL,
    SPECTROGRAM_EXTENSIONS,
    WAV_EXTENSIONS,
    resolve_config,
)
from batbot.classifier.dataloader import BATCH_SIZE, INPUT_SIZE, ImageFilePathList
from batbot.classifier.inference import Classifier, classify, post, pre, predict
from batbot.classifier.model import fetch
from batbot.classifier.types import (
    BulkClassification,
    ClassificationFailure,
    ClassificationItem,
    ClassificationResult,
    ClassificationSummary,
    ClassifierConfig,
    TopPrediction,
)

__all__ = [
    'BATCH_SIZE',
    'CLASSES',
    'CONFIGS',
    'DEFAULT_CONFIG',
    'INPUT_SIZE',
    'MODEL_HASH',
    'MODEL_NAME',
    'MODEL_URL',
    'SPECTROGRAM_EXTENSIONS',
    'WAV_EXTENSIONS',
    'BulkClassification',
    'ClassificationFailure',
    'ClassificationItem',
    'ClassificationResult',
    'ClassificationSummary',
    'Classifier',
    'ClassifierConfig',
    'ImageFilePathList',
    'TopPrediction',
    'classify',
    'classify_bulk',
    'classify_wav',
    'discover_inputs',
    'fetch',
    'post',
    'pre',
    'predict',
    'resolve_config',
    'summarize',
]
