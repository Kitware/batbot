"""Classifier model declarations and configuration resolution."""

from __future__ import annotations

import os
from collections.abc import Mapping
from types import MappingProxyType

from batbot.classifier.types import ClassifierConfig

MODEL_NAME = 'batbot.mobilenet.9dc57ea3.onnx'
MODEL_URL = 'https://data.kitware.com/api/v1/file/6a8377e32688ba21262c3907/download'
MODEL_HASH = '351aa656ce5df717472d3c1cda7cef883b321d3bbb382740e16b3f2a1d74f621'

CLASSES = (
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
)

MOBILENET = ClassifierConfig(
    key='mobilenet',
    filename=MODEL_NAME,
    url=MODEL_URL,
    sha256=MODEL_HASH,
    classes=CLASSES,
)

_CONFIGS: dict[str | None, ClassifierConfig] = {
    'mobilenet': MOBILENET,
    None: MOBILENET,
}
CONFIGS: Mapping[str | None, ClassifierConfig] = MappingProxyType(_CONFIGS)

DEFAULT_CONFIG = (
    os.getenv(
        'BATBOT_CLASSIFIER_CONFIG',
        os.getenv('CLASSIFIER_CONFIG', 'mobilenet'),
    )
    .strip()
    .lower()
)
if DEFAULT_CONFIG not in CONFIGS:
    raise ValueError(f'Unknown classifier configuration: {DEFAULT_CONFIG}')

SPECTROGRAM_EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png', '.tif', '.tiff'})
WAV_EXTENSIONS = frozenset({'.wav'})


def resolve_config(config: str | ClassifierConfig | None = None) -> ClassifierConfig:
    """Resolve a configuration name or return an existing configuration."""
    if isinstance(config, ClassifierConfig):
        return config
    key = DEFAULT_CONFIG if config is None else str(config).strip().lower()
    try:
        return CONFIGS[key]
    except KeyError as error:
        choices = ', '.join(sorted(key for key in CONFIGS if key is not None))
        raise ValueError(
            f'Unknown classifier configuration {key!r}; choose from {choices}'
        ) from error
