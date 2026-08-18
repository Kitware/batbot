"""Integrity-checked access to bundled and mirrored ONNX models."""

from __future__ import annotations

import os
import shutil
import tempfile
from functools import lru_cache
from importlib.resources import as_file, files
from pathlib import Path

import pooch

from batbot._config import QUIET, log
from batbot._version import __version__
from batbot.classifier.config import DEFAULT_CONFIG, resolve_config
from batbot.classifier.types import ClassifierConfig


def _cache_directory() -> Path:
    """Return BatBot's versioned model cache directory."""
    return Path(pooch.os_cache('batbot')) / 'models' / __version__


def _has_expected_hash(path: Path, config: ClassifierConfig) -> bool:
    return pooch.file_hash(str(path), alg='sha256') == config.sha256


def _materialize_resource(source: Path, config: ClassifierConfig) -> Path:
    """Copy a resource extracted from a non-filesystem loader into the cache."""
    target = _cache_directory() / config.filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and _has_expected_hash(target, config):
        return target

    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shutil.copyfile(source, temporary_path)
        if not _has_expected_hash(temporary_path, config):
            raise ValueError(f'Bundled classifier model failed its checksum: {config.filename}')
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)
    return target


def _bundled_model(config: ClassifierConfig) -> Path | None:
    resource = files(config.resource_package).joinpath(*config.resource_parts, config.filename)
    if not resource.is_file():
        return None

    with as_file(resource) as candidate:
        candidate = Path(candidate)
        if not _has_expected_hash(candidate, config):
            log.warning('Bundled classifier model failed checksum; using the mirror')
            return None
        if isinstance(resource, Path):
            return candidate
        return _materialize_resource(candidate, config)


def _download_model(config: ClassifierConfig) -> Path:
    downloaded = pooch.retrieve(
        url=config.url,
        known_hash=f'sha256:{config.sha256}',
        path=_cache_directory(),
        fname=config.filename,
        progressbar=not QUIET,
    )
    return Path(downloaded)


@lru_cache(maxsize=None)
def _fetch_cached(pull: bool, config: ClassifierConfig) -> str:
    model = None if pull else _bundled_model(config)
    if model is None:
        model = _download_model(config)
    if not model.is_file():  # pragma: no cover - Pooch raises before this in normal failures
        raise OSError('Classifier model could not be fetched')
    log.debug('Classifier model: %s', model)
    return str(model)


def fetch(
    pull: bool = False,
    config: str | ClassifierConfig | None = DEFAULT_CONFIG,
) -> str:
    """Return a verified local model, downloading the mirror when necessary."""
    return _fetch_cached(pull, resolve_config(config))
