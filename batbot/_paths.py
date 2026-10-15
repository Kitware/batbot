"""Native filesystem paths and Unicode-safe image I/O on every platform."""

from __future__ import annotations

import os
from collections.abc import Iterable
from glob import escape, glob
from hashlib import sha256
from pathlib import Path
from typing import Any


def native_path(value: str | os.PathLike[str]) -> Path:
    """Expand home/environment variables while retaining native path semantics."""
    return Path(os.path.expandvars(os.path.expanduser(os.fspath(value))))


def cache_root() -> Path:
    """Return a writable, platform-native cache root shared by models/examples."""
    import pooch

    return native_path(os.getenv('BATBOT_CACHE_DIR') or pooch.os_cache('batbot'))


def discover_files(
    inputs: Iterable[str | os.PathLike[str]],
    extensions: set[str],
    recursive: bool = True,
    patterns: bool = False,
) -> list[str]:
    """Discover files case-insensitively, preserving literal existing paths."""
    found: set[str] = set()
    for value in inputs:
        path = native_path(value)
        candidates = [path]
        if not path.exists():
            if not patterns:
                raise FileNotFoundError(f'Input does not exist: {value}')
            # Existing directory names may themselves contain glob characters.
            prefix = path.parent
            while not prefix.is_dir() and prefix.parent != prefix:
                prefix = prefix.parent
            pattern = str(Path(escape(str(prefix))) / path.relative_to(prefix))
            candidates = [Path(match) for match in glob(pattern, recursive=recursive)]
        for candidate in candidates:
            if candidate.is_dir():
                iterator = candidate.rglob('*') if recursive else candidate.glob('*')
            else:
                iterator = [candidate]
            found.update(
                str(item)
                for item in iterator
                if item.is_file() and item.suffix.lower() in extensions
            )
    return sorted(found)


def output_stems(
    inputs: Iterable[str | os.PathLike[str]],
    output_folder: str | os.PathLike[str],
    flatten: bool = False,
) -> list[str]:
    """Map inputs below an output root, including inputs on different drives."""
    paths = [native_path(value).absolute() for value in inputs]
    if not paths:
        return []
    try:
        common_root = Path(os.path.commonpath([path.parent for path in paths]))
    except ValueError:
        common_root = None  # Windows drives and UNC shares need separate namespaces.
    output_root = native_path(output_folder)
    stems = []
    for path in paths:
        if flatten:
            relative = Path(path.name)
        elif common_root is not None:
            relative = path.relative_to(common_root)
        else:
            digest = sha256(str(path.parent).encode('utf8')).hexdigest()[:10]
            relative = Path(f'input-{digest}') / path.name
        stems.append(str(output_root / relative.with_suffix('')))
    if len(set(map(os.path.normcase, stems))) != len(stems):
        raise ValueError('Input filenames would overwrite each other in the output directory')
    return stems


def read_image(filepath: str | os.PathLike[str]) -> Any:
    """Decode bytes using Python's Unicode-aware filesystem handling."""
    import cv2
    import numpy as np

    path = native_path(filepath)
    try:
        data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except (OSError, cv2.error) as error:
        raise OSError(f'Unable to load spectrogram: {path}') from error
    if image is None:
        raise OSError(f'Unable to load spectrogram: {path}')
    return image


def write_image(filepath: str | os.PathLike[str], image: Any, params=None) -> bool:
    """Encode before writing so Windows paths never pass through OpenCV fopen."""
    import cv2

    path = native_path(filepath)
    try:
        success, encoded = cv2.imencode(path.suffix, image, params or [])
        if not success:
            raise OSError(f'Unable to encode image: {path}')
        path.write_bytes(encoded.tobytes())
    except (OSError, cv2.error) as error:
        raise OSError(f'Unable to write image: {path}') from error
    return True
