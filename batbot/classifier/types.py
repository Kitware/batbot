"""Public type definitions for classifier configuration and results."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Literal, NotRequired, TypeAlias, TypedDict

PathInput: TypeAlias = str | PathLike[str]
InputType: TypeAlias = Literal['auto', 'spectrogram', 'wav']


@dataclass(frozen=True, slots=True)
class ClassifierConfig:
    """Immutable description of an ONNX classifier model."""

    key: str
    filename: str
    url: str
    sha256: str
    classes: tuple[str, ...]
    resource_package: str = 'batbot.classifier'
    resource_parts: tuple[str, ...] = ('models', 'onnx')


class TopPrediction(TypedDict):
    """One ranked classifier prediction."""

    label: str
    confidence: float


class ClassificationResult(TypedDict):
    """Successful classification serialized by the Python API and CLI."""

    path: str
    label: str
    confidence: float
    window_count: int
    top: list[TopPrediction]
    scores: dict[str, float]
    spectrogram_paths: NotRequired[list[str]]


class ClassificationFailure(TypedDict):
    """Input that could not be classified during fault-tolerant bulk work."""

    path: str
    error: str


ClassificationItem: TypeAlias = ClassificationResult | ClassificationFailure


class ClassificationSummary(TypedDict):
    """Aggregate species and confidence statistics."""

    total: int
    classified: int
    failed: int
    label_counts: dict[str, int]
    species_counts: dict[str, int]
    noise_count: int
    mean_confidence: float | None


class BulkClassification(TypedDict):
    """JSON-compatible bulk classification response."""

    results: list[ClassificationItem]
    summary: ClassificationSummary
