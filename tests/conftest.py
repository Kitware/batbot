"""Small local recordings for functional tests; no downloads or private data."""

from pathlib import Path

import pytest


@pytest.fixture
def wav_clip(tmp_path):
    """Keep real call structure while bounding processing/debugging costs."""
    import soundfile as sf
    from scipy.io import wavfile

    source = Path(__file__).resolve().parents[1] / 'examples' / 'example1.wav'
    samples, rate = sf.read(source, frames=100_000, dtype='float32')
    destination = tmp_path / 'recording 空間.WAV'
    wavfile.write(destination, rate, samples)
    return destination
