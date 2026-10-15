import json
from pathlib import Path

import numpy as np
import pytest

from batbot._paths import read_image


def test_spectrogram_compute(tmp_path):
    from batbot.spectrogram import compute

    wav_filepath = Path(__file__).resolve().parents[1] / 'examples' / 'example2.wav'
    output_folder = tmp_path / 'spectrograms 空間'
    output_paths, compressed_paths, metadata_path, metadata = compute(
        wav_filepath, output_folder=output_folder
    )
    assert output_paths and compressed_paths and metadata['segments']
    assert metadata['wav.path'] == str(wav_filepath)
    assert metadata['duration.ms'] == pytest.approx(4999.022)
    assert metadata['sr.hz'] == 250_000
    assert json.loads(Path(metadata_path).read_text()) == json.loads(json.dumps(metadata))
    assert metadata['spectrogram']['uncompressed.path'] == output_paths
    assert metadata['spectrogram']['compressed.path'] == compressed_paths
    widths = []
    for path in output_paths:
        image = read_image(path)
        size = metadata['size']['uncompressed']
        assert image.shape[0] == size['height.px'] and image.shape[2] == 3
        widths.append(image.shape[1])
        assert image.dtype == np.uint8 and image.max() > image.min()
    assert widths == [50_000, size['width.px'] - 50_000]
    assert sum(widths) == metadata['size']['uncompressed']['width.px']
    for segment in metadata['segments']:
        assert segment['segment start.ms'] < segment['segment end.ms']
        assert segment['lo f.hz'] <= segment['peak f.hz'] <= segment['hi f.hz']
        assert np.isfinite(segment['bandwidth.hz']) and segment['bandwidth.hz'] > 0
