"""Validate real processing artifacts, signal geometry, and persisted metadata."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import resample_poly

from batbot import spectrogram


def assert_outputs_match_metadata(outputs, destination):
    full, compressed, metadata_path, metadata = outputs
    persisted = json.loads(Path(metadata_path).read_text(encoding='utf8'))
    assert persisted == json.loads(json.dumps(metadata))
    assert metadata['spectrogram']['uncompressed.path'] == full
    assert metadata['spectrogram']['compressed.path'] == compressed
    assert full and compressed and metadata['segments']
    bands = metadata['frequencies']['pixels.hz']
    assert all(first > second for first, second in zip(bands, bands[1:]))
    assert metadata['size']['uncompressed']['height.px'] == len(bands)
    for paths in metadata['spectrogram'].values():
        for path in paths:
            assert Path(path).parent == destination
            filename = Path(path).name
            image = cv2.imdecode(
                np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED
            )
            assert image is not None and image.dtype == np.uint8
            assert image.max() > image.min()
            if 'waveplot' in filename:
                assert image.ndim == 3 and image.shape[2] == 3
                continue
            kind = (
                'compressed'
                if any(word in filename for word in ['compressed', 'mask'])
                else 'uncompressed'
            )
            if '.origsr.' in filename:
                kind += '_origsr'
            size = metadata['size'][kind]
            assert image.shape[:2] == (size['height.px'], size['width.px'])
    for segment in metadata['segments']:
        start, stop = segment['segment start.ms'], segment['segment end.ms']
        # STFT frames are centered/padded, and exported times round to microseconds.
        assert 0 <= start < stop <= metadata['duration.ms'] + 1
        assert segment['segment duration.ms'] == pytest.approx(stop - start, abs=0.002)
        if 'curve.(hz,ms)' in segment:
            curve = np.asarray(segment['curve.(hz,ms)'])
            assert len(curve) >= 2 and np.isfinite(curve).all()
            assert set(curve[:, 0]).issubset(bands)
            assert np.all((start <= curve[:, 1]) & (curve[:, 1] <= stop))
            assert segment['lo f.hz'] <= segment['peak f.hz'] <= segment['hi f.hz']
            assert type(segment['harmonic.flag']) is bool
            assert type(segment['echo.flag']) is bool
    return metadata


@pytest.mark.parametrize('original_rate,debug', [(500_000, True), (125_000, False)])
def test_full_pipeline_original_rate_images_and_debug_are_consistent(
    wav_clip, tmp_path, original_rate, debug
):
    samples, rate = sf.read(wav_clip, dtype='float32')
    if original_rate != rate:
        samples = resample_poly(samples, 1, rate // original_rate)
        wavfile.write(wav_clip, original_rate, samples.astype(np.float32))
    before = wav_clip.read_bytes()
    destination = tmp_path / 'derived 空間'
    outputs = spectrogram.compute(
        wav_clip,
        output_folder=destination,
        quiet=True,
        debug=debug,
        include_original_sr=True,
        plot_uncompressed_amplitude=True,
        mask_secondary_effects=True,
        time_buffer_ms=2,
    )
    metadata = assert_outputs_match_metadata(outputs, destination)
    assert wav_clip.read_bytes() == before
    assert metadata['sr.hz'] == 250_000
    assert metadata['metadata_origsr']['sr.hz'] == original_rate
    assert metadata['duration.ms'] == pytest.approx(len(samples) / original_rate * 1000, abs=0.001)
    assert len(outputs[0]) == len(outputs[1]) == 2
    for key in ['mask.path', 'masked.path', 'waveplot.path', 'waveplot.compressed.path']:
        assert len(metadata['spectrogram'][key]) == 2
    original_height = metadata['size']['uncompressed_origsr']['height.px']
    height = metadata['size']['uncompressed']['height.px']
    assert (original_height > height) == (original_rate > 250_000)
    debug_dir = destination / f'batbot.debug.{wav_clip.name}'
    if debug:
        raw = cv2.imdecode(
            np.frombuffer((debug_dir / 'spectrogram.tif').read_bytes(), np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        assert raw.dtype == np.uint16 and raw.shape[0] == height
        assert raw.min() == 0 and raw.max() == np.iinfo(np.uint16).max
        assert list(debug_dir.glob('contour.*.10.slope.png'))
        for name in [
            'candidates.png',
            'spectrogram.compressed.keypoints.tif',
            'spectrogram.compressed.threshold.jet.tif',
            'spectrogram.compressed.combined.png',
        ]:
            image = cv2.imdecode(
                np.frombuffer((debug_dir / name).read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED
            )
            assert image is not None and image.max() > image.min()
    else:
        assert not debug_dir.exists()


def test_fast_pipeline_has_timing_metadata_without_full_contours(wav_clip, tmp_path):
    destination = tmp_path / 'fast'
    full, compressed, metadata_path, metadata = spectrogram.compute(
        wav_clip,
        output_folder=destination,
        fast_mode=True,
    )
    assert full == [] and compressed
    assert metadata['global_threshold.amp'] == 0
    assert metadata['size']['compressed']['width.px'] < metadata['size']['uncompressed']['width.px']
    assert metadata['spectrogram']['mask.path'] == []
    assert metadata['spectrogram']['waveplot.path'] == []
    assert metadata['segments']
    for segment in metadata['segments']:
        assert set(segment) == {'segment start.ms', 'segment end.ms', 'segment duration.ms'}
        assert segment['segment start.ms'] < segment['segment end.ms']
    assert json.loads(Path(metadata_path).read_text()) == metadata


@pytest.mark.parametrize('quiet', [True, False])
def test_cached_output_is_preserved_without_loading_input(tmp_path, capsys, quiet):
    stem = tmp_path / 'recording [1]'
    cached = stem.with_name(stem.name + '.metadata.json')
    cached.write_text('{"original": true}', encoding='utf8')
    assert spectrogram.compute(
        tmp_path / 'missing.wav',
        out_file_stem=stem,
        quiet=quiet,
    ) == ([], [], [], {})
    assert cached.read_text() == '{"original": true}'
    assert ('Skipping file' in capsys.readouterr().out) == (not quiet)
