import json

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from batbot import classifier
from batbot.batbot_cli import classify, classify_bulk, classify_wav


def _result(path):
    return {
        'path': str(path),
        'label': 'EPFU',
        'confidence': 0.75,
        'window_count': 1,
        'top': [{'label': 'EPFU', 'confidence': 0.75}],
        'scores': {'EPFU': 0.75},
    }


def _bulk(path):
    result = _result(path)
    return {'results': [result], 'summary': classifier.summarize([result])}


def test_classify_cli_writes_json(monkeypatch, tmp_path):
    image = tmp_path / 'spectrogram.png'
    output = tmp_path / 'result.json'
    cv2.imwrite(str(image), np.zeros((300, 700, 3), dtype=np.uint8))
    calls = []
    monkeypatch.setattr(
        classifier,
        'classify',
        lambda paths, **kwargs: calls.append(kwargs) or [_result(paths[0])],
    )

    invocation = CliRunner().invoke(
        classify,
        [str(image), '--output', str(output), '--num-workers', '3'],
    )

    assert invocation.exit_code == 0, invocation.output
    assert json.loads(output.read_text())['results'][0]['label'] == 'EPFU'
    assert calls[0]['num_workers'] == 3


def test_classify_wav_cli(monkeypatch, tmp_path):
    wav = tmp_path / 'recording.wav'
    wav.touch()
    calls = []
    monkeypatch.setattr(
        classifier,
        'classify_bulk',
        lambda paths, **kwargs: calls.append(kwargs) or _bulk(paths[0]),
    )

    invocation = CliRunner().invoke(classify_wav, [str(wav), '--num-workers', '4'])

    assert invocation.exit_code == 0, invocation.output
    assert json.loads(invocation.output)['summary']['species_counts'] == {'EPFU': 1}
    assert calls[0]['num_workers'] == 4


def test_classify_bulk_cli(monkeypatch, tmp_path):
    wav = tmp_path / 'recording.wav'
    wav.touch()
    calls = []
    monkeypatch.setattr(
        classifier,
        'classify_bulk',
        lambda paths, **kwargs: calls.append(kwargs) or _bulk(paths[0]),
    )

    invocation = CliRunner().invoke(
        classify_bulk,
        [str(tmp_path), '--input-type', 'wav', '--num-workers', '5'],
    )

    assert invocation.exit_code == 0, invocation.output
    assert json.loads(invocation.output)['results'][0]['path'] == str(tmp_path)
    assert calls[0]['num_workers'] == 5


@pytest.mark.parametrize('command', [classify_wav, classify_bulk])
def test_wav_cli_forwards_gpu_and_preprocessing_controls(monkeypatch, tmp_path, command):
    wav = tmp_path / 'recording.wav'
    wav.touch()
    calls = []
    monkeypatch.setattr(
        classifier,
        'classify_bulk',
        lambda paths, **kwargs: calls.append(kwargs) or _bulk(paths[0]),
    )

    result = CliRunner().invoke(
        command,
        [str(wav), '--preprocess-workers', '3', '--provider', 'cuda', '--no-progress'],
    )

    assert result.exit_code == 0, result.output
    assert calls[0]['preprocess_workers'] == 3
    assert calls[0]['providers'] == ['CUDAExecutionProvider']
    assert calls[0]['progress'] is False


@pytest.mark.parametrize(
    'provider,expected',
    [
        ('cpu', 'CPUExecutionProvider'),
        ('coreml', 'CoreMLExecutionProvider'),
    ],
)
def test_image_cli_forwards_provider_and_progress_controls(
    monkeypatch, tmp_path, provider, expected
):
    image = tmp_path / 'spectrogram.jpg'
    image.touch()
    calls = []
    monkeypatch.setattr(
        classifier,
        'classify',
        lambda paths, **kwargs: calls.append(kwargs) or [_result(paths[0])],
    )

    result = CliRunner().invoke(classify, [str(image), '--provider', provider, '--no-progress'])

    assert result.exit_code == 0, result.output
    assert calls[0]['providers'] == [expected]
    assert calls[0]['progress'] is False
