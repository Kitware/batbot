import json

import cv2
import numpy as np
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
