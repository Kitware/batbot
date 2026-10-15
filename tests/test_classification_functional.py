"""Exercise CLI -> files -> preprocessing -> real ONNX -> JSON without mocks."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from batbot import classifier
from batbot._paths import write_image
from batbot.batbot_cli import cli


def assert_prediction(result):
    assert set(result['scores']) == set(classifier.CLASSES)
    scores = np.asarray(list(result['scores'].values()))
    assert np.isfinite(scores).all() and np.all((0 <= scores) & (scores <= 1))
    assert scores.sum() == pytest.approx(1, abs=1e-5)
    ranked = sorted(result['scores'], key=result['scores'].get, reverse=True)
    assert result['label'] == ranked[0]
    assert result['confidence'] == result['scores'][ranked[0]]
    assert result['top'] == [
        {'label': label, 'confidence': result['scores'][label]} for label in ranked[:5]
    ]
    assert result['window_count'] >= 1


@pytest.mark.parametrize('command', ['classify', 'classify-wav', 'classify-bulk', 'batch'])
def test_cli_predictions_failures_retained_inputs_and_summaries_are_consistent(
    wav_clip, tmp_path, command
):
    directory = tmp_path / 'inputs 空間'
    directory.mkdir()
    recording = directory / 'call.WAV'
    recording.write_bytes(wav_clip.read_bytes())
    invalid_wav = directory / 'invalid.wav'
    invalid_wav.write_bytes(b'not a waveform')
    image = directory / 'blank.PNG'
    write_image(image, np.zeros((300, 700, 3), dtype=np.uint8))
    invalid_image = directory / 'invalid.JPG'
    invalid_image.write_bytes(b'not an image')
    (directory / 'notes.txt').write_text('not an input')
    inputs = {
        'classify': [image],
        'classify-wav': [recording, invalid_wav],
        'classify-bulk': [directory],
        'batch': [recording, recording],  # Preserve duplicates in the legacy list API.
    }[command]
    expected_paths = (
        sorted([image, recording, invalid_image, invalid_wav])
        if command == 'classify-bulk'
        else inputs
    )
    destination = tmp_path / 'derived'
    arguments = [
        command,
        *map(str, inputs),
        '--provider',
        'cpu',
        '--no-progress',
        '--num-workers',
        '2',
    ]
    if command != 'classify':
        arguments += ['--spectrogram-dir', str(destination)]
    output = tmp_path / 'results 空間.json'
    use_file = command in {'classify-bulk', 'batch'}
    if use_file:
        arguments += ['--output', str(output)]
    invocation = CliRunner().invoke(cli, arguments)
    assert invocation.exit_code == 0, invocation.output
    report = json.loads(output.read_text(encoding='utf8') if use_file else invocation.stdout)
    assert [item['path'] for item in report['results']] == list(map(str, expected_paths))
    successful = [item for item in report['results'] if 'label' in item]
    failures = [item for item in report['results'] if 'error' in item]
    expected_failures = 2 if command == 'classify-bulk' else 1 if command == 'classify-wav' else 0
    assert len(failures) == expected_failures
    assert all(item['error'] and 'label' not in item for item in failures)
    assert len(successful) == len(expected_paths) - expected_failures
    counts = Counter(item['label'] for item in successful)
    assert report['summary'] == {
        'total': len(expected_paths),
        'classified': len(successful),
        'failed': len(failures),
        'label_counts': dict(sorted(counts.items())),
        'species_counts': dict(
            sorted((label, count) for label, count in counts.items() if label != 'NOISE')
        ),
        'noise_count': counts['NOISE'],
        'mean_confidence': pytest.approx(np.mean([item['confidence'] for item in successful])),
    }
    reference = None
    for result in successful:
        assert_prediction(result)
        if result['path'] == str(image):
            assert result['label'] == 'NOISE' and result['window_count'] == 1
            continue
        retained = result['spectrogram_paths']
        assert retained and all(Path(path).is_file() for path in retained)
        metadata_paths = list(destination.glob('*.metadata.json'))
        assert len(metadata_paths) == 1
        metadata = json.loads(metadata_paths[0].read_text())
        assert retained == metadata['spectrogram']['uncompressed.path']
        assert set(retained).isdisjoint(metadata['spectrogram']['compressed.path'])
        if reference is None:
            runner = classifier.Classifier(providers=['CPUExecutionProvider'])
            reference = runner.classify(retained, progress=False)
        weights = [item['window_count'] for item in reference]
        scores = np.average(
            [[item['scores'][label] for label in classifier.CLASSES] for item in reference],
            axis=0,
            weights=weights,
        )
        np.testing.assert_allclose(
            [result['scores'][label] for label in classifier.CLASSES], scores, rtol=1e-5, atol=1e-7
        )
        assert result['window_count'] == sum(weights)
