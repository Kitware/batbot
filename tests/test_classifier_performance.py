import json

import pytest

from examples import plot_classifier_performance as performance


def test_discover_inputs_accepts_wav_jpg_and_jpeg_case_insensitively(tmp_path):
    label = tmp_path / 'EPFU'
    label.mkdir()
    expected = [label / 'call.JPG', label / 'call.jpeg', label / 'recording.wav']
    ignored = [label / 'notes.txt', label / 'spectrogram.png']
    for path in [*expected, *ignored]:
        path.touch()

    assert performance.discover_inputs(tmp_path) == sorted(expected)


def test_run_predictions_dispatches_mixed_inputs_through_one_classifier(monkeypatch, tmp_path):
    wav = tmp_path / 'EPFU' / 'recording.wav'
    jpg = tmp_path / 'NOISE' / 'call.JPG'
    jpeg = tmp_path / 'EPFU' / 'call.jpeg'
    cache = tmp_path / 'predictions.json'
    calls = []

    class FakeClassifier:
        def __init__(self, **kwargs):
            calls.append(('init', kwargs))

        def classify_wav(self, path):
            calls.append(('wav', path))
            return {'path': str(path), 'source': 'wav'}

        def classify(self, path):
            calls.append(('image', path))
            return [{'path': str(path), 'source': 'image'}]

    monkeypatch.setattr(performance.classifier, 'Classifier', FakeClassifier)

    predictions = performance.run_predictions(
        [wav, jpg, jpeg],
        cache_path=cache,
        batch_size=6,
        num_workers=3,
    )

    assert calls == [
        ('init', {'batch_size': 6, 'num_workers': 3}),
        ('wav', wav),
        ('image', jpg),
        ('image', jpeg),
    ]
    assert [prediction['source'] for prediction in predictions] == ['wav', 'image', 'image']
    assert json.loads(cache.read_text()) == {'results': predictions}


def test_run_predictions_reuses_matching_cache_without_loading_model(monkeypatch, tmp_path):
    path = tmp_path / 'EPFU' / 'call.jpg'
    cache = tmp_path / 'predictions.json'
    expected = [{'path': str(path), 'label': 'EPFU'}]
    cache.write_text(json.dumps({'results': expected}))
    monkeypatch.setattr(
        performance.classifier,
        'Classifier',
        lambda **kwargs: pytest.fail('Classifier should not be created for a valid cache'),
    )

    assert performance.run_predictions([path], cache_path=cache) == expected


def test_run_predictions_rejects_unsupported_explicit_input(monkeypatch, tmp_path):
    class FakeClassifier:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(performance.classifier, 'Classifier', FakeClassifier)

    with pytest.raises(ValueError, match='Unsupported classifier input'):
        performance.run_predictions([tmp_path / 'call.png'])
