import json
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from examples import plot_classifier_performance as performance


def test_genus_reordering_preserves_score_and_label_correspondence():
    classes = list(performance.classifier.CLASSES)
    targets = np.arange(len(classes))
    predicted = np.roll(targets, 1)
    confidence = np.eye(len(classes))[predicted] * 0.9 + 0.1 / len(classes)
    original_confidence = confidence.copy()
    reordered, scores, truth, predictions = performance.apply_genus_order(
        classes,
        confidence,
        targets,
        predicted,
        classes.index('NOISE'),
    )
    # Expected biological display order, independent of the swap implementation.
    assert reordered[4:6] == ['EUMA', 'EUFL']
    assert reordered[8:13] == ['LANO', 'LABL', 'LABO', 'LACI', 'LAIN']
    assert reordered[29:31] == ['NYHU', 'NYFE']
    assert [reordered[index] for index in truth] == classes
    assert [reordered[index] for index in predictions] == [classes[index] for index in predicted]
    np.testing.assert_array_equal(scores.argmax(axis=1), predictions)
    for index, label in enumerate(classes):
        np.testing.assert_array_equal(scores[:, reordered.index(label)], confidence[:, index])
    np.testing.assert_array_equal(targets, np.arange(len(classes)))
    np.testing.assert_array_equal(confidence, original_confidence)
    np.testing.assert_array_equal(predicted, np.roll(np.arange(len(classes)), 1))
    assert classes == list(performance.classifier.CLASSES)


def test_genus_shading_counts_only_cross_genus_errors_and_handles_empty_data():
    # Aliases separate EUMA (Euderma) from Eumops and LANO from Lasiurus.
    display = ['EUMA', 'EUFL', 'EUPE', 'LABL', 'LABO', 'LANO', 'NOISE']
    matrix = np.eye(7, dtype=int)
    matrix[1, 2] = 2  # Within Eumops.
    matrix[3, 4] = 3  # Within Lasiurus.
    matrix[4, 5] = 4  # Lasiurus -> Lasionycteris: an error.
    matrix[6, 0] = 5  # NOISE -> species: an error.
    matrix[0, 1] = 1  # Euderma -> Eumops: an error despite the common EU prefix.
    figure, axis = plt.subplots()
    try:
        error = performance.shade_regions(display, axis, SimpleNamespace(confusion_matrix=matrix))
        assert error == pytest.approx(10 / matrix.sum())
        bounds = sorted(
            (rectangle.get_x(), rectangle.get_y(), rectangle.get_width(), rectangle.get_height())
            for rectangle in axis.patches
        )
        expected = sorted(
            [
                (0.52, -0.52, 6, 1),  # Euderma: columns outside row 0's genus.
                (-0.48, 0.48, 1, 2),
                (2.52, 0.48, 4, 2),  # Eumops: rows 1–2.
                (-0.48, 2.48, 3, 2),
                (4.52, 2.48, 2, 2),  # Lasiurus: rows 3–4.
                (-0.48, 4.48, 5, 1),
                (5.52, 4.48, 1, 1),  # Lasionycteris: row 5.
                (-0.48, 5.48, 6, 1),  # NOISE: row 6.
            ]
        )
        np.testing.assert_allclose(bounds, expected, rtol=0, atol=1e-12)
        assert (
            performance.shade_regions(
                display, axis, SimpleNamespace(confusion_matrix=np.zeros_like(matrix))
            )
            == 0
        )
    finally:
        plt.close(figure)


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
