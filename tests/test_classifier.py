from dataclasses import FrozenInstanceError
from pathlib import Path

import cv2
import numpy as np
import pooch
import pytest

from batbot import classifier
from batbot.classifier import dataloader, inference, model


class _ModelValue:
    def __init__(self, name='input', shape=None):
        self.name = name
        self.shape = shape


class _ModelMetadata:
    custom_metadata_map = {}


class FakeSession:
    def __init__(self):
        self.batch_sizes = []

    def get_inputs(self):
        return [_ModelValue(shape=['batch_size', 224, 224, 3])]

    def get_outputs(self):
        return [_ModelValue(name='output', shape=['batch_size', len(classifier.CLASSES)])]

    def get_modelmeta(self):
        return _ModelMetadata()

    def run(self, output_names, inputs):
        batch = inputs['input']
        self.batch_sizes.append(len(batch))
        scores = np.zeros((len(batch), len(classifier.CLASSES)), dtype=np.float32)
        scores[:, classifier.CLASSES.index('EPFU')] = 0.75
        scores[:, classifier.CLASSES.index('NOISE')] = 0.25
        return [scores]


def test_prepare_image_matches_model_shape():
    image = np.zeros((300, 1200, 3), dtype=np.uint8)
    windows = dataloader._prepare_image(image)

    assert windows.shape == (3, 224, 224, 3)
    assert windows.dtype == np.uint8
    assert windows.flags['C_CONTIGUOUS']


def test_prepare_image_pads_narrow_spectrogram():
    image = np.full((237, 100, 3), 255, dtype=np.uint8)
    windows = dataloader._prepare_image(image)

    assert windows.shape == (1, 224, 224, 3)
    assert np.all(windows[:, :, -1, :] == 0)


def test_predict_batches_and_averages(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(inference, 'fetch', lambda **kwargs: 'model.onnx')
    monkeypatch.setattr(inference, '_create_session', lambda *args, **kwargs: session)
    windows = np.zeros((5, 224, 224, 3), dtype=np.uint8)

    output = classifier.post(classifier.predict(iter([(windows, 'mobilenet')]), batch_size=2))

    assert session.batch_sizes == [2, 2, 1]
    assert output[0]['EPFU'] == 0.75
    assert output[0]['NOISE'] == 0.25


def test_classifier_reuses_its_session(monkeypatch, tmp_path):
    path = tmp_path / 'spectrogram.jpg'
    cv2.imwrite(str(path), np.zeros((300, 700, 3), dtype=np.uint8))
    session = FakeSession()
    created = []

    monkeypatch.setattr(inference, 'fetch', lambda **kwargs: 'model.onnx')

    def create_session(*args, **kwargs):
        created.append(True)
        return session

    monkeypatch.setattr(inference, '_create_session', create_session)
    runner = classifier.Classifier(top_k=2)

    first = runner.classify(path)[0]
    second = runner.classify(path)[0]

    assert len(created) == 1
    assert first == second
    assert first['top'] == [
        {'label': 'EPFU', 'confidence': 0.75},
        {'label': 'NOISE', 'confidence': 0.25},
    ]


def test_config_is_immutable():
    config = classifier.resolve_config('mobilenet')

    with pytest.raises(FrozenInstanceError):
        config.key = 'changed'


def test_fetch_uses_verified_bundled_model():
    model_path = Path(classifier.fetch())

    assert model_path.name == classifier.MODEL_NAME
    assert model_path.exists()
    assert pooch.file_hash(str(model_path), alg='sha256') == classifier.MODEL_HASH


def test_fetch_uses_versioned_mirror_cache(monkeypatch, tmp_path):
    downloaded = tmp_path / classifier.MODEL_NAME
    downloaded.write_bytes(b'model')
    calls = []
    model._fetch_cached.cache_clear()
    monkeypatch.setattr(model, '_bundled_model', lambda config: None)
    monkeypatch.setattr(model, '_cache_directory', lambda: tmp_path / 'models' / '0.2.0')

    def retrieve(**kwargs):
        calls.append(kwargs)
        return str(downloaded)

    monkeypatch.setattr(model.pooch, 'retrieve', retrieve)

    assert classifier.fetch() == str(downloaded)
    assert calls[0]['known_hash'] == f'sha256:{classifier.MODEL_HASH}'
    assert calls[0]['path'] == tmp_path / 'models' / '0.2.0'
    model._fetch_cached.cache_clear()


def test_real_model_inference(tmp_path):
    path = tmp_path / 'spectrogram.png'
    cv2.imwrite(str(path), np.zeros((300, 700, 3), dtype=np.uint8))

    result = classifier.Classifier(top_k=3).classify(path)[0]

    assert result['label'] in classifier.CLASSES
    assert len(result['scores']) == len(classifier.CLASSES)
    assert len(result['top']) == 3
    assert result['window_count'] > 0


def test_discover_inputs_and_summary(tmp_path):
    species = tmp_path / 'EPFU'
    species.mkdir()
    wav = species / 'one.WAV'
    image = species / 'one.jpg'
    ignored = species / 'notes.txt'
    for path in [wav, image, ignored]:
        path.touch()

    discovered = classifier.discover_inputs([tmp_path])
    summary = classifier.summarize(
        [
            {'label': 'EPFU', 'confidence': 0.8},
            {'label': 'NOISE', 'confidence': 0.6},
            {'path': 'bad.wav', 'error': 'bad input'},
        ]
    )

    assert discovered == sorted([str(image), str(wav)])
    assert summary['species_counts'] == {'EPFU': 1}
    assert summary['noise_count'] == 1
    assert summary['classified'] == 2
    assert summary['failed'] == 1
