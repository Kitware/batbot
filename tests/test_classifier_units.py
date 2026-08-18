import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from batbot import classifier
from batbot.classifier import bulk, dataloader, inference, model

MOBILENET = classifier.resolve_config('mobilenet')


def _result(path, label='EPFU', confidence=0.75, window_count=1):
    scores = {'EPFU': confidence, 'NOISE': 1.0 - confidence}
    return {
        'path': str(path),
        'label': label,
        'confidence': confidence,
        'window_count': window_count,
        'top': [{'label': label, 'confidence': confidence}],
        'scores': scores,
    }


class _Session:
    def __init__(self, inputs=None, outputs=None, metadata=None):
        self._inputs = (
            [SimpleNamespace(name='input', shape=['batch', 224, 224, 3])]
            if inputs is None
            else inputs
        )
        self._outputs = [SimpleNamespace(name='output')] if outputs is None else outputs
        self._metadata = {} if metadata is None else metadata

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=self._metadata)

    def run(self, output_names, inputs):
        batch = inputs['input']
        return [np.zeros((len(batch), len(classifier.CLASSES)), dtype=np.float32)]


class _Runner:
    def __init__(self):
        self.calls = []

    def classify(self, paths, top_k=5):
        paths = [str(path) for path in paths]
        self.calls.append((paths, top_k))
        return [_result(path) for path in paths]


@pytest.mark.parametrize(
    'image',
    [None, np.zeros((20, 20), dtype=np.uint8), np.zeros((20, 20, 4), dtype=np.uint8)],
)
def test_prepare_image_rejects_non_color_images(image):
    with pytest.raises(ValueError, match='three-channel'):
        dataloader._prepare_image(image)


@pytest.mark.parametrize(
    ('argument', 'value', 'message'),
    [('input_size', 0, 'input_size'), ('window_stride', 0, 'window_stride')],
)
def test_prepare_image_rejects_nonpositive_geometry(argument, value, message):
    kwargs = {argument: value}

    with pytest.raises(ValueError, match=message):
        dataloader._prepare_image(np.zeros((20, 20, 3), dtype=np.uint8), **kwargs)


def test_load_image_reports_missing_file(tmp_path):
    with pytest.raises(OSError, match='Unable to load spectrogram'):
        dataloader._load_image(tmp_path / 'missing.png')


def test_dataset_applies_sample_and_target_transforms():
    dataset = dataloader.ImageFilePathList(
        ['one.png', 'two.png'],
        targets=['NOISE', 'EPFU'],
        transform=lambda image: image + 1,
        target_transform=str.lower,
    )
    dataset.loader = lambda path: np.zeros((2, 2, 3), dtype=np.uint8)

    sample, target = dataset[0]

    assert len(dataset) == 2
    assert np.all(sample == 1)
    assert target == 'noise'
    assert dataset.classes == ['EPFU', 'NOISE']
    assert dataset.class_to_idx == {'EPFU': 0, 'NOISE': 1}


def test_dataset_without_targets_and_mismatched_targets():
    dataset = dataloader.ImageFilePathList(['one.png'])
    dataset.loader = lambda path: np.zeros((2, 2, 3), dtype=np.uint8)

    assert len(dataset[0]) == 1
    assert dataset.classes is None
    assert dataset.class_to_idx is None
    with pytest.raises(ValueError, match='same length'):
        dataloader.ImageFilePathList(['one.png'], targets=[])


def test_resolve_config_normalizes_names_and_rejects_unknown_values():
    selected = classifier.resolve_config('  MOBILENET ')

    assert classifier.resolve_config(selected) is selected
    assert classifier.resolve_config(None) is selected
    with pytest.raises(ValueError, match='choose from mobilenet'):
        classifier.resolve_config('unknown')


@pytest.mark.parametrize(
    ('session', 'message'),
    [
        (
            _Session(inputs=[SimpleNamespace(shape=[]), SimpleNamespace(shape=[])]),
            'one input',
        ),
        (_Session(outputs=[]), 'one input'),
        (_Session(inputs=[SimpleNamespace(shape=['batch', 3, 224, 224])]), 'input shape'),
    ],
)
def test_validate_session_rejects_incompatible_graphs(session, message):
    with pytest.raises(ValueError, match=message):
        inference._validate_session(session, MOBILENET)


def test_validate_session_checks_embedded_labels():
    labels = {
        'labels': json.dumps(
            {'forward': {str(index): label for index, label in enumerate(classifier.CLASSES)}}
        )
    }
    inference._validate_session(_Session(metadata=labels), MOBILENET)

    labels['labels'] = json.dumps({'forward': {'0': 'WRONG'}})
    with pytest.raises(ValueError, match='labels do not match'):
        inference._validate_session(_Session(metadata=labels), MOBILENET)


def test_predict_rejects_invalid_batches_and_empty_windows():
    with pytest.raises(ValueError, match='batch_size'):
        list(inference.predict([], batch_size=0))

    empty = np.empty((0, 224, 224, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match='no image windows'):
        list(
            inference.predict(
                [(empty, 'mobilenet')],
                sessions={'mobilenet': _Session()},
            )
        )


def test_post_rejects_wrong_score_count():
    predictions = np.zeros((1, len(classifier.CLASSES) - 1), dtype=np.float32)

    with pytest.raises(ValueError, match='scores'):
        inference.post([(predictions, 'mobilenet')])


def test_classifier_validates_options_and_accepts_an_existing_session():
    with pytest.raises(ValueError, match='batch_size'):
        classifier.Classifier(batch_size=0)
    with pytest.raises(ValueError, match='top_k'):
        classifier.Classifier(top_k=0)

    session = _Session()
    runner = classifier.Classifier(sessions={'mobilenet': session})

    assert runner.session is session
    with pytest.raises(ValueError, match='top_k'):
        runner.classify([], top_k=0)


def test_classifier_convenience_methods_reuse_runner(monkeypatch):
    captured = []

    def fake_wav(filepath, **kwargs):
        captured.append(('wav', filepath, kwargs))
        return _result(filepath)

    def fake_bulk(inputs, **kwargs):
        captured.append(('bulk', inputs, kwargs))
        return {'results': [], 'summary': bulk.summarize([])}

    monkeypatch.setattr(bulk, 'classify_wav', fake_wav)
    monkeypatch.setattr(bulk, 'classify_bulk', fake_bulk)
    runner = classifier.Classifier(sessions={'mobilenet': _Session()}, top_k=3)

    runner.classify_wav('recording.wav', keep_spectrograms=True)
    runner.classify_bulk('recordings', input_type='wav', recursive=False)

    assert captured[0][2]['top_k'] == 3
    assert captured[0][2]['keep_spectrograms'] is True
    assert captured[0][2]['_runner'] is runner
    assert captured[1][2]['input_type'] == 'wav'
    assert captured[1][2]['recursive'] is False
    assert captured[1][2]['_runner'] is runner


def test_model_resource_materialization_is_verified_and_reused(monkeypatch, tmp_path):
    payload = b'verified ONNX model'
    source = tmp_path / 'source.onnx'
    source.write_bytes(payload)
    config = replace(
        MOBILENET,
        filename='test.onnx',
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    cache = tmp_path / 'cache'
    monkeypatch.setattr(model, '_cache_directory', lambda: cache)

    materialized = model._materialize_resource(source, config)
    source.write_bytes(b'changed after the copy')

    assert materialized.read_bytes() == payload
    assert model._materialize_resource(source, config) == materialized


def test_model_resource_materialization_rejects_bad_checksum(monkeypatch, tmp_path):
    source = tmp_path / 'source.onnx'
    source.write_bytes(b'corrupt')
    config = replace(MOBILENET, filename='test.onnx', sha256='0' * 64)
    cache = tmp_path / 'cache'
    monkeypatch.setattr(model, '_cache_directory', lambda: cache)

    with pytest.raises(ValueError, match='checksum'):
        model._materialize_resource(source, config)

    assert not (cache / config.filename).exists()
    assert list(cache.iterdir()) == []


def test_bundled_model_missing_or_corrupt_uses_mirror(monkeypatch):
    missing = replace(MOBILENET, resource_parts=('missing',))

    assert model._bundled_model(missing) is None

    monkeypatch.setattr(model, '_has_expected_hash', lambda path, config: False)
    assert model._bundled_model(MOBILENET) is None


def test_aggregate_results_weights_windows_and_rejects_empty_inputs():
    first = _result('first.png', confidence=0.2, window_count=1)
    second = _result('second.png', confidence=0.8, window_count=3)

    combined = bulk._aggregate_results(
        'recording.wav',
        [first, second],
        ['first.png', 'second.png'],
        top_k=2,
    )

    assert combined['path'] == 'recording.wav'
    assert combined['scores']['EPFU'] == pytest.approx(0.65)
    assert combined['scores']['NOISE'] == pytest.approx(0.35)
    assert combined['window_count'] == 4
    assert combined['spectrogram_paths'] == ['first.png', 'second.png']
    with pytest.raises(ValueError, match='No spectrograms'):
        bulk._aggregate_results('empty.wav', [], [])


def test_classify_wav_cleans_temporary_spectrograms(monkeypatch):
    from batbot import spectrogram

    output_directories = []

    def fake_compute(filepath, **kwargs):
        output_directory = Path(kwargs['output_folder'])
        output_directories.append(output_directory)
        return [str(output_directory / 'one.png')], [], None, None

    monkeypatch.setattr(spectrogram, 'compute', fake_compute)
    result = bulk.classify_wav('recording.wav', _runner=_Runner())

    assert result['spectrogram_paths'] == []
    assert not output_directories[0].exists()


def test_classify_wav_retains_requested_spectrograms(monkeypatch, tmp_path):
    from batbot import spectrogram

    captured = {}

    def fake_compute(filepath, **kwargs):
        captured.update(kwargs)
        return [str(tmp_path / 'one.png')], [], None, None

    monkeypatch.setattr(spectrogram, 'compute', fake_compute)
    result = bulk.classify_wav(
        'recording.wav',
        output_folder=tmp_path,
        out_file_stem=tmp_path / 'custom',
        _runner=_Runner(),
    )

    assert result['spectrogram_paths'] == [str(tmp_path / 'one.png')]
    assert captured['output_folder'] == str(tmp_path)
    assert captured['out_file_stem'] == str(tmp_path / 'custom')


def test_discover_inputs_filters_type_recursion_and_invalid_paths(tmp_path):
    image = tmp_path / 'top.png'
    wav = tmp_path / 'top.WAV'
    nested = tmp_path / 'nested'
    nested.mkdir()
    nested_image = nested / 'nested.jpg'
    for path in [image, wav, nested_image, tmp_path / 'ignored.txt']:
        path.touch()

    assert bulk.discover_inputs(tmp_path, input_type='spectrogram', recursive=False) == [str(image)]
    assert bulk.discover_inputs(tmp_path, input_type='wav') == [str(wav)]
    assert bulk.discover_inputs(image) == [str(image)]
    with pytest.raises(ValueError, match='input_type'):
        bulk.discover_inputs(tmp_path, input_type='video')
    with pytest.raises(FileNotFoundError, match='does not exist'):
        bulk.discover_inputs(tmp_path / 'missing')


def test_summarize_empty_results():
    assert bulk.summarize([]) == {
        'total': 0,
        'classified': 0,
        'failed': 0,
        'label_counts': {},
        'species_counts': {},
        'noise_count': 0,
        'mean_confidence': None,
    }


def test_classify_bulk_recovers_individual_images_and_names_wav_outputs(monkeypatch, tmp_path):
    good = tmp_path / 'good.jpg'
    bad = tmp_path / 'bad.png'
    wav = tmp_path / 'call.wav'
    for path in [good, bad, wav]:
        path.touch()
    spectrogram_output = tmp_path / 'spectrograms'
    captured_wav = {}

    class FaultTolerantRunner(_Runner):
        def classify(self, paths, top_k=5):
            paths = [str(path) for path in paths]
            if len(paths) > 1:
                raise OSError('batch decoder failed')
            if Path(paths[0]).name == 'bad.png':
                raise ValueError('invalid image')
            return [_result(paths[0])]

    def fake_classify_wav(path, **kwargs):
        captured_wav.update(kwargs)
        return _result(path)

    monkeypatch.setattr(bulk, 'classify_wav', fake_classify_wav)
    output = bulk.classify_bulk(
        tmp_path,
        spectrogram_output=spectrogram_output,
        _runner=FaultTolerantRunner(),
    )

    assert output['summary']['total'] == 3
    assert output['summary']['classified'] == 2
    assert output['summary']['failed'] == 1
    assert next(item for item in output['results'] if item['path'] == str(bad))['error'] == (
        'invalid image'
    )
    assert captured_wav['output_folder'] == spectrogram_output
    assert captured_wav['keep_spectrograms'] is True
    assert Path(captured_wav['out_file_stem']).parent == spectrogram_output
    assert Path(captured_wav['out_file_stem']).name.startswith('call.')
