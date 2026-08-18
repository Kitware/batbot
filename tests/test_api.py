from pathlib import Path

import pooch
import pytest
import tqdm

import batbot.api as api
from batbot import classifier, spectrogram


def _result(path):
    return {
        'path': str(path),
        'label': 'EPFU',
        'confidence': 0.75,
        'window_count': 1,
        'top': [{'label': 'EPFU', 'confidence': 0.75}],
        'scores': {'EPFU': 0.75},
    }


def test_fetch_delegates_to_classifier(monkeypatch):
    calls = []
    monkeypatch.setattr(
        classifier,
        'fetch',
        lambda **kwargs: calls.append(kwargs) or '/models/model.onnx',
    )

    assert api.fetch(pull=True, config='mobilenet') == '/models/model.onnx'
    assert calls == [{'pull': True, 'config': 'mobilenet'}]


def test_pipeline_forwards_spectrogram_options(monkeypatch):
    calls = []

    def fake_compute(filepath, **kwargs):
        calls.append((filepath, kwargs))
        return ['one.png'], ['one-compressed.png'], 'one.json', {'ignored': True}

    monkeypatch.setattr(spectrogram, 'compute', fake_compute)

    output = api.pipeline(
        Path('recording.wav'),
        out_file_stem='custom',
        output_folder='output',
        fast_mode=True,
        force_overwrite=True,
        quiet=True,
        plot_uncompressed_amplitude=True,
        include_original_sr=True,
        time_buffer_ms=2.5,
        debug=True,
    )

    assert output == (['one.png'], ['one-compressed.png'], 'one.json')
    assert calls == [
        (
            'recording.wav',
            {
                'out_file_stem': 'custom',
                'output_folder': 'output',
                'fast_mode': True,
                'force_overwrite': True,
                'quiet': True,
                'plot_uncompressed_amplitude': True,
                'include_original_sr': True,
                'time_buffer_ms': 2.5,
                'debug': True,
            },
        )
    ]


def test_pipeline_multi_wrapper_collects_outputs_and_failures(monkeypatch):
    def fake_pipeline(filepath, **kwargs):
        if filepath == 'bad.wav':
            raise ValueError('bad recording')
        return [f'{filepath}.png'], [f'{filepath}.compressed.png'], f'{filepath}.json'

    monkeypatch.setattr(api, 'pipeline', fake_pipeline)

    output, compressed, metadata, failures = api.pipeline_multi_wrapper(
        ['good.wav', 'bad.wav'],
        out_file_stems=['good', 'bad'],
        quiet=True,
    )

    assert output == ['good.wav.png']
    assert compressed == ['good.wav.compressed.png']
    assert metadata == ['good.wav.json']
    assert failures[0][0] == 'bad.wav'
    assert str(failures[0][1]) == 'bad recording'


def test_pipeline_multi_wrapper_builds_default_stems_and_sets_lock(monkeypatch):
    lock = object()
    captured = []
    monkeypatch.setattr(
        api,
        'pipeline',
        lambda filepath, **kwargs: captured.append((filepath, kwargs)) or ([], [], None),
    )
    monkeypatch.setattr(tqdm.tqdm, 'set_lock', lambda value: captured.append(('lock', value)))

    output = api.pipeline_multi_wrapper(['one.wav'], tqdm_lock=lock, quiet=True)

    assert output == ([], [], [None], [])
    assert captured[0] == ('lock', lock)
    assert captured[1][1]['out_file_stem'] is None


def test_pipeline_multi_wrapper_validates_stem_count():
    with pytest.raises(ValueError, match='different length'):
        api.pipeline_multi_wrapper(['one.wav'], out_file_stems=[])


def test_parallel_pipeline_validates_work(monkeypatch):
    assert api.parallel_pipeline([]) is None
    with pytest.raises(ValueError, match='same length'):
        api.parallel_pipeline([['one.wav']], out_stem_chunks=[])
    with pytest.raises(ValueError, match='num_workers'):
        api.parallel_pipeline([['one.wav']], num_workers=0)


def test_parallel_pipeline_combines_threaded_results(monkeypatch):
    class LockManager:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def Lock(self):
            return object()

    def fake_wrapper(filepaths, **kwargs):
        path = filepaths[0]
        return [f'{path}.png'], [f'{path}.compressed.png'], [f'{path}.json'], []

    monkeypatch.setattr(api, 'Manager', LockManager)
    monkeypatch.setattr(api, 'pipeline_multi_wrapper', fake_wrapper)

    output = api.parallel_pipeline(
        [['one.wav'], ['two.wav']],
        num_workers=2,
        threaded=True,
        quiet=True,
    )

    assert output is not None
    paths, compressed, metadata, failures = output
    assert sorted(paths) == ['one.wav.png', 'two.wav.png']
    assert sorted(compressed) == ['one.wav.compressed.png', 'two.wav.compressed.png']
    assert sorted(metadata) == ['one.wav.json', 'two.wav.json']
    assert failures == []


def test_batch_reuses_one_classifier(monkeypatch):
    instances = []

    class FakeClassifier:
        def __init__(self, config=None):
            self.config = config
            self.paths = []
            instances.append(self)

        def classify_wav(self, filepath):
            self.paths.append(filepath)
            return _result(filepath)

    monkeypatch.setattr(classifier, 'Classifier', FakeClassifier)

    results = api.batch(['one.wav', Path('two.wav')], config='mobilenet', clean=False)

    assert len(instances) == 1
    assert instances[0].config == 'mobilenet'
    assert instances[0].paths == ['one.wav', Path('two.wav')]
    assert [result['path'] for result in results] == ['one.wav', 'two.wav']


def test_example_downloads_missing_wav_and_runs_pipeline(monkeypatch, tmp_path, capsys):
    downloaded = tmp_path / 'downloaded.wav'
    downloaded.touch()
    pipeline_calls = []
    retrieve_calls = []
    times = iter([10.0, 12.5])
    monkeypatch.setattr(api, 'PROJECT_ROOT', tmp_path / 'missing-project')
    monkeypatch.setattr(
        pooch,
        'retrieve',
        lambda **kwargs: retrieve_calls.append(kwargs) or str(downloaded),
    )
    monkeypatch.setattr(
        api,
        'pipeline',
        lambda filepath, **kwargs: pipeline_calls.append((filepath, kwargs)) or ([], [], None),
    )
    monkeypatch.setattr(api.time, 'time', lambda: next(times))

    api.example()

    assert retrieve_calls[0]['known_hash'].startswith('sha256:')
    assert pipeline_calls[0][0] == downloaded
    assert pipeline_calls[0][1]['out_file_stem'] == 'output/downloaded'
    assert '2.5 seconds' in capsys.readouterr().out
