"""Regression tests for the integrated preprocessing and inference pipeline."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from click.testing import CliRunner

from batbot import classifier
from batbot.batbot_cli import classify_bulk
from batbot.classifier import bulk


class Runner:
    num_workers = 2

    def __init__(self, **kwargs):
        self.calls = []

    def classify(self, paths, top_k=5, progress=None):
        self.calls.append(list(paths))
        return [
            {
                'path': path,
                'label': 'EPFU',
                'confidence': 0.75,
                'scores': {'EPFU': 0.75, 'NOISE': 0.25},
                'window_count': 1,
                'top': [{'label': 'EPFU', 'confidence': 0.75}],
            }
            for path in paths
        ]


def make_wavs(tmp_path, count):
    paths = [tmp_path / f'input-{index:02d}.wav' for index in range(count)]
    for path in paths:
        path.touch()
    return paths


@pytest.mark.parametrize('keep_spectrograms', [True, False])
def test_parallel_preprocessing_feeds_one_inference_batch(monkeypatch, tmp_path, keep_spectrograms):
    paths = make_wavs(tmp_path, 2)
    barrier = Barrier(2)
    contexts = []

    def executor(max_workers, mp_context):
        contexts.append((max_workers, mp_context.get_start_method()))
        return ThreadPoolExecutor(max_workers=max_workers)

    def prepare(path, output_folder):
        assert Path(output_folder).is_dir()
        barrier.wait(timeout=5)
        image = Path(output_folder) / f'{Path(path).stem}.jpg'
        image.touch()
        return [str(image)]

    monkeypatch.setattr(bulk, 'ProcessPoolExecutor', executor)
    monkeypatch.setattr(bulk, '_prepare_wav', prepare)
    runner = Runner()

    output = bulk.classify_bulk(
        paths,
        preprocess_workers=2,
        progress=False,
        spectrogram_output=tmp_path / 'generated' if keep_spectrograms else None,
        _runner=runner,
    )

    assert contexts == [(2, 'spawn')]
    assert len(runner.calls) == 1
    assert len(runner.calls[0]) == 2
    assert [result['path'] for result in output['results']] == list(map(str, paths))
    assert output['summary']['species_counts'] == {'EPFU': 2}
    if keep_spectrograms:
        assert all(Path(path).exists() for path in runner.calls[0])
        assert all(len(result['spectrogram_paths']) == 1 for result in output['results'])
    else:
        assert all(not Path(path).exists() for path in runner.calls[0])
        assert all(result['spectrogram_paths'] == [] for result in output['results'])


def test_batches_bound_temporary_storage_and_clean_on_failure(monkeypatch, tmp_path):
    paths = make_wavs(tmp_path, 9)
    directories = []
    events = []

    def prepare(path, output_folder):
        directory = Path(output_folder)
        if directory not in directories:
            if directories:
                assert not directories[-1].exists()
            directories.append(directory)
        events.append(('prepare', path))
        return [str(directory / f'{Path(path).stem}.jpg')]

    class FailingRunner(Runner):
        def classify(self, paths, **kwargs):
            events.append(('classify', len(paths)))
            if len(self.calls) == 1:
                raise RuntimeError('inference failed')
            return super().classify(paths, **kwargs)

    monkeypatch.setattr(bulk, '_prepare_wav', prepare)

    with pytest.raises(RuntimeError, match='inference failed'):
        bulk.classify_bulk(paths, progress=False, _runner=FailingRunner())

    assert events[8] == ('classify', 8)
    assert events[-1] == ('classify', 1)
    assert len(directories) == 2
    assert all(not directory.exists() for directory in directories)


def test_preprocessing_failures_count_toward_both_progress_bars(monkeypatch, tmp_path):
    paths = make_wavs(tmp_path, 2)
    bars = []

    class Progress:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.count = 0
            bars.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, count):
            self.count += count

    def prepare(path, output_folder):
        if path == str(paths[0]):
            raise ValueError('invalid WAV')
        return [str(Path(output_folder) / 'valid.jpg')]

    monkeypatch.setattr(bulk.tqdm, 'tqdm', Progress)
    monkeypatch.setattr(bulk, '_prepare_wav', prepare)

    output = bulk.classify_bulk(paths, progress=True, _runner=Runner())

    assert [(bar.count, bar.options['total']) for bar in bars] == [(2, 2), (2, 2)]
    assert all(bar.options['disable'] is False for bar in bars)
    assert output['results'][0] == {'path': str(paths[0]), 'error': 'invalid WAV'}
    assert output['summary']['classified'] == 1
    assert output['summary']['failed'] == 1


def test_parallel_preprocessing_retains_worker_failures(monkeypatch, tmp_path):
    paths = make_wavs(tmp_path, 2)

    def prepare(path, output_folder):
        if path == str(paths[0]):
            raise ValueError('invalid WAV in worker')
        return [str(Path(output_folder) / 'valid.jpg')]

    monkeypatch.setattr(
        bulk,
        'ProcessPoolExecutor',
        lambda max_workers, mp_context: ThreadPoolExecutor(max_workers=max_workers),
    )
    monkeypatch.setattr(bulk, '_prepare_wav', prepare)

    output = bulk.classify_bulk(paths, preprocess_workers=2, progress=False, _runner=Runner())

    assert output['results'][0] == {'path': str(paths[0]), 'error': 'invalid WAV in worker'}
    assert output['summary']['classified'] == 1
    assert output['summary']['failed'] == 1


def test_empty_preprocessing_outputs_are_reported_without_inference(monkeypatch, tmp_path):
    (wav,) = make_wavs(tmp_path, 1)
    monkeypatch.setattr('batbot.api.pipeline', lambda *args, **kwargs: ([], [], None))
    runner = Runner()

    output = bulk.classify_bulk([wav], progress=False, _runner=runner)

    assert runner.calls == []
    assert output['summary']['failed'] == 1
    assert output['results'][0]['error'] == f'No spectrograms were created for {wav}'


def test_generated_image_failure_only_fails_its_recording(monkeypatch, tmp_path):
    paths = make_wavs(tmp_path, 2)

    def prepare(path, output_folder):
        return [str(Path(output_folder) / f'{Path(path).stem}.jpg')]

    class FaultyRunner(Runner):
        def classify(self, paths, **kwargs):
            if any(Path(path).stem == 'input-00' for path in paths):
                raise OSError('unreadable image')
            return super().classify(paths, **kwargs)

    monkeypatch.setattr(bulk, '_prepare_wav', prepare)

    output = bulk.classify_bulk(paths, progress=False, _runner=FaultyRunner())

    assert output['results'][0] == {'path': str(paths[0]), 'error': 'unreadable image'}
    assert output['summary']['classified'] == 1
    assert output['summary']['failed'] == 1


def test_same_named_recordings_have_distinct_retained_outputs(monkeypatch, tmp_path):
    paths = [tmp_path / 'one' / 'call.wav', tmp_path / 'two' / 'call.wav']
    output_folder = tmp_path / 'spectrograms'
    for path in paths:
        path.parent.mkdir()
        path.touch()

    def pipeline(path, **kwargs):
        stem = Path(kwargs['out_file_stem'])
        stem.parent.mkdir(exist_ok=True)
        image = stem.with_name(f'{stem.name}.jpg')
        image.touch()
        return [str(image)], [], None

    monkeypatch.setattr('batbot.api.pipeline', pipeline)

    output = bulk.classify_bulk(
        paths, spectrogram_output=output_folder, progress=False, _runner=Runner()
    )

    images = [result['spectrogram_paths'][0] for result in output['results']]
    assert len(set(images)) == 2
    assert all(Path(image).exists() for image in images)


@pytest.mark.parametrize('workers', [0, -1])
def test_rejects_invalid_preprocessing_worker_counts(workers):
    with pytest.raises(ValueError, match='preprocess_workers'):
        bulk.classify_bulk([], preprocess_workers=workers)


@pytest.mark.parametrize('show_progress', [True, False])
def test_cli_keeps_json_on_stdout_with_optional_progress(monkeypatch, tmp_path, show_progress):
    (wav,) = make_wavs(tmp_path, 1)
    monkeypatch.setattr(bulk, 'Classifier', Runner)
    monkeypatch.setattr(
        bulk,
        '_prepare_wav',
        lambda path, folder: [str(Path(folder) / 'generated.jpg')],
    )

    result = CliRunner().invoke(
        classify_bulk,
        [str(wav)] + ([] if show_progress else ['--no-progress']),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)['summary']['classified'] == 1
    if show_progress:
        assert 'Creating spectrograms' in result.stderr
        assert 'Classifying files' in result.stderr
        assert '100%' in result.stderr
    else:
        assert result.stderr == ''


def test_real_spawned_preprocessing_matches_serial_inference(tmp_path):
    """Exercise real worker imports, spectrogram files, and the bundled model."""
    from scipy.io import wavfile

    source = Path(__file__).resolve().parents[1] / 'examples' / 'example1.wav'
    rate, samples = wavfile.read(source)
    paths = make_wavs(tmp_path, 2)
    for path in paths:
        wavfile.write(path, rate, samples[: rate // 5])
    runner = classifier.Classifier(num_workers=2, providers=['CPUExecutionProvider'])

    serial = runner.classify_bulk(paths, preprocess_workers=1, progress=False)
    parallel = runner.classify_bulk(paths, preprocess_workers=2, progress=False)

    assert serial['summary']['classified'] == 2
    assert parallel == serial
    assert len(runner._sessions) == 1
