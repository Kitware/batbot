"""Tutorial utilities and source hygiene; optional visual tests run in notebook CI."""

import ast
import json
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.notebooks import notebook_utils as tutorial

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('path', sorted((ROOT / 'examples/notebooks').glob('*.ipynb')))
def test_notebook_sources_are_clean_and_compile(path):
    notebook = json.loads(path.read_text(encoding='utf8'))
    assert notebook['nbformat'] == 4
    assert len(notebook['cells']) > 10
    for cell in notebook['cells']:
        assert cell.get('id')
        assert not cell.get('attachments')
        if cell['cell_type'] == 'code':
            assert cell['outputs'] == []
            assert cell['execution_count'] is None
            ast.parse(''.join(cell['source']))


@pytest.mark.parametrize('major,expected', [(11, 'cpu'), (12, 'cuda12'), (13, 'cuda13')])
def test_runtime_auto_uses_driver_supported_major(monkeypatch, major, expected):
    monkeypatch.setattr(
        tutorial.subprocess,
        'run',
        lambda *a, **kw: SimpleNamespace(stdout=f'CUDA Version: {major}.0', returncode=0),
    )
    assert tutorial.runtime_profile('auto')[0] == expected


@pytest.mark.parametrize(
    'failure', [FileNotFoundError(), subprocess.TimeoutExpired('nvidia-smi', 15)]
)
def test_runtime_auto_without_driver_is_cpu(monkeypatch, failure):
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(tutorial.subprocess, 'run', fail)
    assert tutorial.runtime_profile('auto')[1] == 'CPUExecutionProvider'
    with pytest.raises(ValueError, match='Choose'):
        tutorial.runtime_profile('unsupported')


def test_runtime_install_never_overwrites_opposite_distribution(monkeypatch, tmp_path):
    monkeypatch.setattr(tutorial.importlib.metadata, 'version', lambda name: '1.31.0')
    monkeypatch.setattr(
        tutorial.subprocess, 'check_call', lambda *a: pytest.fail('must not install')
    )
    with pytest.raises(RuntimeError, match='fresh runtime'):
        tutorial.prepare_environment(tmp_path, 'cpu')


def test_runtime_install_uses_active_kernel_and_cuda12_pin(monkeypatch, tmp_path):
    calls = []

    def missing(name):
        raise tutorial.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(tutorial.importlib.metadata, 'version', missing)
    monkeypatch.delenv('BATBOT_NOTEBOOK_SKIP_INSTALL', raising=False)
    monkeypatch.setattr(tutorial.subprocess, 'check_call', lambda command: calls.append(command))
    assert tutorial.prepare_environment(tmp_path, 'cuda12')[1] == 'CUDAExecutionProvider'
    assert calls[0][0] == tutorial.sys.executable
    assert calls[0][-1] == 'onnxruntime-gpu[cuda,cudnn]==1.26.0'
    assert calls[0][-2] == f'{tmp_path}[notebook]'
    assert calls[1][-2:] == ['pip', 'check']


def test_input_discovery_bounds_and_missing_data(tmp_path):
    (tmp_path / 'a.WAV').touch()
    (tmp_path / 'b.JPG').touch()
    assert tutorial.choose_inputs(tmp_path, tmp_path, tmp_path, maximum=1) == [tmp_path / 'a.WAV']
    with pytest.raises(FileNotFoundError):
        tutorial.choose_inputs(tmp_path / 'missing', tmp_path, tmp_path, allow_example=False)
    with pytest.raises(ValueError):
        tutorial.choose_inputs(tmp_path, tmp_path, tmp_path, maximum=0)


def test_example_fallback_ignores_lfs_pointer(monkeypatch, tmp_path):
    import pooch

    example = tmp_path / 'examples/example1.wav'
    example.parent.mkdir()
    example.write_text('version https://git-lfs.github.com/spec/v1\n')
    calls = []
    monkeypatch.setattr(pooch, 'retrieve', lambda **kwargs: calls.append(kwargs) or str(example))
    assert tutorial.sample_wav(tmp_path, tmp_path / 'cache') == example
    assert calls[0]['known_hash'].startswith('sha256:')
    assert calls[0]['fname'] == 'example1.wav'


def test_archive_excludes_symlinks_and_keeps_nested_outputs(tmp_path):
    directory = tmp_path / 'run'
    (directory / 'nested').mkdir(parents=True)
    (directory / 'nested/report.json').write_text('{}')
    outside = tmp_path / 'private.txt'
    outside.write_text('not for sharing')
    try:
        (directory / 'external').symlink_to(outside)
    except OSError:
        pass  # Windows may require special privileges to create links.
    archive = tutorial.archive_outputs(directory)
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == ['nested/report.json']


def test_crop_validation_preserves_original(tmp_path):
    import numpy as np
    from scipy.io import wavfile

    source = tmp_path / 'original.WAV'
    wavfile.write(source, 250_000, np.zeros(25_000, dtype=np.int16))
    before = source.read_bytes()
    _, rate, output = tutorial.crop_recording(source, tmp_path / 'clip.wav', seconds=0.05)
    assert rate == 250_000 and output.is_file()
    assert source.read_bytes() == before
    with pytest.raises(ValueError, match='too short'):
        tutorial.crop_recording(source, output, start=2, seconds=0.05)
    with pytest.raises(ValueError, match='Choose'):
        tutorial.crop_recording(source, output, seconds=100)
    with pytest.raises(ValueError, match='overwrite'):
        tutorial.crop_recording(source, source, seconds=0.05)
    assert source.read_bytes() == before


@pytest.mark.parametrize('start,seconds', [(float('nan'), 1), (0, float('inf')), (-1, 1)])
def test_crop_rejects_invalid_intervals_before_reading(tmp_path, start, seconds):
    with pytest.raises(ValueError, match='Choose'):
        tutorial.crop_recording(tmp_path / 'missing.wav', tmp_path / 'clip.wav', start, seconds)


def test_empty_reports_render_and_export(tmp_path):
    pytest.importorskip('pandas')
    import matplotlib.pyplot as plt

    from batbot import classifier

    report = {
        'results': [{'path': 'broken.wav', 'error': 'bad input'}],
        'summary': classifier.summarize([{'path': 'broken.wav', 'error': 'bad input'}]),
    }
    table, scores = tutorial.result_tables(report)
    assert table.iloc[0]['error'] == 'bad input' and scores.empty
    figure = tutorial.plot_results(report, tmp_path / 'empty.png')
    assert (tmp_path / 'empty.png').stat().st_size > 0
    plt.close(figure)
