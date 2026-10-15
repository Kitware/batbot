"""Portable paths, image codecs, and preprocessing destinations."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

import batbot
from batbot._paths import discover_files, native_path, output_stems, read_image, write_image
from batbot.batbot_cli import preprocess
from batbot.classifier import dataloader, model


def test_unicode_image_roundtrip_uses_native_paths(tmp_path):
    directory = tmp_path / 'résultats 空間 [batbot]'
    directory.mkdir()
    path = directory / 'chauve-souris.png'
    image = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)

    assert write_image(path, image)
    np.testing.assert_array_equal(read_image(path), image)
    np.testing.assert_array_equal(dataloader._load_image(path), image)


@pytest.mark.parametrize('payload', [b'', b'invalid image'])
def test_image_decoder_reports_corrupt_files(tmp_path, payload):
    path = tmp_path / 'invalid.png'
    path.write_bytes(payload)
    with pytest.raises(OSError, match='Unable to load spectrogram'):
        read_image(path)


def test_image_writer_reports_missing_parent(tmp_path):
    with pytest.raises(OSError, match='Unable to write image'):
        write_image(tmp_path / 'missing' / 'image.png', np.zeros((4, 5, 3), dtype=np.uint8))


def test_native_paths_expand_variables_and_home(monkeypatch, tmp_path):
    monkeypatch.setenv('BATBOT_TEST_ROOT', str(tmp_path))
    assert native_path('$BATBOT_TEST_ROOT/calls.wav') == tmp_path / 'calls.wav'
    if os.name == 'nt':
        assert native_path('%BATBOT_TEST_ROOT%/calls.wav') == tmp_path / 'calls.wav'
    assert native_path('~/calls.wav') == Path.home() / 'calls.wav'
    assert native_path(Path('data') / 'calls.wav') == Path('data') / 'calls.wav'


def test_discovery_preserves_literal_brackets_and_uppercase_extensions(tmp_path):
    root = tmp_path / 'calls [EPFU]'
    root.mkdir()
    wav = root / 'ONE.WAV'
    image = root / 'TWO.JpG'
    ignored = root / 'notes.txt'
    for path in (wav, image, ignored):
        path.touch()

    assert discover_files([root], {'.wav', '.jpg'}) == sorted([str(wav), str(image)])
    assert discover_files([wav], {'.wav'}, patterns=True) == [str(wav)]
    assert discover_files([root / '*.WAV'], {'.wav'}, patterns=True) == [str(wav)]


def test_single_absolute_input_keeps_its_filename_in_output_root(tmp_path):
    path = tmp_path / 'inputs' / 'one.WAV'
    destination = tmp_path / 'outputs'
    assert output_stems([path], destination) == [str(destination / 'one')]
    assert output_stems([path], destination, flatten=True) == [str(destination / 'one')]


def test_output_layout_mirrors_directories_and_rejects_flattened_collisions(tmp_path):
    paths = [tmp_path / 'one' / 'call.wav', tmp_path / 'two' / 'call.wav']
    destination = tmp_path / 'output'
    assert output_stems(paths, destination) == [
        str(destination / 'one' / 'call'),
        str(destination / 'two' / 'call'),
    ]
    with pytest.raises(ValueError, match='overwrite each other'):
        output_stems(paths, destination, flatten=True)


def test_inputs_on_different_drives_get_distinct_output_namespaces(monkeypatch, tmp_path):
    def different_drives(paths):
        raise ValueError('Paths are on different drives')

    monkeypatch.setattr(os.path, 'commonpath', different_drives)
    destination = tmp_path / 'output'
    stems = output_stems([tmp_path / 'a' / 'call.wav', tmp_path / 'b' / 'call.wav'], destination)

    assert len(set(stems)) == 2
    assert all(Path(stem).is_relative_to(destination) for stem in stems)
    assert all(Path(stem).name == 'call' for stem in stems)


@pytest.mark.parametrize('flatten', [False, True])
def test_preprocess_single_unicode_file_stays_inside_destination(monkeypatch, tmp_path, flatten):
    source = tmp_path / '呼叫 [EPFU].WAV'
    source.touch()
    destination = tmp_path / 'résultats'
    report = tmp_path / 'report.json'
    calls = []
    monkeypatch.setattr(
        batbot, 'pipeline', lambda path, **kwargs: calls.append(kwargs) or ([], [], None)
    )

    result = CliRunner().invoke(
        preprocess,
        [str(source), '-o', str(destination), '--output-json', str(report)]
        + (['--no-file-structure'] if flatten else []),
    )

    assert result.exit_code == 0, result.output
    assert calls[0]['out_file_stem'] == str(destination / source.stem)
    assert json.loads(report.read_text())['failed_files'] == []


def test_preprocess_error_report_is_json_serializable(monkeypatch, tmp_path):
    source = tmp_path / 'bad.WAV'
    source.touch()
    report = tmp_path / 'report.json'

    def fail(*args, **kwargs):
        raise OSError('bad recording')

    monkeypatch.setattr(batbot, 'pipeline', fail)
    with pytest.warns(UserWarning, match='Pipeline failed'):
        result = CliRunner().invoke(
            preprocess, [str(source), '-o', str(tmp_path / 'output'), '--output-json', str(report)]
        )

    assert result.exit_code == 0, result.output
    assert json.loads(report.read_text())['failed_files'] == [[str(source), 'bad recording']]


def test_model_cache_can_use_a_platform_native_override(monkeypatch, tmp_path):
    monkeypatch.setenv('BATBOT_TEST_ROOT', str(tmp_path))
    monkeypatch.setenv('BATBOT_CACHE_DIR', '$BATBOT_TEST_ROOT/cache 空間')
    assert model._cache_directory() == tmp_path / 'cache 空間' / 'models' / model.__version__


def test_cleanup_plan_handles_literal_output_directory_brackets(tmp_path):
    source = tmp_path / 'call.WAV'
    source.touch()
    destination = tmp_path / 'outputs [EPFU]'
    destination.mkdir()
    expected = destination / 'call.compressed.jpg'
    extra = destination / 'extra.jpg'
    expected.touch()
    extra.touch()
    report = tmp_path / 'plan.json'

    result = CliRunner().invoke(
        preprocess,
        [
            str(source),
            '-o',
            str(destination),
            '--dry-run',
            '--force-overwrite',
            '--output-json',
            str(report),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(report.read_text())['files to be deleted in cleanup'] == [str(extra)]
    assert expected.exists() and extra.exists()
