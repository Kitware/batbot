import json
import warnings
from pathlib import Path

from click.testing import CliRunner

import batbot
from batbot.batbot_cli import batch, cli, example, fetch, pipeline, preprocess


def _result(path):
    return {
        'path': str(path),
        'label': 'EPFU',
        'confidence': 0.75,
        'window_count': 1,
        'top': [{'label': 'EPFU', 'confidence': 0.75}],
        'scores': {'EPFU': 0.75},
    }


def _wav_files(tmp_path):
    input_directory = tmp_path / 'input'
    input_directory.mkdir()
    paths = [input_directory / 'one.wav', input_directory / 'two.wav']
    for path in paths:
        path.touch()
    return input_directory, paths


def test_fetch_cli_forwards_configuration_and_pull(monkeypatch):
    calls = []
    monkeypatch.setattr(
        batbot,
        'fetch',
        lambda **kwargs: calls.append(kwargs) or '/models/batbot.onnx',
    )

    invocation = CliRunner().invoke(fetch, ['--config', 'mobilenet', '--pull'])

    assert invocation.exit_code == 0, invocation.output
    assert invocation.output.strip() == '/models/batbot.onnx'
    assert calls == [{'config': 'mobilenet', 'pull': True}]


def test_pipeline_cli_validates_input_and_forwards_output(monkeypatch, tmp_path):
    wav = tmp_path / 'recording.wav'
    wav.touch()
    output = tmp_path / 'output'
    calls = []
    monkeypatch.setattr(batbot, 'pipeline', lambda *args, **kwargs: calls.append((args, kwargs)))

    invocation = CliRunner().invoke(pipeline, [str(wav), '--output', str(output)])
    missing = CliRunner().invoke(pipeline, [str(tmp_path / 'missing.wav')])

    assert invocation.exit_code == 0, invocation.output
    assert calls == [((str(wav),), {'output_folder': str(output)})]
    assert missing.exit_code == 2
    assert 'Input filepath does not exist' in missing.output


def test_batch_cli_writes_stdout_and_json(monkeypatch, tmp_path):
    wav = tmp_path / 'recording.wav'
    wav.touch()
    output = tmp_path / 'batch.json'
    calls = []

    def fake_batch(paths, **kwargs):
        calls.append((paths, kwargs))
        return [_result(paths[0])]

    monkeypatch.setattr(batbot, 'batch', fake_batch)

    printed = CliRunner().invoke(batch, [str(wav), '--config', 'mobilenet'])
    written = CliRunner().invoke(batch, [str(wav), '--output', str(output)])

    assert printed.exit_code == 0, printed.output
    assert json.loads(printed.output)['summary']['species_counts'] == {'EPFU': 1}
    assert written.exit_code == 0, written.output
    assert json.loads(output.read_text())['results'][0]['path'] == str(wav)
    assert calls[0][1] == {'config': 'mobilenet'}
    assert calls[1][1] == {'config': None}


def test_example_and_root_cli(monkeypatch):
    calls = []
    monkeypatch.setattr(batbot, 'example', lambda: calls.append(True))

    example_result = CliRunner().invoke(example)
    help_result = CliRunner().invoke(cli, ['--help'])

    assert example_result.exit_code == 0, example_result.output
    assert calls == [True]
    assert help_result.exit_code == 0
    assert 'classify-bulk' in help_result.output


def test_preprocess_reports_when_no_inputs_exist(tmp_path):
    invocation = CliRunner().invoke(preprocess, [str(tmp_path / '*.wav')])

    assert invocation.exit_code == 0, invocation.output
    assert 'Found no files' in invocation.output


def test_preprocess_dry_run_writes_plan_without_processing(monkeypatch, tmp_path):
    input_directory, paths = _wav_files(tmp_path)
    output_directory = tmp_path / 'output'
    output_directory.mkdir()
    stale = output_directory / 'stale.txt'
    stale.touch()
    report = tmp_path / 'dry-run.json'
    monkeypatch.setattr(
        batbot,
        'pipeline',
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('pipeline was called')),
    )

    invocation = CliRunner().invoke(
        preprocess,
        [
            str(input_directory),
            '--output-dir',
            str(output_directory),
            '--force-overwrite',
            '--dry-run',
            '--output-json',
            str(report),
        ],
    )

    assert invocation.exit_code == 0, invocation.output
    data = json.loads(report.read_text())
    assert [pair[0] for pair in data['input file, output file stem']] == [
        str(path) for path in paths
    ]
    assert data['files to be deleted in cleanup'] == [str(stale)]
    assert 'Dry run mode active' in invocation.output


def test_preprocess_dry_run_prints_plan_and_flattens_structure(tmp_path):
    _, paths = _wav_files(tmp_path)
    output_directory = tmp_path / 'output'

    invocation = CliRunner().invoke(
        preprocess,
        [
            *(str(path) for path in paths),
            '--output-dir',
            str(output_directory),
            '--force-overwrite',
            '--dry-run',
            '--no-file-structure',
        ],
    )

    assert invocation.exit_code == 0, invocation.output
    assert 'Flattening output file structure' in invocation.output
    assert 'files to be deleted in cleanup' in invocation.output


def test_preprocess_skips_files_with_existing_outputs(tmp_path):
    input_directory, _ = _wav_files(tmp_path)
    output_directory = tmp_path / 'output'
    output_directory.mkdir()
    (output_directory / 'one.jpg').touch()
    (output_directory / 'two.jpg').touch()

    invocation = CliRunner().invoke(
        preprocess,
        [str(input_directory), '--output-dir', str(output_directory)],
    )

    assert invocation.exit_code == 0, invocation.output
    assert 'Found no unprocessed files' in invocation.output
    assert 'use --force-overwrite' in invocation.output


def test_preprocess_serial_writes_pipeline_results(monkeypatch, tmp_path):
    input_directory, paths = _wav_files(tmp_path)
    output_directory = tmp_path / 'output'
    report = tmp_path / 'results.json'
    calls = []

    def fake_pipeline(filepath, **kwargs):
        calls.append((filepath, kwargs))
        stem = str(kwargs['out_file_stem'])
        return [f'{stem}.png'], [f'{stem}.compressed.jpg'], f'{stem}.json'

    monkeypatch.setattr(batbot, 'pipeline', fake_pipeline)

    invocation = CliRunner().invoke(
        preprocess,
        [
            str(input_directory),
            '--output-dir',
            str(output_directory),
            '--force-overwrite',
            '--process-metadata',
            '--output-json',
            str(report),
        ],
    )

    assert invocation.exit_code == 0, invocation.output
    data = json.loads(report.read_text())
    assert len(calls) == len(paths)
    assert all(call[1]['fast_mode'] is False for call in calls)
    assert len(data['output_path']) == len(paths)
    assert data['failed_files'] == []


def test_preprocess_serial_retains_individual_failures(monkeypatch, tmp_path):
    input_directory, _ = _wav_files(tmp_path)

    def fake_pipeline(filepath, **kwargs):
        if Path(filepath).name == 'two.wav':
            raise ValueError('corrupt WAV')
        return ['one.png'], ['one.compressed.jpg'], 'one.json'

    monkeypatch.setattr(batbot, 'pipeline', fake_pipeline)

    with warnings.catch_warnings(record=True) as caught:
        invocation = CliRunner().invoke(
            preprocess,
            [str(input_directory), '--force-overwrite'],
        )

    assert invocation.exit_code == 0, invocation.output
    assert 'corrupt WAV' in invocation.output
    assert any('Pipeline failed' in str(warning.message) for warning in caught)


def test_preprocess_parallel_delegates_chunked_work(monkeypatch, tmp_path):
    input_directory, _ = _wav_files(tmp_path)
    captured = []

    def fake_parallel(**kwargs):
        captured.append(kwargs)
        return ['one.png'], ['one.compressed.jpg'], ['one.json'], []

    monkeypatch.setattr(batbot, 'parallel_pipeline', fake_parallel)

    invocation = CliRunner().invoke(
        preprocess,
        [str(input_directory), '--force-overwrite', '--num-workers', '2'],
    )

    assert invocation.exit_code == 0, invocation.output
    assert len(captured) == 1
    assert captured[0]['num_workers'] == 2
    assert len(captured[0]['in_file_chunks']) == 2
    assert captured[0]['fast_mode'] is True
    assert 'one.compressed.jpg' in invocation.output


def test_preprocess_cleanup_can_abort_or_delete(monkeypatch, tmp_path):
    input_directory, _ = _wav_files(tmp_path)
    output_directory = tmp_path / 'output'
    output_directory.mkdir()
    stale = output_directory / 'stale.txt'
    stale.touch()
    monkeypatch.setattr(
        batbot,
        'pipeline',
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('pipeline was called')),
    )
    arguments = [
        str(input_directory),
        '--output-dir',
        str(output_directory),
        '--force-overwrite',
        '--cleanup',
    ]

    aborted = CliRunner().invoke(preprocess, arguments, input='n\n')

    assert aborted.exit_code == 0, aborted.output
    assert 'Aborting cleanup mode' in aborted.output
    assert stale.exists()

    deleted = CliRunner().invoke(preprocess, arguments, input='yes\n')

    assert deleted.exit_code == 0, deleted.output
    assert f'Deleting file: {stale}' in deleted.output
    assert not stale.exists()


def test_preprocess_cleanup_handles_no_extra_files(monkeypatch, tmp_path):
    input_directory, _ = _wav_files(tmp_path)
    output_directory = tmp_path / 'output'
    monkeypatch.setattr(
        batbot,
        'pipeline',
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('pipeline was called')),
    )

    invocation = CliRunner().invoke(
        preprocess,
        [
            str(input_directory),
            '--output-dir',
            str(output_directory),
            '--force-overwrite',
            '--cleanup',
        ],
    )

    assert invocation.exit_code == 0, invocation.output
    assert 'No files to delete' in invocation.output
