import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from batbot.batbot_cli import preprocess


def test_preprocess_parallel(tmp_path):
    runner = CliRunner()
    source = Path(__file__).resolve().parents[1] / 'examples'
    report = tmp_path / 'report.json'
    data = runner.invoke(
        preprocess,
        [
            str(source / 'example1.wav'),
            str(source / 'example2.wav'),
            '-o',
            str(tmp_path),
            '--process-metadata',
            '--force-overwrite',
            '-n',
            2,
            '--output-json',
            str(report),
        ],
    )
    assert data.exit_code == 0, data.output
    results = json.loads(report.read_text(encoding='utf8'))
    assert results['failed_files'] == []
    # parse stdout to ensure example files were processed properly
    # limiting to 2 examples for now
    num_examples = 2
    for ii in range(num_examples):
        expected_file = str(tmp_path / f'example{ii + 1}.01of01.compressed.jpg')
        assert expected_file in results['compressed_path']
        assert os.path.exists(expected_file), 'Did not find file in filesystem: {}'.format(
            expected_file
        )
    for ii in range(num_examples):
        expected_file = str(tmp_path / f'example{ii + 1}.metadata.json')
        assert expected_file in results['metadata_path']
        assert os.path.exists(expected_file), 'Did not find file in filesystem: {}'.format(
            expected_file
        )
        metadata = json.loads(Path(expected_file).read_text())
        assert metadata['segments']
        assert metadata['sr.hz'] == 250_000
        assert metadata['duration.ms'] == pytest.approx(4999.022)
        assert metadata['size']['uncompressed']['height.px'] == len(
            metadata['frequencies']['pixels.hz']
        )
        assert metadata['spectrogram']['compressed.path'] == [
            str(tmp_path / f'example{ii + 1}.01of01.compressed.jpg')
        ]
        for segment in metadata['segments']:
            assert (
                0
                <= segment['segment start.ms']
                < segment['segment end.ms']
                <= metadata['duration.ms'] + 1
            )
            assert segment['lo f.hz'] <= segment['peak f.hz'] <= segment['hi f.hz']
