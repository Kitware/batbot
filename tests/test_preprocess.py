import json
import os
from pathlib import Path

from click.testing import CliRunner

from batbot.batbot_cli import preprocess


def test_preprocess(tmp_path):
    """Test of batbot preprocess CLI ensuring the example files are processed without error.
    Additionally, a regression test ensuring that the number of detected bat calls in the examples
    does not decrease. The minumum numbers of bat call segments listed below correspond to the number
    of bat call segments detected at the time of writing minus detected noise segments (noise counted by hand).
    Note that this test uses "fast mode" processing, which is more permissive of low-amplitude calls and noise.
    """
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
            '--force-overwrite',
            '--output-json',
            str(report),
        ],
    )
    assert data.exit_code == 0
    results = json.loads(report.read_text(encoding='utf8'))
    assert results['failed_files'] == []
    # parse stdout to ensure example files were processed properly
    # limiting to 2 examples for now
    num_examples = 2
    for ii in range(num_examples):
        expected_file = str(tmp_path / f'example{ii + 1}.01of01.compressed.jpg')
        assert expected_file in results['compressed_path']
        assert os.path.exists(expected_file), 'Did not find file on filesystem: {}'.format(
            expected_file
        )
    # These two fixture-specific lower bounds detect lost segments, not biological accuracy.
    num_min_call_segments = [65, 18]
    for ii in range(num_examples):
        expected_file = str(tmp_path / f'example{ii + 1}.metadata.json')
        assert expected_file in results['metadata_path']
        assert os.path.exists(expected_file), 'Did not find file on filesystem: {}'.format(
            expected_file
        )
        # load metadata file and ensure minimum number of call segments were detected
        with open(expected_file) as hf:
            data = json.load(hf)
        n_segments = len(data['segments'])
        err_str = (
            'Expected at least {} bat call segments in file {}, found only {} segments'.format(
                num_min_call_segments[ii], expected_file, n_segments
            )
        )
        assert n_segments >= num_min_call_segments[ii], err_str
