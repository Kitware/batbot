import json
import os

from click.testing import CliRunner

from batbot.batbot_cli import preprocess


def test_preprocess():
    """Test of batbot preprocess CLI ensuring the example files are processed without error.
    Additionally, a regression test ensuring that the number of detected bat calls in the examples
    does not decrease. The minumum numbers of bat call segments listed below correspond to the number
    of bat call segments detected at the time of writing minus detected noise segments (noise counted by hand).
    Note that this test uses "fast mode" processing, which is more permissive of low-amplitude calls and noise.
    """
    runner = CliRunner()
    data = runner.invoke(preprocess, ['examples', '-o', './output', '--force-overwrite'])
    assert data.exit_code == 0
    # parse stdout to ensure example files were processed properly
    # limiting to 2 examples for now
    num_examples = 2
    output_str = str(data.output).split('\n')
    for ii in range(num_examples):
        expected_file = './output/example{}.01of01.compressed.jpg'.format(ii + 1)
        assert any(
            [expected_file in x for x in output_str]
        ), 'Did not find file listed among outputs: {}'.format(expected_file)
        assert os.path.exists(expected_file), 'Did not find file on filesystem: {}'.format(
            expected_file
        )
    num_min_call_segments = [65, 18, 149, 47]
    for ii in range(num_examples):
        expected_file = './output/example{}.metadata.json'.format(ii + 1)
        assert any(
            [expected_file in x for x in output_str]
        ), 'Did not find file listed among outputs: {}'.format(expected_file)
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
