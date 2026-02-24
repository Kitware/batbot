import os

from click.testing import CliRunner

from batbot.batbot_cli import preprocess


def test_preprocess_parallel():
    runner = CliRunner()
    data = runner.invoke(
        preprocess,
        ['examples', '-o', './output', '--process-metadata', '--force-overwrite', '-n', 4],
    )
    # parse stdout to ensure example files were processed properly
    num_examples = 4
    output_str = str(data.output).split('\n')
    for ii in range(num_examples):
        expected_file = './output/example{}.01of01.compressed.jpg'.format(ii + 1)
        assert any(
            [expected_file in x for x in output_str]
        ), 'Did not find file listed among outputs: {}'.format(expected_file)
        assert os.path.exists(expected_file), 'Did not find file in filesystem: {}'.format(
            expected_file
        )
    for ii in range(num_examples):
        expected_file = './output/example{}.metadata.json'.format(ii + 1)
        assert any(
            [expected_file in x for x in output_str]
        ), 'Did not find file listed among outputs: {}'.format(expected_file)
        assert os.path.exists(expected_file), 'Did not find file in filesystem: {}'.format(
            expected_file
        )
