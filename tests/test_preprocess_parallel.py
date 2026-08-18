import os

from click.testing import CliRunner

from batbot.batbot_cli import preprocess


def test_preprocess_parallel():
    runner = CliRunner()
    data = runner.invoke(
        preprocess,
        ['examples', '-o', './output', '--process-metadata', '--force-overwrite', '-n', 2],
    )
    # parse stdout to ensure example files were processed properly
    # limiting to 2 examples for now
    num_examples = 2
    output_str = str(data.output).split('\n')
    for ii in range(num_examples):
        expected_file = f'./output/example{ii + 1}.01of01.compressed.jpg'
        assert any(
            [expected_file in x for x in output_str]
        ), f'Did not find file listed among outputs: {expected_file}'
        assert os.path.exists(expected_file), 'Did not find file in filesystem: {}'.format(
            expected_file
        )
    for ii in range(num_examples):
        expected_file = f'./output/example{ii + 1}.metadata.json'
        assert any(
            [expected_file in x for x in output_str]
        ), f'Did not find file listed among outputs: {expected_file}'
        assert os.path.exists(expected_file), 'Did not find file in filesystem: {}'.format(
            expected_file
        )
