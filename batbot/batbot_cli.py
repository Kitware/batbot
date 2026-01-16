#!/usr/bin/env python
"""
CLI for BatBot
"""
from glob import glob
import json
from os.path import exists
import warnings

import click

import batbot
from batbot import log

from tqdm import tqdm


def pipeline_filepath_validator(ctx, param, value):
    if not exists(value):
        log.error(f'Input filepath does not exist: {value}')
        ctx.exit()
    return value


@click.command('fetch')
@click.option(
    '--config',
    help='Which ML model to use for inference',
    default=None,
    type=click.Choice(['usgs']),
)
def fetch(config):
    """
    Fetch the required machine learning ONNX model for the classifier
    """
    batbot.fetch(config=config)


@click.command('pipeline')
@click.argument(
    'filepath',
    nargs=1,
    type=str,
    callback=pipeline_filepath_validator,
)
@click.option(
    '--config',
    help='Which ML model to use for inference',
    default=None,
    type=click.Choice(['usgs']),
)
@click.option(
    '--output',
    help='Path to output JSON (if unspecified, results are printed to screen)',
    default=None,
    type=str,
)
# @click.option(
#     '--classifier_thresh',
#     help='Classifier confidence threshold',
#     default=int(classifier.CONFIGS[None]['thresh'] * 100),
#     type=click.IntRange(0, 100, clamp=True),
# )
def pipeline(
    filepath,
    config,
    output,
    # classifier_thresh,
):
    """
    Run the BatBot pipeline on an input WAV filepath.  An example output of the JSON
    can be seen below.

    .. code-block:: javascript

            {
                '/path/to/file.wav': {
                    'classifier': 0.5,
                }
            }
    """
    if config is not None:
        config = config.strip().lower()
    # classifier_thresh /= 100.0

    score = batbot.pipeline(
        filepath,
        config=config,
        # classifier_thresh=classifier_thresh,
    )

    data = {
        filepath: {
            'classifier': score,
        }
    }

    log.debug('Outputting results...')
    if output:
        with open(output, 'w') as outfile:
            json.dump(data, outfile)
    else:
        print(data)

@click.command('preprocess')
@click.argument(
    'filepaths',
    nargs=-1,
    type=str,
)
# @click.option(
#     '--output-dir',
#     help='Processed file output directory. Defaults to current working directory.',
#     default='.',
#     type=str,
# )
@click.option(
    '--metadata', '-m',
    help='Use a much slower version of the pipeline which increases spectogram compression quality and outputs additional bat call metadata.',
    is_flag=True,
)
@click.option(
    '--output-json',
    help='Path to output JSON (if unspecified, output file locations are printed to screen)',
    default=None,
    type=str,
)
def preprocess(filepaths, metadata, output_json):
    """Generate compressed spectrogram images for wav files into the current working directory. 
    Takes one or more space separated arguments of filepaths to process.
    Filepaths can use wildcards ** for folders and/or * within filenames (if ** wildcard is used, 
    will recursively search through all subfolders).
    """
    all_filepaths = []
    for file in filepaths:
        all_filepaths.extend(glob(file, recursive=True))
    # remove any repeats
    all_filepaths = sorted(list(set(all_filepaths)))

    if len(all_filepaths) == 0:
        print('Found no files given filepaths input {}'.format(filepaths))
        return

    print('Running preprocessing on {} located files'.format(len(all_filepaths)))
    print('\tFast processing mode {}'.format('OFF' if metadata else 'ON'))
    print('\tName of first file to process: {}'.format(all_filepaths[0]))
    if len(all_filepaths) > 2:
        print('\tName of last file to process: {}'.format(all_filepaths[-1]))
    
    data = {'output_path':[], 'metadata_paths':[]}
    for file in tqdm(all_filepaths, desc='Preprocessing files', total=len(all_filepaths)):
        try:
            output_paths, metadata_path = batbot.pipeline(file, fast_mode=(not metadata)) #, extra_arg=True)
            data['output_path'].extend(output_paths)
            if metadata:
                data['metadata_path'].extend(metadata_path)
        except:
            warnings.warn('WARNING: Pipeline failed for file {}'.format(file))

    if output_json is None:
        print('Processed output paths:')
        print(data['output_path'])
        if metadata:
            print('Processed metadata paths:')
            print(data['metadata_path'])
    else:
        with open(output_json, 'w') as outfile:
            json.dump(data, outfile)
    print('Complete.')

    return data


@click.command('batch')
@click.argument(
    'filepaths',
    nargs=-1,
    type=str,
)
@click.option(
    '--config',
    help='Which ML model to use for inference',
    default=None,
    type=click.Choice(['usgs']),
)
@click.option(
    '--output',
    help='Path to output JSON (if unspecified, results are printed to screen)',
    default=None,
    type=str,
)
# @click.option(
#     '--classifier_thresh',
#     help='Classifier confidence threshold',
#     default=int(classifier.CONFIGS[None]['thresh'] * 100),
#     type=click.IntRange(0, 100, clamp=True),
# )
def batch(
    filepaths,
    config,
    output,
    # classifier_thresh,
):
    """
    Run the BatBot pipeline in batch on a list of input WAV filepaths.
    An example output of the JSON can be seen below.

    .. code-block:: javascript

            {
                '/path/to/file1.wav': {
                    'classifier': 0.5,
                },
                '/path/to/file2.wav': {
                    'classifier': 0.8,
                },
                ...
            }
    """
    if config is not None:
        config = config.strip().lower()
    # classifier_thresh /= 100.0

    log.debug(f'Running batch on {len(filepaths)} files...')

    score_list = batbot.batch(
        filepaths,
        config=config,
        # classifier_thresh=classifier_thresh,
    )

    data = {}
    for filepath, score in zip(filepaths, score_list):
        data[filepath] = {
            'classifier': score,
        }

    log.debug('Outputting results...')
    if output:
        with open(output, 'w') as outfile:
            json.dump(data, outfile)
    else:
        print(data)


@click.command('example')
def example():
    """
    Run a test of the pipeline on an example WAV with the default configuration.
    """
    batbot.example()


@click.group()
def cli():
    """
    BatBot CLI
    """
    pass


cli.add_command(fetch)
cli.add_command(pipeline)
cli.add_command(preprocess)
cli.add_command(batch)
cli.add_command(example)


if __name__ == '__main__':
    cli()
