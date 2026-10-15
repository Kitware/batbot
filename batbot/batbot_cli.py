#!/usr/bin/env python
"""
CLI for BatBot
"""

import json
import pprint
import warnings
from glob import escape, glob
from os import getcwd, makedirs, remove
from os.path import (
    basename,
    exists,
    join,
)
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

import batbot
from batbot import classifier
from batbot._config import log
from batbot._paths import discover_files, native_path, output_stems
from batbot.classifier.config import WAV_EXTENSIONS


class _NativePath(click.Path):
    def convert(self, value, param, ctx):
        return super().convert(str(native_path(value)), param, ctx)


def pipeline_filepath_validator(ctx, param, value):
    value = str(native_path(value))
    if not exists(value):
        raise click.BadParameter(f'Input filepath does not exist: {value}')
    return value


@click.command('fetch')
@click.option(
    '--config',
    help='Which ML model to use for inference',
    default=None,
    type=click.Choice(['mobilenet']),
)
@click.option(
    '--pull',
    is_flag=True,
    help='Download the mirrored model even when a bundled copy is available.',
)
def fetch(config, pull):
    """
    Fetch the required machine learning ONNX model for the classifier
    """
    print(batbot.fetch(config=config, pull=pull))


@click.command('pipeline')
@click.argument(
    'filepath',
    nargs=1,
    type=str,
    callback=pipeline_filepath_validator,
)
# @click.option(
#     '--config',
#     help='Which ML model to use for inference',
#     default=None,
#     type=click.Choice(['usgs']),
# )
@click.option(
    '--output',
    'output_path',
    help='Path to output folder for the results',
    default='.',
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
    # config,
    output_path,
    # classifier_thresh,
):

    batbot.pipeline(
        filepath,
        # config=config,
        # classifier_thresh=classifier_thresh,
        output_folder=output_path,
    )


@click.command('preprocess')
@click.argument(
    'filepaths',
    nargs=-1,
    type=str,
)
@click.option(
    '--output-dir',
    '-o',
    help='Processed file root output directory. Outputs will attempt to mirror input file directory structure if given multiple inputs (unless --no-file-structure flag is given). Defaults to current working directory.',
    nargs=1,
    default='.',
    type=str,
)
@click.option(
    '--process-metadata',
    '-m',
    help='Use a slower version of the pipeline which increases spectogram compression quality and also outputs bat call metadata.',
    is_flag=True,
)
@click.option(
    '--force-overwrite',
    '-f',
    help='Force overwriting of compressed spectrogram and other output files.',
    is_flag=True,
)
@click.option(
    '--num-workers',
    '-n',
    help='Number of parallel workers to use. Set to zero for serial computation only.',
    nargs=1,
    default=0,
    type=int,
)
@click.option(
    '--output-json',
    help='Path to output JSON (if unspecified, output file locations are printed to screen).',
    default=None,
    type=str,
)
@click.option(
    '--dry-run',
    '-d',
    help='List out all the audio files to be loaded and all the anticipated output files. Additionally lists all "extra" files in the output directory that would be deleted if using the --cleanup flag.',
    is_flag=True,
)
@click.option(
    '--cleanup',
    help='For the given input filepaths and --output-dir arguments, delete any extra files that would not have been created by the batbot preprocess. Skips hidden files starting with ".". Acts as if --force-overwrite flag is given (does not delete existing, preprocessed outputs). WARNING: This will delete files, recommend running with the --dry-run flag first and carefully examining the output!',
    is_flag=True,
)
@click.option(
    '--no-file-structure',
    help='Write outputs directly into the output directory. Duplicate filenames are rejected.',
    is_flag=True,
)
def preprocess(
    filepaths,
    output_dir,
    process_metadata,
    force_overwrite,
    num_workers,
    output_json,
    dry_run,
    cleanup,
    no_file_structure,
):
    """Generate compressed spectrogram images for input wav files into the same or a specified output directory.
    Takes one or more space separated arguments of filepaths to process. If given a directory name argument,
    will recursively search through the directory and all subfolders to find all contained \\*.wav files.
    Alternatively, the argument can be given as a string using wildcard \\** for folders and/or \\* in filenames
    (if \\** wildcard is used, will recursively search through all subfolders).

    \b
    Examples:
        batbot preprocess ../data -o ./tmp
        batbot preprocess "../data/\\**/\\*.wav"
        batbot preprocess ../data -o ./tmp -n 32
        batbot preprocess ../data -o ./tmp -n 32 -fm
        batbot preprocess ../data -o ./tmp -f --dry-run --output-json dry_run.json
        batbot preprocess ../data -o ./tmp --cleanup
    """
    in_filepaths = discover_files(filepaths, WAV_EXTENSIONS, patterns=True)

    if len(in_filepaths) == 0:
        print(f'Found no files given filepaths input {filepaths}')
        return

    # set up output paths for each input path
    root_outpath = '.' if output_dir is None else str(native_path(output_dir))
    makedirs(root_outpath, exist_ok=True)
    try:
        out_filepath_stems = output_stems(in_filepaths, root_outpath, flatten=no_file_structure)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    for stem in out_filepath_stems:
        Path(stem).parent.mkdir(parents=True, exist_ok=True)

    # look for existing output files and remove from the set
    in_filepaths = np.array(in_filepaths)
    if dry_run or cleanup:
        # save copy of all outputs before removing already processed data
        out_filepath_stems_all = out_filepath_stems.copy()
    out_filepath_stems = np.array(out_filepath_stems)
    if not force_overwrite:
        idx_remove = np.full((len(in_filepaths),), False)
        for ii, out_file_stem in enumerate(out_filepath_stems):
            test_file = f'{escape(str(out_file_stem))}.*'
            test_glob = glob(test_file)
            if len(test_glob) > 0:
                idx_remove[ii] = True
        in_filepaths = in_filepaths[np.invert(idx_remove)]
        out_filepath_stems = out_filepath_stems[np.invert(idx_remove)]
        n_skipped = sum(idx_remove)
        if len(in_filepaths) == 0:
            print(
                'Found no unprocessed files given filepaths input {} and output directory "{}" after skipping {} files'.format(
                    filepaths, root_outpath, n_skipped
                )
            )
            print('If desired, use --force-overwrite flag to overwrite existing processed data')
            return

    if dry_run or cleanup:
        # Find all "extra" files that would be deleted in cleanup mode
        all_files = set(glob(join(escape(root_outpath), '**/*'), recursive=True))
        for out_stem in out_filepath_stems_all:
            out_files = glob(f'{escape(str(out_stem))}.*')
            all_files -= set(out_files)
        dir_files = []
        # remove directories
        for file in all_files:
            if Path(file).is_dir():
                dir_files.append(file)
        all_files -= set(dir_files)
        extra_files = all_files

    print(f'Located {len(in_filepaths)} total unprocessed files')
    print('\tFast processing mode {}'.format('OFF' if process_metadata else 'ON'))
    if process_metadata:
        print('\t\tFull bat call metadata will be produced')
    print('\tForce output overwrite {}'.format('ON' if force_overwrite else 'OFF'))
    if not force_overwrite:
        print(f'\t\tSkipped {n_skipped} files with already preprocessed outputs')
    print(f'\tNum parallel workers: {num_workers}')
    if no_file_structure:
        print('\tFlattening output file structure')
    print(f'\tCurrent working dir: {getcwd()}')
    print(f'\tOutput root dir: {output_dir}')
    print(
        '\tFirst input file -> output files: {} -> {}.*'.format(
            in_filepaths[0], out_filepath_stems[0]
        )
    )
    if len(in_filepaths) > 2:
        print(
            '\tLast input file -> output files: {} -> {}.*'.format(
                in_filepaths[-1], out_filepath_stems[-1]
            )
        )

    if dry_run:
        # Print out files to be processed, anticipated outputs, and files that would be deleted in cleanup mode.
        print('\nDry run mode active - skipping all processing')
        data = {}
        data['input file, output file stem'] = [
            (str(x), f'{y}.*') for x, y in zip(in_filepaths, out_filepath_stems)
        ]
        data['files to be deleted in cleanup'] = list(extra_files)
        if output_json is None:
            pprint.pp(data)
        else:
            with native_path(output_json).open('w', encoding='utf8') as outfile:
                json.dump(data, outfile, indent=4)
            print(f'Outputs written to {output_json}')
        print('Complete.')
        return

    if cleanup:
        print('\nCleanup mode active - skipping all processing')
        if len(extra_files) == 0:
            print('No files to delete')
        else:
            usr_in = input(
                'Found {} files to delete (recommend to see details by running with --dry-run flag). Continue (y/n)? '.format(
                    len(extra_files)
                )
            )
            if usr_in.lower() not in ['y', 'yes']:
                print('Aborting cleanup mode.')
                return
        for file in extra_files:
            print(f'Deleting file: {file}')
            remove(file)
        print('Complete.')
        return

    # Begin execution loop.
    data = {
        'output_path': [],
        'compressed_path': [],
        'metadata_path': [],
        'failed_files': [],
    }
    if num_workers is None or num_workers == 0:

        # Serial execution.
        for file, out_stem in tqdm(
            zip(in_filepaths, out_filepath_stems),
            desc='Preprocessing files',
            total=len(in_filepaths),
        ):
            try:
                output_paths, compressed_paths, metadata_path = batbot.pipeline(
                    file,
                    out_file_stem=out_stem,
                    fast_mode=(not process_metadata),
                    force_overwrite=force_overwrite,
                    quiet=True,
                )
                data['output_path'].extend(output_paths)
                data['compressed_path'].extend(compressed_paths)
                data['metadata_path'].append(metadata_path)
            except Exception as e:
                warnings.warn(f'WARNING: Pipeline failed for file {file}')
                data['failed_files'].append((str(file), str(e)))
    else:
        # Parallel execution.
        # shuffle input and output paths
        zipped = np.stack((in_filepaths, out_filepath_stems), axis=-1)
        np.random.seed(0)
        np.random.shuffle(zipped)
        assert all(
            [
                x in zipped[:, 0] and y in zipped[:, 1]
                for x, y in zip(in_filepaths, out_filepath_stems)
            ]
        )
        in_filepaths, out_filepath_stems = zipped.T
        assert all([basename(y) in basename(x) for x, y in zip(in_filepaths, out_filepath_stems)])

        # make num_workers chunks
        in_file_chunks = np.array_split(in_filepaths, num_workers)
        out_stem_chunks = np.array_split(out_filepath_stems, num_workers)

        # send to parallel function
        output_paths, compressed_paths, metadata_paths, failed_files = batbot.parallel_pipeline(
            in_file_chunks=in_file_chunks,
            out_stem_chunks=out_stem_chunks,
            fast_mode=(not process_metadata),
            force_overwrite=force_overwrite,
            num_workers=num_workers,
            threaded=False,
            quiet=True,
            desc=f'Preprocessing chunks of files with {num_workers} workers',
        )
        data['output_path'].extend(output_paths)
        data['compressed_path'].extend(compressed_paths)
        data['metadata_path'].extend(metadata_paths)
        data['failed_files'].extend((path, str(error)) for path, error in failed_files)

    if output_json is None:
        print('\nFull spectrogram output paths:')
        pprint.pp(sorted(data['output_path']))
        print('\nCompressed spectrogram output paths:')
        pprint.pp(sorted(data['compressed_path']))
        print('\nProcessed metadata paths:')
        pprint.pp(sorted(data['metadata_path']))
        print('\nFiles skipped due to failure, and corresponding exceptions:')
        pprint.pp(sorted(data['failed_files']))
    else:
        with native_path(output_json).open('w', encoding='utf8') as outfile:
            json.dump(data, outfile, indent=4)
        print(f'Outputs written to {output_json}')
    print('\nComplete.')

    return data


def _inference_options(command):
    command = click.option(
        '--provider',
        type=click.Choice(['auto', 'cpu', 'cuda', 'coreml']),
        default='auto',
        show_default=True,
        help='Inference device. Explicit CUDA selection fails if GPU initialization fails.',
    )(command)
    return click.option(
        '--progress/--no-progress',
        default=True,
        show_default=True,
        help='Show processing progress on stderr.',
    )(command)


def _wav_inference_options(command):
    command = click.option(
        '--preprocess-workers',
        default=1,
        show_default=True,
        type=click.IntRange(min=1),
        help='Number of processes generating WAV spectrograms before batch inference.',
    )(command)
    return _inference_options(command)


def _execution_providers(provider):
    if provider == 'auto':
        return None
    return [
        {
            'cpu': 'CPUExecutionProvider',
            'cuda': 'CUDAExecutionProvider',
            'coreml': 'CoreMLExecutionProvider',
        }[provider]
    ]


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
    type=click.Choice(['mobilenet']),
)
@click.option(
    '--output',
    help='Path to output JSON (if unspecified, results are printed to screen)',
    default=None,
    type=str,
)
@click.option(
    '--num-workers',
    default=1,
    show_default=True,
    type=click.IntRange(min=1),
    help='Number of concurrent ONNX inference workers.',
)
@click.option(
    '--spectrogram-dir',
    default=None,
    type=_NativePath(file_okay=False),
    help='Keep generated spectrograms in this directory.',
)
@_wav_inference_options
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
    num_workers,
    spectrogram_dir,
    preprocess_workers,
    progress,
    provider,
    # classifier_thresh,
):
    """Classify a list of WAV files (legacy alias for ``classify-wav``)."""
    if config is not None:
        config = config.strip().lower()
    # classifier_thresh /= 100.0

    log.debug(f'Running batch on {len(filepaths)} files...')

    results = batbot.batch(
        filepaths,
        config=config,
        num_workers=num_workers,
        preprocess_workers=preprocess_workers,
        progress=progress,
        providers=_execution_providers(provider),
        spectrogram_output=spectrogram_dir,
    )

    data = {
        'results': results,
        'summary': classifier.summarize(results),
    }

    log.debug('Outputting results...')
    if output:
        with native_path(output).open('w', encoding='utf8') as outfile:
            json.dump(data, outfile, indent=4)
    else:
        print(json.dumps(data, indent=2))


def _write_classification_output(data, output):
    encoded = json.dumps(data, indent=2)
    if output:
        with native_path(output).open('w', encoding='utf8') as outfile:
            outfile.write(encoded)
            outfile.write('\n')
        click.echo(f'Outputs written to {output}')
    else:
        click.echo(encoded)


@click.command('classify')
@click.argument('spectrograms', nargs=-1, required=True, type=_NativePath(exists=True))
@click.option(
    '--output',
    '-o',
    default=None,
    type=_NativePath(dir_okay=False),
    help='Path to output JSON (defaults to stdout).',
)
@click.option(
    '--batch-size',
    default=classifier.BATCH_SIZE,
    show_default=True,
    type=click.IntRange(min=1),
)
@click.option(
    '--num-workers',
    default=1,
    show_default=True,
    type=click.IntRange(min=1),
    help='Number of concurrent ONNX inference workers.',
)
@click.option('--top-k', default=5, show_default=True, type=click.IntRange(1, 35))
@_inference_options
def classify(spectrograms, output, batch_size, num_workers, top_k, progress, provider):
    """Classify one or more spectrogram image files."""
    paths = classifier.discover_inputs(spectrograms, input_type='spectrogram')
    results = classifier.classify(
        paths,
        batch_size=batch_size,
        top_k=top_k,
        num_workers=num_workers,
        progress=progress,
        providers=_execution_providers(provider),
    )
    data = {'results': results, 'summary': classifier.summarize(results)}
    _write_classification_output(data, output)


@click.command('classify-wav')
@click.argument('wav_files', nargs=-1, required=True, type=_NativePath(exists=True))
@click.option(
    '--output',
    '-o',
    default=None,
    type=_NativePath(dir_okay=False),
    help='Path to output JSON (defaults to stdout).',
)
@click.option(
    '--spectrogram-dir',
    default=None,
    type=_NativePath(file_okay=False),
    help='Keep generated spectrograms in this directory.',
)
@click.option(
    '--batch-size',
    default=classifier.BATCH_SIZE,
    show_default=True,
    type=click.IntRange(min=1),
)
@click.option(
    '--num-workers',
    default=1,
    show_default=True,
    type=click.IntRange(min=1),
    help='Number of concurrent ONNX inference workers.',
)
@click.option('--top-k', default=5, show_default=True, type=click.IntRange(1, 35))
@_wav_inference_options
def classify_wav(
    wav_files,
    output,
    spectrogram_dir,
    batch_size,
    num_workers,
    top_k,
    preprocess_workers,
    progress,
    provider,
):
    """Create spectrograms from WAV files and classify each recording."""
    data = classifier.classify_bulk(
        wav_files,
        input_type='wav',
        batch_size=batch_size,
        top_k=top_k,
        spectrogram_output=spectrogram_dir,
        num_workers=num_workers,
        preprocess_workers=preprocess_workers,
        progress=progress,
        providers=_execution_providers(provider),
    )
    _write_classification_output(data, output)


@click.command('classify-bulk')
@click.argument('inputs', nargs=-1, required=True, type=_NativePath(exists=True))
@click.option(
    '--input-type',
    default='auto',
    show_default=True,
    type=click.Choice(['auto', 'spectrogram', 'wav']),
    help='Limit inputs to spectrograms, WAV files, or detect both.',
)
@click.option('--recursive/--no-recursive', default=True, show_default=True)
@click.option(
    '--output',
    '-o',
    default=None,
    type=_NativePath(dir_okay=False),
    help='Path to output JSON (defaults to stdout).',
)
@click.option(
    '--spectrogram-dir',
    default=None,
    type=_NativePath(file_okay=False),
    help='Keep spectrograms generated for WAV inputs in this directory.',
)
@click.option(
    '--batch-size',
    default=classifier.BATCH_SIZE,
    show_default=True,
    type=click.IntRange(min=1),
)
@click.option(
    '--num-workers',
    default=1,
    show_default=True,
    type=click.IntRange(min=1),
    help='Number of concurrent ONNX inference workers.',
)
@click.option('--top-k', default=5, show_default=True, type=click.IntRange(1, 35))
@_wav_inference_options
def classify_bulk(
    inputs,
    input_type,
    recursive,
    output,
    spectrogram_dir,
    batch_size,
    num_workers,
    top_k,
    preprocess_workers,
    progress,
    provider,
):
    """Recursively classify a large folder and report species counts."""
    data = classifier.classify_bulk(
        inputs,
        input_type=input_type,
        recursive=recursive,
        batch_size=batch_size,
        top_k=top_k,
        spectrogram_output=spectrogram_dir,
        num_workers=num_workers,
        preprocess_workers=preprocess_workers,
        progress=progress,
        providers=_execution_providers(provider),
    )
    _write_classification_output(data, output)


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
cli.add_command(classify)
cli.add_command(classify_wav)
cli.add_command(classify_bulk)
cli.add_command(example)


if __name__ == '__main__':
    cli()
