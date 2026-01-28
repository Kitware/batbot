#!/usr/bin/env python
"""
CLI for BatBot
"""
from glob import glob
import json
from os.path import exists, commonpath, join, relpath, split, splitext, basename, isdir, isfile
from os import makedirs, getcwd, remove
import warnings

import click
import numpy as np

import batbot
from batbot import log

from tqdm import tqdm


# warnings.filterwarnings("error")

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
            json.dump(data, outfile, indent=4)
    else:
        print(data)

@click.command('preprocess')
@click.argument(
    'filepaths',
    nargs=-1,
    type=str,
)
@click.option(
    '--output-dir', '-o',
    help='Processed file root output directory. Outputs will attempt to mirror input file directory structure if given multiple inputs (unless --no-file-structure flag is given). Defaults to current working directory.',
    nargs=1,
    default='.',
    type=str,
)
@click.option(
    '--process-metadata', '-m',
    help='Use a slower version of the pipeline which increases spectogram compression quality and also outputs bat call metadata.',
    is_flag=True,
)
@click.option(
    '--force-overwrite', '-f',
    help='Force overwriting of compressed spectrogram and other output files.',
    is_flag=True,
)
@click.option(
    '--num-workers', '-n',
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
    '--dry-run', '-d',
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
    help='(Not recommended) Turn off input file directory structure mirroring. All outputs will be written directly into the provided output dir. WARNING: If multiple input files have the same filename, outputs will overwrite!',
    is_flag=True,
)
def preprocess(filepaths, output_dir, process_metadata, force_overwrite, num_workers, output_json, dry_run, cleanup, no_file_structure):
    """Generate compressed spectrogram images for wav files into the current working directory. 
    Takes one or more space separated arguments of filepaths to process. If given a directory name,
    will recursively search through the directory and all subfolders to find all contained *.wav files.
    Alternatively, the argument can be given as a string using wildcard ** for folders and/or * in filenames
    (if ** wildcard is used, will recursively search through all subfolders).
    
    \b
    Examples:
        batbot preprocess ../data -o ./tmp
        batbot preprocess "../data/**/*.wav"
        batbot preprocess ../data -o ./tmp -n 32
        batbot preprocess ../data -o ./tmp -n 32 -fm
        batbot preprocess ../data -o ./tmp -f --dry-run --output-json dry_run.json
        batbot preprocess ../data -o ./tmp --cleanup
    """
    in_filepaths = []
    for file in filepaths:
        if isdir(file):
            in_filepaths.extend(glob(join(file,'**/*.wav'), recursive=True))
        elif isfile(file):
            in_filepaths.append(file)
        else:
            in_filepaths.extend(glob(file, recursive=True))
    # remove any repeats
    in_filepaths = sorted(list(set(in_filepaths)))

    if len(in_filepaths) == 0:
        print('Found no files given filepaths input {}'.format(filepaths))
        return

    # set up output paths for each input path
    root_inpath = commonpath(in_filepaths)
    root_outpath = '.' if output_dir is None else output_dir
    makedirs(root_outpath, exist_ok=True)
    if no_file_structure:
        out_filepath_stems = [join(root_outpath, splitext(x)[0]) for x in in_filepaths]
    else:
        out_filepath_stems = [splitext(join(root_outpath, relpath(x, root_inpath)))[0] for x in in_filepaths]
        new_dirs = [split(x)[0] for x in out_filepath_stems]
        for new_dir in set(new_dirs):
            makedirs(new_dir, exist_ok=True)

    # look for existing output files and remove from the set
    in_filepaths = np.array(in_filepaths)
    if dry_run or cleanup:
        # save copy of all outputs before removing already processed data
        out_filepath_stems_all = out_filepath_stems.copy()
    out_filepath_stems = np.array(out_filepath_stems)
    if not force_overwrite:
        idx_remove = np.full((len(in_filepaths),), False)
        for ii, out_file_stem in enumerate(out_filepath_stems):
            test_file = '{}.*'.format(out_file_stem)
            test_glob = glob(test_file)
            if len(test_glob) > 0:
                idx_remove[ii] = True
        in_filepaths = in_filepaths[np.invert(idx_remove)]
        out_filepath_stems = out_filepath_stems[np.invert(idx_remove)]
        n_skipped = sum(idx_remove)
        if len(in_filepaths) == 0:
            print('Found no unprocessed files given filepaths input {} and output directory "{}" after skipping {} files'.format(filepaths, root_outpath, n_skipped))
            print('If desired, use --force-overwrite flag to overwrite existing processed data')
            return

    if dry_run or cleanup:
        # Find all "extra" files that would be deleted in cleanup mode
        all_files = set(glob(join(root_outpath, '**/*'), recursive=True))
        for out_stem in out_filepath_stems_all:
            out_files = glob('{}.*'.format(out_stem))
            all_files -= set(out_files)
        dir_files = []
        # remove directories
        for file in all_files:
            if isdir(file):
                dir_files.append(file)
        all_files -= set(dir_files)
        extra_files = all_files

    print('Located {} total unprocessed files'.format(len(in_filepaths)))
    print('\tFast processing mode {}'.format('OFF' if process_metadata else 'ON'))
    if process_metadata:
        print('\t\tFull bat call metadata will be produced')
    print('\tForce output overwrite {}'.format('ON' if force_overwrite else 'OFF'))
    if not force_overwrite:
        print('\t\tSkipped {} files with already preprocessed outputs'.format(n_skipped))
    print('\tNum parallel workers: {}'.format(num_workers))
    if no_file_structure:
        print('\tFlattening output file structure')
    print('\tCurrent working dir: {}'.format(getcwd()))
    print('\tOutput root dir: {}'.format(output_dir))
    print('\tFirst input file -> output files: {} -> {}.*'.format(in_filepaths[0], out_filepath_stems[0]))
    if len(in_filepaths) > 2:
        print('\tLast input file -> output files: {} -> {}.*'.format(in_filepaths[-1], out_filepath_stems[-1]))

    if dry_run:
        # Print out files to be processed, anticipated outputs, and files that would be deleted in cleanup mode.
        print('\nDry run mode active - skipping all processing')
        data = {}
        data['input file, output file stem'] = [(str(x),'{}.*'.format(y)) for x, y in zip(in_filepaths, out_filepath_stems)]
        data['files to be deleted in cleanup'] = list(extra_files)
        if output_json is None:
            import pprint
            pprint.pp(data)
        else:
            with open(output_json, 'w') as outfile:
                json.dump(data, outfile, indent=4)
            print('Outputs written to {}'.format(output_json))
        print('Complete.')
        return
    
    if cleanup:
        print('\nCleanup mode active - skipping all processing')
        if len(extra_files) == 0:
            print('No files to delete')
        else:
            usr_in = input('Found {} files to delete (recommend to see details by running with --dry-run flag). Continue (y/n)? '.format(len(extra_files)))
            if usr_in.lower() not in ['y', 'yes']:
                print('Aborting cleanup mode.')
                return
        for file in extra_files:
            print('Deleting file: {}'.format(file))
            remove(file)
        print('Complete.')
        return
    
    # Begin execution loop.
    data = {'output_path':[], 'compressed_path':[], 'metadata_path':[], 'failed_files':[]}
    if num_workers is None or num_workers == 0:

        # Serial execution.
        for file, out_stem in tqdm(zip(in_filepaths, out_filepath_stems), desc='Preprocessing files', total=len(in_filepaths)): 
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
                if process_metadata:
                    data['metadata_path'].append(metadata_path)
            except:
                warnings.warn('WARNING: Pipeline failed for file {}'.format(file))
                data['failed_files'].append(str(file))
            #     raise
    else:
        # Parallel execution.
        # shuffle input and output paths
        zipped = np.stack((in_filepaths, out_filepath_stems), axis=-1)
        np.random.seed(0)
        np.random.shuffle(zipped)
        assert all([x in zipped[:,0] and y in zipped[:,1] for x, y in zip(in_filepaths, out_filepath_stems)])
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
            desc='Preprocessing chunks of files with {} workers'.format(num_workers),
        )
        data['output_path'].extend(output_paths)
        data['compressed_path'].extend(compressed_paths)
        if process_metadata:
            data['metadata_path'].extend(metadata_paths)
        data['failed_files'].extend(failed_files)

    if output_json is None:
        import pprint
        print('\nFull spectrogram output paths:')
        pprint.pp(sorted(data['output_path']))
        print('\nCompressed spectrogram output paths:')
        pprint.pp(sorted(data['compressed_path']))
        if process_metadata:
            print('\nProcessed metadata paths:')
            pprint.pp(sorted(data['metadata_path']))
        print('\nFiles that failed processing and were skipped:')
        pprint.pp(sorted(data['failed_files']))
    else:
        with open(output_json, 'w') as outfile:
            json.dump(data, outfile, indent=4)
        print('Outputs written to {}'.format(output_json))
    print('\nComplete.')

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
            json.dump(data, outfile, indent=4)
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
