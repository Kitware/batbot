from os.path import abspath, basename, join, splitext


def test_spectrogram_compute():
    from batbot.spectrogram import compute

    wav_filepath = abspath(join('examples', 'example2.wav'))
    output_folder = './output'
    output_stem = join(output_folder, splitext(basename(wav_filepath))[0])
    output_paths, compressed_paths, metadata_path, metadata = compute(wav_filepath, out_file_stem=output_stem)
