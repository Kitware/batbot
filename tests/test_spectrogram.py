from os.path import abspath, join


def test_spectrogram_compute():
    from batbot.spectrogram import compute

    wav_filepath = abspath(join('examples', 'example2.wav'))
    output_folder = './output'
    output_paths, metadata_path, metadata = compute(wav_filepath, output_folder=output_folder)
