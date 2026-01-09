from os.path import abspath, join


def test_spectrogram_compute():
    from batbot.spectrogram import compute

    wav_filepath = abspath(join('examples', 'example2.wav'))
    output_paths, metadata_path, metadata = compute(wav_filepath)
