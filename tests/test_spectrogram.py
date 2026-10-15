from pathlib import Path


def test_spectrogram_compute(tmp_path):
    from batbot.spectrogram import compute

    wav_filepath = Path(__file__).resolve().parents[1] / 'examples' / 'example2.wav'
    output_folder = tmp_path / 'spectrograms 空間'
    output_paths, compressed_paths, metadata_path, metadata = compute(
        wav_filepath, output_folder=output_folder
    )
