from os.path import abspath, join

import numpy as np
import pytest


def test_spectrogram_compute():
    from batbot.spectrogram import compute

    wav_filepath = abspath(join('examples', 'example1.wav'))
    output_paths, metadata_path, metadata = compute(wav_filepath)


class TestNoiseReduction:
    """Tests for noise reduction algorithms."""

    @pytest.fixture
    def synthetic_spectrogram(self):
        """Create a synthetic spectrogram with noise and signal."""
        np.random.seed(42)
        freq_bins, time_frames = 64, 200

        # Create noise floor (first 20 frames are pure noise)
        noise_level = -60.0
        noise = np.random.randn(freq_bins, time_frames) * 5 + noise_level

        # Add a synthetic bat chirp signal (frequency sweep)
        signal = np.zeros((freq_bins, time_frames))
        for t in range(50, 150):
            freq_idx = int(freq_bins * 0.8 - (t - 50) * 0.4)
            if 0 <= freq_idx < freq_bins:
                signal[freq_idx, t] = 20.0  # Signal 20 dB above noise

        return noise + signal

    def test_spectral_subtraction_preserves_shape(self, synthetic_spectrogram):
        """Test that spectral subtraction preserves input shape."""
        from batbot.spectrogram import spectral_subtraction

        result = spectral_subtraction(synthetic_spectrogram)
        assert result.shape == synthetic_spectrogram.shape

    def test_spectral_subtraction_reduces_noise(self, synthetic_spectrogram):
        """Test that spectral subtraction reduces noise in noise-only regions."""
        from batbot.spectrogram import spectral_subtraction

        result = spectral_subtraction(synthetic_spectrogram, noise_frames=20)

        # Noise region (first 20 frames) should be reduced
        noise_region_before = synthetic_spectrogram[:, :20].mean()
        noise_region_after = result[:, :20].mean()
        assert noise_region_after < noise_region_before

    def test_spectral_subtraction_with_oversubtraction(self, synthetic_spectrogram):
        """Test spectral subtraction with different oversubtraction factors."""
        from batbot.spectrogram import spectral_subtraction

        result_1x = spectral_subtraction(synthetic_spectrogram, oversubtraction=1.0)
        result_2x = spectral_subtraction(synthetic_spectrogram, oversubtraction=2.0)

        # Higher oversubtraction should result in lower mean values
        assert result_2x.mean() < result_1x.mean()

    def test_wiener_filter_preserves_shape(self, synthetic_spectrogram):
        """Test that Wiener filter preserves input shape."""
        from batbot.spectrogram import wiener_filter

        result = wiener_filter(synthetic_spectrogram)
        assert result.shape == synthetic_spectrogram.shape

    def test_wiener_filter_reduces_noise(self, synthetic_spectrogram):
        """Test that Wiener filter reduces noise while preserving signal."""
        from batbot.spectrogram import wiener_filter

        result = wiener_filter(synthetic_spectrogram, noise_frames=20)

        # Signal region should still have higher values than noise region
        signal_region = result[:, 50:150].max()
        noise_region = result[:, :20].mean()
        assert signal_region > noise_region

    def test_wiener_filter_gain_floor(self, synthetic_spectrogram):
        """Test that Wiener filter respects gain floor."""
        from batbot.spectrogram import wiener_filter

        result_high_floor = wiener_filter(synthetic_spectrogram, gain_floor=0.5)
        result_low_floor = wiener_filter(synthetic_spectrogram, gain_floor=0.01)

        # Higher gain floor should preserve more energy
        assert result_high_floor.mean() > result_low_floor.mean()

    def test_adaptive_noise_floor_preserves_shape(self, synthetic_spectrogram):
        """Test that adaptive noise floor preserves input shape."""
        from batbot.spectrogram import adaptive_noise_floor

        result = adaptive_noise_floor(synthetic_spectrogram)
        assert result.shape == synthetic_spectrogram.shape

    def test_adaptive_noise_floor_non_negative(self, synthetic_spectrogram):
        """Test that adaptive noise floor output is non-negative."""
        from batbot.spectrogram import adaptive_noise_floor

        result = adaptive_noise_floor(synthetic_spectrogram)
        assert result.min() >= 0.0

    def test_adaptive_noise_floor_percentile_effect(self, synthetic_spectrogram):
        """Test that different percentiles affect the output."""
        from batbot.spectrogram import adaptive_noise_floor

        result_low = adaptive_noise_floor(synthetic_spectrogram, percentile=5)
        result_high = adaptive_noise_floor(synthetic_spectrogram, percentile=25)

        # Lower percentile = lower noise floor = more signal preserved
        assert result_low.mean() > result_high.mean()

    def test_noise_reduction_none_unchanged(self, synthetic_spectrogram):
        """Test that None noise_reduction leaves data unchanged."""
        from batbot.spectrogram import (
            spectral_subtraction,
            wiener_filter,
            adaptive_noise_floor,
        )

        # Each function should return a copy, not modify in place
        original = synthetic_spectrogram.copy()

        spectral_subtraction(synthetic_spectrogram)
        assert np.allclose(synthetic_spectrogram, original)

        wiener_filter(synthetic_spectrogram)
        assert np.allclose(synthetic_spectrogram, original)

        adaptive_noise_floor(synthetic_spectrogram)
        assert np.allclose(synthetic_spectrogram, original)

    def test_empty_spectrogram_handling(self):
        """Test that noise reduction handles edge cases gracefully."""
        from batbot.spectrogram import (
            spectral_subtraction,
            wiener_filter,
            adaptive_noise_floor,
        )

        # Very small spectrogram
        small = np.random.randn(4, 5) - 40.0

        result1 = spectral_subtraction(small, noise_frames=2)
        result2 = wiener_filter(small, noise_frames=2)
        result3 = adaptive_noise_floor(small, window_frames=2)

        assert result1.shape == small.shape
        assert result2.shape == small.shape
        assert result3.shape == small.shape


class TestLoadStftNoiseReduction:
    """Tests for load_stft with noise reduction options."""

    @pytest.fixture
    def wav_filepath(self):
        return abspath(join('examples', 'example1.wav'))

    def test_load_stft_no_noise_reduction(self, wav_filepath):
        """Test load_stft with no noise reduction (default)."""
        from batbot.spectrogram import load_stft

        result = load_stft(wav_filepath, noise_reduction=None)
        assert len(result) == 6  # stft_db, waveplot, sr, bands, duration, min_index

    def test_load_stft_spectral_subtraction(self, wav_filepath):
        """Test load_stft with spectral subtraction."""
        from batbot.spectrogram import load_stft

        result = load_stft(wav_filepath, noise_reduction='spectral')
        stft_db = result[0]
        assert stft_db is not None
        assert len(stft_db.shape) == 2

    def test_load_stft_wiener_filter(self, wav_filepath):
        """Test load_stft with Wiener filter."""
        from batbot.spectrogram import load_stft

        result = load_stft(wav_filepath, noise_reduction='wiener')
        stft_db = result[0]
        assert stft_db is not None
        assert len(stft_db.shape) == 2

    def test_load_stft_adaptive_noise_floor(self, wav_filepath):
        """Test load_stft with adaptive noise floor."""
        from batbot.spectrogram import load_stft

        result = load_stft(wav_filepath, noise_reduction='adaptive')
        stft_db = result[0]
        assert stft_db is not None
        assert len(stft_db.shape) == 2

    def test_load_stft_with_custom_params(self, wav_filepath):
        """Test load_stft with custom noise reduction parameters."""
        from batbot.spectrogram import load_stft

        params = {'noise_frames': 20, 'oversubtraction': 1.5}
        result = load_stft(
            wav_filepath,
            noise_reduction='spectral',
            noise_reduction_params=params
        )
        stft_db = result[0]
        assert stft_db is not None
