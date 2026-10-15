"""Numerical/algorithmic expectations using deterministic, synthetic inputs."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.io import wavfile

from batbot import spectrogram as spec


def test_islands_and_largest_slope_run():
    islands = spec.get_islands(np.array([1, 1, 2, 3, 3, 3, 4]))
    assert [island.tolist() for island in islands] == [[1, 1], [3, 3, 3]]
    assert spec.get_slope_islands(np.array([False, False, True, True, True, False, False])) == (
        2,
        4,
    )
    assert spec.get_slope_islands(np.zeros(4, dtype=bool)) == (0, 3)


@pytest.mark.parametrize(
    'ranges,expected', [([], []), ([(8, 10), (1, 4), (3, 6), (6, 7), (2, 3)], [(1, 7), (8, 10)])]
)
def test_merge_ranges_sorts_and_unions_overlapping_or_touching_intervals(ranges, expected):
    original = list(ranges)
    assert spec.merge_ranges(ranges, 10) == expected
    assert ranges == original


@pytest.mark.parametrize(
    'value,dtype,expected',
    [
        (1, None, [0, 0.5, 1]),
        (None, np.uint8, [0, 128, 255]),
        (None, np.uint16, [0, 32768, 65535]),
    ],
)
def test_normalization_preserves_relative_intensity_and_dtype(value, dtype, expected):
    data = np.array([[5, 7, 9]])
    result = spec.normalize_stft(data, value, dtype)
    np.testing.assert_array_equal(result, [expected])
    np.testing.assert_array_equal(data, [[5, 7, 9]])
    assert result.dtype == (np.float32 if dtype is None else dtype)


def test_gain_removes_frequency_bias_and_clips_below_dynamic_range():
    data = np.array([[10.0, 20.0, 30.0], [110.0, 120.0, 130.0]])
    result = spec.gain_stft(data.copy(), gain_db=15, autogain_stddev=100, max_band_idx=None)
    np.testing.assert_array_equal(result, [[0, 5, 15], [0, 5, 15]])


def test_window_geometry_and_candidates_use_time_and_frequency_limits():
    image = np.tile(np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1]), (4, 1)).astype(np.uint16)
    image[0] = 0  # Nonphysical frequency band, blanked by load_stft.
    image[2, 4] = 100
    candidates, amplitudes = spec.create_coarse_candidates(
        image, window=4, stride=2, min_band_idx=1
    )
    assert candidates == [(1, 2, 6), (2, 4, 8)]
    np.testing.assert_allclose(amplitudes, [np.nan, 100, 100, np.nan], equal_nan=True)
    assert spec.calculate_window_and_stride(np.zeros((4, 1000)), 1) == (12, 4)
    assert spec.calculate_window_and_stride(
        np.zeros((4, 1000)), 2, time_vec=np.arange(1000) * 0.002
    ) == (6, 2)


def test_skew_normalization_handles_missing_or_uninformative_values():
    np.testing.assert_allclose(spec.normalize_skew(np.array([1.0, 3.0, np.nan]), 1), [0, 1, 0])
    np.testing.assert_array_equal(spec.normalize_skew(np.zeros(3), 0), np.zeros(3))
    assert (
        spec.calculate_mean_within_stddev_window(np.array([1.0, 1.0, 1.0, 100.0, np.nan]), 1) == 1
    )


def test_histogram_statistics_and_saved_plot_match_selected_pixels(tmp_path):
    data = np.array([[0, 0, 255, 255], [0, 1, 1, 2]], dtype=np.uint8)
    result = spec.plot_histogram(
        data, ignore_zeros=True, min_band_idx=1, smoothing=0, output_path=None
    )
    assert result == (1, 0, 1)
    plotted = spec.plot_histogram(
        data, ignore_zeros=True, min_band_idx=1, smoothing=1, output_path=tmp_path
    )
    assert plotted[:2] == (1, 0)
    image = cv2.imread(str(tmp_path / 'histogram.png'))
    assert image is not None and image.std() > 0


@pytest.mark.parametrize('sample_rate', [125_000, 500_000])
def test_stft_preserves_tone_frequency_and_blanks_above_original_nyquist(tmp_path, sample_rate):
    time = np.arange(sample_rate // 20) / sample_rate
    tone = np.sin(2 * np.pi * 40_000 * time).astype(np.float32)
    path = tmp_path / 'tone.WAV'
    wavfile.write(path, sample_rate, tone)
    stft, waveplot, rate, bands, duration, _, times, original_rate, max_band = spec.load_stft(
        path, fast_mode=True
    )
    assert rate == 250_000 and original_rate == sample_rate
    assert duration == pytest.approx(0.05)
    assert waveplot == [] and stft.shape == (len(bands), len(times))
    # Half of a 250 kHz / 512 frequency bin is approximately 244 Hz.
    assert bands[np.argmax(stft.mean(axis=1))] == pytest.approx(40_000, abs=250)
    assert times[1] - times[0] == pytest.approx(16 / 250_000)
    if sample_rate < rate:
        assert bands[max_band] < sample_rate / 2
        assert np.all(stft[max_band + 1 :] == stft[: max_band + 1].min())


def test_stft_reports_invalid_recording(tmp_path):
    path = tmp_path / 'bad.wav'
    path.write_bytes(b'not a recording')
    with pytest.raises(OSError, match='Error loading file'):
        spec.load_stft(path)


def test_connected_components_follow_peak_locations_and_refinement_falls_back():
    segment = np.zeros((12, 12), dtype=np.uint8)
    segment[2:5, 2:5] = 100
    segment[8:10, 8:10] = 200
    locations = [(2, 2), (3, 3), (8, 8)]
    valid, mask, peak = spec.find_contour_connected_components(
        segment, 0, locations, output_path=None
    )
    expected = np.zeros_like(segment, dtype=bool)
    expected[2:5, 2:5] = True
    assert valid and peak == (2, 2)  # round-to-even of the two selected peak locations.
    np.testing.assert_array_equal(mask, expected)
    refined_mask, refined_peak = spec.refine_contour(
        segment, 0, [(8, 8)], mask, peak, output_path=None
    )
    assert refined_peak == (8, 8) and refined_mask.sum() == 4
    unchanged, unchanged_peak = spec.refine_contour(
        segment, 0, [(0, 0)], mask, peak, output_path=None
    )
    np.testing.assert_array_equal(unchanged, mask)
    assert unchanged_peak == peak


def test_thresholding_and_median_filter_cutoffs_have_expected_pixel_values():
    original = np.arange(25, dtype=np.uint16).reshape(5, 5)
    thresholded, median, deviation, _ = spec.threshold_contour(original.copy(), 0, output_path=None)
    assert median == 12.5 and deviation == pytest.approx(6 * 1.482602218505602)
    expected = original.copy()
    expected[expected < 22] = 0
    np.testing.assert_array_equal(thresholded, expected)
    filtered = spec.filter_contour(original, 0, med_db=20, std_db=5, kernel=1, output_path=None)
    expected = np.where(original >= 15, np.rint(original / 24 * 65535), 0).astype(np.uint16)
    np.testing.assert_array_equal(filtered, expected)
    normalized = spec.normalize_contour(original, 0, dtype=np.uint8, blur=False, output_path=None)
    np.testing.assert_array_equal(normalized, np.rint(original / 24 * 255).astype(np.uint8))


def test_harmonic_and_echo_masks_obey_flipped_frequency_and_forward_time():
    primary = np.zeros((10, 20), dtype=bool)
    primary[7, 4] = True
    harmonic = spec.find_harmonic(primary, 0, freq_offset=0, kernel=3, output_path=None)
    expected = np.zeros_like(primary)
    expected[4:6, 4] = True
    np.testing.assert_array_equal(harmonic, expected)
    echo = spec.find_echo(primary, 0, kernel=3, output_path=None)
    expected[:] = False
    expected[7, 5:] = True
    np.testing.assert_array_equal(echo, expected)


@pytest.mark.parametrize('statistics', [{}, {'med_db': 20, 'std_db': 5}])
def test_secondary_effect_removal_keeps_primary_and_suppresses_masked_pixels(statistics):
    segment = np.ones((10, 10), dtype=np.uint8)
    segment[2, 2], segment[6, 6], segment[8, 8] = 200, 150, 100
    harmonic, echo = np.zeros_like(segment, dtype=bool), np.zeros_like(segment, dtype=bool)
    harmonic[6, 6], echo[8, 8] = True, True
    result = spec.remove_harmonic_and_echo(
        segment, 0, harmonic, echo, threshold=10, kernel=1, output_path=None, **statistics
    )
    expected = np.zeros_like(segment)
    expected[2, 2] = 255
    np.testing.assert_array_equal(result, expected)
    assert segment[6, 6] == 150  # Source pixels are not modified.


@pytest.mark.parametrize(
    'end,expected',
    [((3, 4), (6000, 2, True)), ((2, 4), (4000, 2, False)), ((3, 3), (6000, 1.5, False))],
)
def test_contour_significance_requires_both_bandwidth_and_duration(end, expected):
    assert spec.significant_contour_path((0, 0), end, freq_step=2000, time_step=0.5) == expected


def test_debug_folder_purge_is_scoped_and_can_be_disabled(tmp_path):
    unrelated = tmp_path / 'keep.txt'
    unrelated.write_text('keep')
    assert spec.get_debug_path(tmp_path, 'call.wav', False) is None
    directory = Path(spec.get_debug_path(tmp_path, 'call.wav', True))
    (directory / 'old.png').touch()
    assert (
        Path(spec.get_debug_path(tmp_path, 'call.wav', True, purge=False, ensure=False))
        == directory
    )
    assert (directory / 'old.png').exists()
    assert Path(spec.get_debug_path(tmp_path, 'call.wav', True)) == directory
    assert list(directory.iterdir()) == [] and unrelated.read_text() == 'keep'
