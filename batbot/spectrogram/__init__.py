""" """

import json
import os
import shutil
import warnings
from os.path import basename, exists, join, splitext

import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt

# import networkx as nx
import numpy as np
import pyastar2d
import scipy.signal  # Ensure this is at the top with other imports
import tqdm
from line_profiler import LineProfiler
from scipy import ndimage

# from PIL import Image
from scipy.ndimage import gaussian_filter1d

# from scipy.ndimage.filters import maximum_filter1d
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from skimage import draw, measure

from batbot import log

lp = LineProfiler()


FREQ_MIN = 5e3
FREQ_MAX = 120e3


def compute(*args, **kwargs):
    retval = compute_wrapper(*args, **kwargs)
    if not kwargs.get('fast_mode', True):
        lp.print_stats()
    return retval


def get_islands(data):
    # Find all islands of contiguous same elements with island length 2 or more
    mask = np.r_[np.diff(data) == 0, False]
    mask_ = np.concatenate(([False], mask, [False]))
    idx_ = np.flatnonzero(mask_[1:] != mask_[:-1])
    return [data[idx_[i] : idx_[i + 1] + 1] for i in range(0, len(idx_), 2)]


def get_slope_islands(slope_flags):
    flags = slope_flags.astype(np.uint8)
    islands = get_islands(flags)
    idx = int(np.argmax([val.sum() for val in islands]))
    islands = [val * (1 if i == idx else 0) for i, val in enumerate(islands)]
    island = np.hstack(islands)
    if not np.any(island):
        min_idx = 0
        max_idx = len(island) - 1
    else:
        island_flags = np.where(island)[0]
        min_idx = int(island_flags.min())
        max_idx = int(island_flags.max())
    return min_idx, max_idx


def merge_ranges(ranges, max_val):
    merged = []
    merge = []
    # sort by range start times in case ranges are out of order
    ranges = sorted(ranges)
    for values in ranges:
        values = list(values)
        if len(merge) == 0:
            merge += values
            continue
        start, stop = values
        if start <= max(merge):
            merge += values
            continue
        merged.append(merge)
        merge = values
    if len(merge) > 0:
        merged.append(merge)

    merged = [(min(val), max(val)) for val in merged]

    for index in range(1, len(merged)):
        start1, stop1 = merged[index - 1]
        start2, stop2 = merged[index]
        assert start1 < stop1
        assert start2 < stop2
        assert start1 < start2
        assert stop1 < stop2
        assert stop1 < start2

    for start1, stop1 in ranges:
        found = False
        for start2, stop2 in merged:
            if start2 <= start1 and start1 <= stop2:
                if start2 <= stop1 and stop1 <= stop2:
                    found = True
                    break
        assert found

    return merged


def plot_histogram(
    image,
    ignore_zeros=False,
    max_val=None,
    smoothing=128,
    csum_threshold=0.95,
    output_path='.',
    output_filename='histogram.png',
):
    if max_val is None:
        max_val = int(image.max())

    if ignore_zeros:
        image = image[image > 0]

    med_ = np.median(image)
    mean_ = np.mean(image)
    std_ = scipy.stats.median_abs_deviation(image, axis=None, scale='normal')
    # skew_ = scipy.stats.skew(image, axis=None)

    # Calculate the histogram
    hist = cv2.calcHist([image], [0], None, [max_val + 1], [0, max_val + 1])
    hist = hist.reshape(-1)
    # if ignore_zeros:
    #     assert hist[0] == 0

    if output_path:
        hist_original = hist.copy()
    if smoothing:
        hist = gaussian_filter1d(hist, smoothing, mode='nearest')
        if output_path:
            hist_original = (hist_original / hist_original.max()) * hist.max()

    mode_ = np.argmax(hist)  # histogram mode

    csum = np.cumsum(hist) / hist.sum()
    csum_ = np.where(csum >= csum_threshold)[0].min()

    retval = med_, std_, mode_
    if output_path is None:
        return retval

    y_max = hist.max() * 1.01
    # Plot the histogram
    plt.figure(figsize=(7, 7))
    plt.title('Grayscale Histogram', y=1.32)
    plt.xlim([0, max_val])
    plt.ylim([hist.max() * -0.01, y_max])
    plt.xlabel('Amplitude')
    plt.ylabel('Pixels')

    plt.axhline(0, color='black')
    plt.plot(hist_original, label='Histogram Raw (Non-zero)', color='orange', alpha=0.8)
    plt.plot(hist, label='Histogram Smoothed (Non-zero)')
    plt.plot(csum * y_max, label='Cumulative Sum')
    plt.plot([mean_] * 2, [0, y_max], color='black', linestyle='--', label=f'Mean ({mean_:0.01f})')
    plt.plot([med_] * 2, [0, y_max], color='red', linestyle='--', label=f'Median ({med_:0.01f})')
    plt.plot(
        [mode_] * 2,
        [0, y_max],
        color='grey',
        linestyle='--',
        label=f'Histogram Peak ({mode_:0.01f} +/- {std_:0.01f})',
    )
    plt.plot(
        [csum_] * 2,
        [0, y_max],
        color='orange',
        linestyle='--',
        label=f'CSUM >= {csum_threshold:0.02f} ({csum_:0.01f})',
    )
    plt.axvspan(mode_ - std_, mode_ + std_, color='grey', alpha=0.1)
    # plt.plot([med_ - std_] * 2, [0, hist.max()], color='blue', linestyle='--', label=f'Median +/- MAD [{med_ - std_:0.01f} - {med_ + std_:0.01f}]')
    # plt.plot([med_ + std_] * 2, [0, hist.max()], color='blue', linestyle='--')

    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=3,
        ncol=1,
        mode='expand',
        borderaxespad=0.0,
    )

    plt.savefig(
        join(output_path, output_filename),
        dpi=150,
        bbox_inches='tight',
    )
    plt.close('all')

    return retval


def generate_waveplot(
    waveform,
    stft_db,
    hop_length=16,
    size_multiplier=4,
    bg_color=(240, 255, 255),
    fg_color=(237, 149, 100),
    zero_color=(0, 0, 0),
):
    # Calculate the minimum and maximum wavepoints within the hop length window
    temp = np.pad(waveform, hop_length // 2, mode='edge')
    views = np.lib.stride_tricks.sliding_window_view(temp, (hop_length,))[::hop_length]
    bin_mins = np.min(views, axis=1)
    bin_maxs = np.max(views, axis=1)
    bin_range = max(np.abs(bin_mins).max(), np.abs(bin_maxs).max())
    waveplot = np.zeros((stft_db.shape[0] * size_multiplier, stft_db.shape[1], 3), dtype=np.uint8)

    bin_mins += bin_range
    bin_maxs += bin_range
    bin_mins /= 2 * bin_range
    bin_maxs /= 2 * bin_range
    bin_mins *= waveplot.shape[0]
    bin_maxs *= waveplot.shape[0]
    bin_mins = np.around(bin_mins).astype(int)
    bin_maxs = np.around(bin_maxs).astype(int)

    waveplot[:, :, :] = bg_color  # ivory
    mid = waveplot.shape[0] // 2
    for bin_index, (bin_min, bin_max) in enumerate(zip(bin_mins, bin_maxs)):
        assert 0 <= bin_min and bin_min <= waveplot.shape[0]
        assert 0 <= bin_max and bin_max <= waveplot.shape[0]
        assert bin_min <= bin_max
        waveplot[bin_min:bin_max, bin_index, :] = fg_color  # cornflowerblue
    if zero_color:
        waveplot[mid, :, :] = zero_color

    return waveplot


# @lp
def load_stft(
    wav_filepath,
    sr=250e3,
    n_fft=512,
    window='blackmanharris',
    win_length=256,
    hop_length=16,
    fast_mode=False,
):
    assert exists(wav_filepath)
    log.debug(f'Computing spectrogram on {wav_filepath}')

    # Load WAV file
    try:
        waveform_, sr_ = librosa.load(wav_filepath, sr=None)
        duration = len(waveform_) / sr_
    except Exception as e:
        raise OSError(f'Error loading file: {e}')

    # Resample the waveform
    waveform = librosa.resample(waveform_, orig_sr=sr_, target_sr=sr)

    # Convert the waveform to a (complex) spectrogram
    stft = librosa.stft(
        waveform, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length
    )
    # Convert the complex power (amplitude + phase) into amplitude (decibels)
    stft_db = librosa.power_to_db(np.abs(stft) ** 2, ref=np.max)
    # Retrieve time vector in seconds corresponding to STFT
    time_vec = librosa.frames_to_time(
        range(stft_db.shape[1]), sr=sr, hop_length=hop_length, n_fft=n_fft
    )

    # Remove frequencies that we do not need [FREQ_MIN - FREQ_MAX]
    bands = librosa.fft_frequencies(sr=sr, n_fft=n_fft)  # band center frequencies
    delta_f = bands[1] - bands[0]  # bandwidth

    goods = []
    for index in range(len(bands)):
        band_min = bands[index] - delta_f / 2.0
        band_max = bands[index] + delta_f / 2.0
        # accept bands with any part of their range within interval [FREQ_MIN, FREQ_MAX]
        if FREQ_MIN <= band_max and band_min <= FREQ_MAX:
            goods.append(index)
    min_index = min(goods)
    max_index = max(goods)

    # Return only valid frequency bands
    stft_db = stft_db[min_index : max_index + 1, :]
    bands = bands[min_index : max_index + 1]

    if fast_mode:
        waveplot = []
    else:
        waveplot = generate_waveplot(waveform, stft_db, hop_length=hop_length)

    return stft_db, waveplot, sr, bands, duration, min_index, time_vec


# @lp
def gain_stft(stft_db, gain_db=80.0, autogain_stddev=5.0, fast_mode=False):
    # Subtract per-frequency median DB
    med = np.median(stft_db, axis=1).reshape(-1, 1)
    stft_db -= med
    # Subtract global DB peak
    stft_db -= stft_db.max()

    # Dynamic range relative to the maximum DB
    assert stft_db.max() == 0
    stft_db += gain_db

    if not fast_mode:
        # Calculate the non-zero median DB and MAD
        #   autogain signal if (median - alpha * deviation) is higher than provided gain
        temp = stft_db[stft_db > 0]
        med_db = np.median(temp)
        std_db = scipy.stats.median_abs_deviation(temp, axis=None, scale='normal')
        autogain_value = med_db - (autogain_stddev * std_db)
        if autogain_value > 0:
            stft_db -= autogain_value

    # Clip values below zero
    stft_db = np.clip(stft_db, 0.0, None)

    return stft_db


def normalize_stft(data, value=1.0, dtype=None):
    if value is None:
        value = np.iinfo(dtype).max

    data = data.astype(np.float32)

    min_val = data.min()
    if min_val != 0:
        data -= min_val

    max_val = data.max()
    if max_val not in [0, 1]:
        data /= max_val

    assert data.min() == 0
    assert data.max() == 1

    if value != 1:
        data *= value

    if dtype:
        data = np.around(data).astype(dtype)

    return data


def calculate_window_and_stride(
    stft_db, duration, window_size_ms=12, strides_per_window=3, time_vec=None
):
    # Create a horizontal (time) sliding window of Numpy views
    #   Window: ~12ms
    #   Stride: ~4ms
    if time_vec is not None:
        # use the precise center time per STFT column if provided
        delta_t = time_vec[1] - time_vec[0]
        window = window_size_ms / delta_t / 1e3
    else:
        # estimate the window size based on audio file length and STFT length
        window = stft_db.shape[1] / (duration * 1e3) * window_size_ms

    stride = window / strides_per_window

    window = int(round(window))
    stride = int(round(stride))

    return window, stride


def create_coarse_candidates(stft_db, window, stride, threshold_stddev=3.0):
    # Re-calculate the non-zero median DB and MAD (scaled like std)
    temp = stft_db[stft_db > 0]
    med_db = np.median(temp)
    std_db = scipy.stats.median_abs_deviation(temp, axis=None, scale='normal')
    threshold = med_db + (threshold_stddev * std_db)

    # First, create the strided windows into the STFT
    views = np.lib.stride_tricks.sliding_window_view(stft_db, (stft_db.shape[0], window))[
        0, ::stride
    ]
    candidate_dbs = np.max(views, axis=(1, 2)).astype(np.float32)
    candidate_dbs[candidate_dbs < threshold] = np.nan

    # Second, calculate the start times (x-axis) for the sliding windows
    domain = np.array(range(stft_db.shape[1])).reshape(1, -1)
    positions = np.lib.stride_tricks.sliding_window_view(domain, (1, window))[0, ::stride]
    positions = np.min(positions, axis=(1, 2))

    # Calculate the windows where the maximum amplitude is above the threshold
    idxs = np.where(~np.isnan(candidate_dbs))[0].tolist()
    starts = positions.take(idxs).tolist()
    stops = [start + window for start in starts]
    candidates = list(zip(idxs, starts, stops))

    return candidates, candidate_dbs


# @lp
def filter_candidates_to_ranges(
    stft_db,
    candidates,
    window=16,
    skew_stddev=2.0,
    area_percent=0.10,
    output_path=None,
    fast_mode=False,
):
    # Filter the candidates based on their distribution skewness
    stride_ = 2 if not fast_mode else 4
    buffer = int(round(window / stride_ / 2))

    reject_idxs = []
    ranges = []
    for index, (idx, start, stop) in tqdm.tqdm(list(enumerate(candidates)), disable=fast_mode):
        # Extract the candidate window of the STFT
        candidate = stft_db[:, start:stop]

        # Create a vertical (frequency) sliding window of Numpy views
        views = np.lib.stride_tricks.sliding_window_view(candidate, (window, candidate.shape[1]))[
            ::stride_, 0
        ]
        skews = scipy.stats.skew(views, axis=(1, 2))

        # Center and clip the skew values
        skew_thresh = calculate_mean_within_stddev_window(skews, skew_stddev)
        # IMPORTANT: Only center positive (right-sided) global skew for the global candidate calculation
        skew_thresh = max(0, skew_thresh)
        skews = normalize_skew(skews, skew_thresh)

        # Calculate the largest contiguous island of non-zeros
        skews = (skews > 0).astype(np.uint8)
        islands = get_islands(skews)
        area = float(max([val.sum() for val in islands]))
        area /= len(skews)
        if area == 0.0 and sum(skews) >= 1:
            # handle edge case with single-element islands
            area = 1.0 / len(skews)

        if area >= area_percent:
            ranges.append((start, stop))

            if output_path:
                # Plot the skew and spectrogram
                plt.figure()
                fig, axes = plt.subplots(1, 2, figsize=(7, 7))
                ax1, ax2 = axes
                plt.suptitle(f'Area: {area * 100.0:0.02f} | Skew Offset: {skew_thresh:0.04f}')
                ax1.set_xlabel('Skew Above Threshold')
                ax1.set_ylabel('Frequency')
                ax1.set_xticks([0, 1])
                ax1.set_xticklabels(['NO', 'YES'])
                ax1.set_yticks([])
                ax1.set_yticklabels([])
                ax1.set_xlim([-0.1, 1.1])
                ax1.set_ylim([-buffer, len(skews) + buffer])

                ax2.set_xticks([])
                ax2.set_xticklabels([])
                ax2.set_yticks([])
                ax2.set_yticklabels([])

                ax1.plot(skews, list(reversed(range(len(skews)))), label='Activated')
                ax1.legend()
                ax2.imshow(candidate, aspect='auto')

                plt.tight_layout()
                plt.savefig(
                    join(output_path, f'candidate.coarse.{index}.png'),
                    dpi=150,
                    bbox_inches='tight',
                )
                plt.close('all')
        else:
            reject_idxs.append(idx)

    return ranges, reject_idxs


def plot_chirp_candidates(
    stft_db, candidate_dbs, ranges, reject_idxs, output_path='.', output_filename='candidates.png'
):
    if output_path is None:
        return

    for index, (start, stop) in enumerate(ranges):
        cv2.imwrite(join(output_path, f'chirp.{index}.png'), stft_db[:, start:stop])

    candidate_dbs_ = candidate_dbs.copy()
    candidate_dbs[reject_idxs] = np.nan

    flags = np.isnan(candidate_dbs)
    candidate_dbs[flags] = np.nanmin(candidate_dbs)
    candidate_dbs -= candidate_dbs.min()
    candidate_dbs /= candidate_dbs.max()
    num = len(flags) - sum(flags)

    flags_ = np.isnan(candidate_dbs_)
    candidate_dbs_[flags_] = np.nanmin(candidate_dbs_)
    candidate_dbs_ -= candidate_dbs_.min()
    candidate_dbs_ /= candidate_dbs_.max()
    candidate_dbs_ = 1.0 - candidate_dbs_
    num_ = len(flags_) - sum(flags_)

    # Plot the skew and spectrogram
    plt.figure(figsize=(7, 3))
    plt.title('Window Candidates', y=1.26)
    plt.xlabel('Time')

    plt.plot(candidate_dbs_, alpha=0.5, label=f'Original Candidates ({num_})')
    plt.plot(candidate_dbs, alpha=0.5, label=f'Filtered Candidates ({num})')
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=3,
        ncol=1,
        mode='expand',
        borderaxespad=0.0,
    )

    plt.savefig(
        join(output_path, output_filename),
        dpi=150,
        bbox_inches='tight',
    )
    plt.close('all')


def normalize_skew(skews, skew_thresh):
    skews -= skew_thresh

    skews = np.clip(skews, 0.0, None)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        skews /= skews.max()

    skews = np.nan_to_num(skews, nan=0.0, posinf=0.0, neginf=-0.0)

    return skews


def calculate_mean_within_stddev_window(values, window):
    # Calculate the average skew within X standard deviations (temperature scaling)
    values_mean = np.mean(values)
    values_std = np.std(values)
    values_flags = np.abs(values - values_mean) <= (values_std * window)
    values_mean_windowed = values[values_flags].mean()
    return values_mean_windowed


def tighten_ranges(
    stft_db, ranges, window, duration, skew_stddev=2.0, min_duration_ms=2.0, output_path='.'
):
    minimum_duration = int(np.around(stft_db.shape[1] / (duration * 1e3) * min_duration_ms))

    stride_ = 2
    window = int(window)
    buffer = int(round(window / stride_ / 2))

    ranges_ = []
    for index, (start, stop) in tqdm.tqdm(list(enumerate(ranges))):
        # Extract the candidate window of the STFT
        candidate = stft_db[:, start:stop]

        # Create a vertical (frequency) sliding window of Numpy views
        views = np.lib.stride_tricks.sliding_window_view(candidate, (candidate.shape[0], window))[
            0, ::stride_
        ]
        skews = scipy.stats.skew(views, axis=(1, 2))

        # Center and clip the skew values
        skew_thresh = calculate_mean_within_stddev_window(skews, skew_stddev)
        skews = normalize_skew(skews, skew_thresh)

        # Calculate the largest contiguous island of non-zeros
        skew_flags = skews > 0
        skews = skew_flags.astype(np.uint8)
        islands = get_islands(skews)
        islands = [(index + 1) * val for index, val in enumerate(islands)]
        island = np.hstack(islands)

        islands_plotting = []
        for unique in set(island):
            if unique == 0:
                continue

            island_flags = np.where(island == unique)[0]
            if len(island_flags) == 0:
                continue

            island_start = island_flags.min()
            island_stop = island_flags.max()
            island_start = int(round((island_start * 2) + (window / 2) - buffer))
            island_stop = int(round((island_stop * 2) + (window / 2) + buffer))

            island_start = max(0, min(candidate.shape[1], island_start))
            island_stop = max(0, min(candidate.shape[1], island_stop))

            island_duration = island_stop - island_start
            if island_duration < minimum_duration:
                continue

            islands_plotting.append((island_start, island_stop))

            island_start += start
            island_stop += start
            ranges_.append((island_start, island_stop))

        if output_path:
            # Plot the skew and spectrogram
            plt.figure()
            fig, axes = plt.subplots(2, 1, figsize=(7, 7))
            ax1, ax2 = axes
            ax2.set_ylabel('Skew Above Threshold')
            ax2.set_xlabel('Time')
            ax2.set_xticks([])
            ax2.set_xticklabels([])
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(['NO', 'YES'])
            ax2.set_ylim([-0.1, 1.1])
            ax2.set_xlim([-buffer, len(skews) + buffer])

            ax1.set_xticks([])
            ax1.set_xticklabels([])
            ax1.set_yticks([])
            ax1.set_yticklabels([])

            ax2.plot(list(range(len(skews))), skews, label='Activated')

            candidate_ = candidate.copy()
            dtype_ = candidate_.dtype
            value_ = np.iinfo(dtype_).max
            candidate_ = candidate_.astype(np.float32)
            for island_idx, (island_start, island_stop) in enumerate(islands_plotting):
                candidate_[:, island_start:island_stop] += value_ * 0.15
                candidate_[:, island_start] = value_
                candidate_[:, island_stop] = value_

                # kwargs = {'label': 'New Start'} if island_idx == 0 else {}
                # ax2.axvline(island_start, color='green', **kwargs)
                # kwargs = {'label': 'New Stop'} if island_idx == 0 else {}
                # ax2.axvline(island_stop, color='red', **kwargs)
            candidate_[candidate_ > value_] = value_
            candidate_ = np.around(candidate_).astype(dtype_)

            # candidate_ = normalize_stft(candidate_, None, dtype_)
            ax1.imshow(candidate_, aspect='auto')

            ax2.legend()
            plt.tight_layout()
            plt.savefig(
                join(output_path, f'candidates.fine.{index}.png'),
                dpi=150,
                bbox_inches='tight',
            )
            plt.close('all')

    return ranges_


def get_debug_path(output_folder, wav_filepath, enabled, purge=True, ensure=True):
    if not enabled:
        return None

    wav_filename = basename(wav_filepath)
    debug_path = join(output_folder, f'batbot.debug.{wav_filename}')

    if purge and exists(debug_path):
        shutil.rmtree(debug_path)

    if ensure:
        os.makedirs(debug_path)
        assert exists(debug_path)

    return debug_path


def write_contour_debug_image(segment, index, counter, tag, output_path='.'):
    if output_path is None:
        return
    cv2.imwrite(join(output_path, f'contour.{index}.{counter:02d}.{tag}.png'), segment)


def find_max_locations(data):
    rows, cols = np.where(data == data.max())
    max_locations = list(zip(rows.tolist(), cols.tolist()))
    return max_locations


def create_contour_debug_canvas(segment, index, output_path='.'):
    # Write debug image (will no-op if output_path is None)
    write_contour_debug_image(segment, index, 0, 'original', output_path)

    if output_path:
        canvas = np.stack((segment, segment, segment), axis=2)
        return canvas


def threshold_contour(segment, index, output_path='.'):
    med_db, std_db, peak_db = plot_histogram(
        segment,
        ignore_zeros=True,
        output_path=output_path,
        output_filename=f'contour.{index}.histogram.png',
    )

    segment_threshold = med_db + std_db
    segment[segment < segment_threshold] = 0

    write_contour_debug_image(segment, index, 2, 'thresholded', output_path=output_path)

    return segment, med_db, std_db, peak_db


def filter_contour(segment, index, med_db=None, std_db=None, kernel=5, output_path='.'):
    # segment = cv2.erode(segment, np.ones((3, 3), np.uint8), iterations=1)

    segment = scipy.signal.medfilt(segment, kernel_size=kernel)

    if None not in {med_db, std_db}:
        segment_threshold = med_db - std_db
        segment[segment < segment_threshold] = 0

    segment = normalize_stft(segment, None, segment.dtype)

    write_contour_debug_image(segment, index, 2, 'filtered', output_path=output_path)

    return segment


def normalize_contour(segment, index, dtype=None, blur=True, kernel=5, output_path='.'):
    if dtype is None:
        dtype = segment.dtype

    segment = segment.astype(np.float32)

    if blur:
        # segment = cv2.erode(segment, np.ones((3, 3), np.uint8), iterations=1)
        segment = cv2.GaussianBlur(
            segment, (kernel, kernel), sigmaX=4, sigmaY=4, borderType=cv2.BORDER_DEFAULT
        )

    segment = normalize_stft(segment, None, dtype)

    write_contour_debug_image(segment, index, 4, 'normalized', output_path)

    return segment


def find_contour_connected_components(segment, index, locations, sequence=4, output_path='.'):
    data = cv2.connectedComponentsWithStats(normalize_stft(segment, None, np.uint8), connectivity=8)
    labels = data[1]

    counter = {}
    for location in locations:
        label = int(labels[location])
        if label not in counter:
            counter[label] = []
        counter[label].append(location)

    common, points = sorted(list(counter.items()), key=lambda x: len(x[1]), reverse=True)[0]

    valid = common > 0
    if valid:
        segmentmask = labels == common
    else:
        segmentmask = np.ones(labels.shape, dtype=bool)

    peaky, peakx = np.around(np.array(points).mean(axis=0)).astype(int).tolist()

    segmentmask_img = segmentmask.astype(np.uint8) * 255
    write_contour_debug_image(segmentmask_img, index, sequence, 'masked', output_path)

    return valid, segmentmask, (peaky, peakx)


def find_harmonic(segmentmask, index, freq_offset, kernel=15, output_path='.'):
    h = segmentmask.shape[0]
    locations = np.array(np.where(segmentmask))
    # convert mask to first harmonic (doubled frequency), accounting for flipped frequency axis
    locations[0] = h - ((h - locations[0] + freq_offset) * 2)

    flags = np.logical_and(0 <= locations[0], locations[0] < h)
    locations = locations[:, flags]

    harmonic = np.zeros(segmentmask.shape, dtype=bool)
    harmonic[tuple(locations)] = True
    locations[0] += 1
    harmonic[tuple(locations)] = True

    harmonic_ = harmonic.astype(np.uint8) * 255
    harmonic_ = cv2.GaussianBlur(
        harmonic_, (kernel, kernel), sigmaX=4, sigmaY=4, borderType=cv2.BORDER_DEFAULT
    )
    write_contour_debug_image(harmonic_, index, 7, 'harmonic', output_path)

    return harmonic


def find_echo(segmentmask, index, kernel=15, output_path='.'):
    echo = np.zeros(segmentmask.shape, dtype=int)
    echo[np.where(segmentmask)] = 1
    echo *= np.array(range(echo.shape[1]))
    maxx = np.max(echo, axis=1)
    maxx[maxx > 0] += kernel // 2
    maxx[maxx >= echo.shape[1]] = 0

    echo = np.zeros(segmentmask.shape, dtype=bool)
    for maxy, maxx in enumerate(maxx):
        if maxx == 0:
            continue
        echo[maxy, maxx:] = True

    echo_ = echo.astype(np.uint8) * 255
    echo_ = cv2.GaussianBlur(
        echo_, (kernel, kernel), sigmaX=4, sigmaY=4, borderType=cv2.BORDER_DEFAULT
    )
    write_contour_debug_image(echo_, index, 7, 'echo', output_path)

    return echo


def remove_harmonic_and_echo(
    segment, index, harmonic, echo, threshold, med_db=None, std_db=None, kernel=15, output_path='.'
):
    combined = np.logical_or(harmonic, echo)

    combined_ = combined.astype(np.uint8) * 255
    combined_ = (
        cv2.GaussianBlur(
            combined_, (kernel, kernel), sigmaX=4, sigmaY=4, borderType=cv2.BORDER_DEFAULT
        )
        / 255.0
    )
    write_contour_debug_image(combined_, index, 7, 'combined', output_path)

    dtype = segment.dtype

    segment = segment.astype(np.float32)
    segment *= 1.0 - combined_.astype(np.float32)

    if None not in {med_db, std_db}:
        segment_threshold = med_db - std_db
        segment[segment < segment_threshold] = 0

    segment = normalize_stft(segment, None, dtype)

    segment[segment < threshold] = 0
    write_contour_debug_image(segment, index, 8, 'refined', output_path=output_path)

    return segment


def refine_contour(segment, index, max_locations, segmentmask, peak, output_path='.'):
    valid, segmentmask_, peak_ = find_contour_connected_components(
        segment, index, max_locations, sequence=6, output_path=output_path
    )

    if valid:
        return segmentmask_, peak_
    else:
        return segmentmask, peak


def calculate_astar_grid_and_endpoints(
    segment, index, segmentmask, peak, canvas, kernel=7, output_path='.'
):
    costs = segment.copy()
    segmentmask_ = np.logical_not(segmentmask)
    costs[segmentmask_] = 0
    write_contour_debug_image(costs, index, 8, 'costs', output_path=output_path)

    ys, xs = np.where(costs > 0)
    points = np.stack([ys, xs], axis=1, dtype=np.float32)
    delta = points - np.array(peak, dtype=np.float32)
    # Pay a higher distance cost in the x-dimension
    delta[:, 0] *= 2
    distances = np.linalg.norm(delta, axis=1)

    peaky, peakx = peak
    flags = np.logical_and(xs <= peakx, ys <= peaky)
    begin = points[flags][np.argmax(distances[flags])]
    flags = np.logical_and(xs >= peakx, ys >= peaky)
    end = points[flags][np.argmax(distances[flags])]

    begin = tuple(begin.astype(int).tolist())
    end = tuple(end.astype(int).tolist())

    maxcost = costs.max()
    grid = costs.astype(np.float32)
    assert 0 <= grid.min() and grid.max() <= maxcost
    grid = maxcost - grid
    segmentmask_ = np.logical_not(segmentmask)
    grid[segmentmask_] = maxcost
    grid += 1
    assert grid.min() > 0

    bounds = np.where(np.sum(costs, axis=0) > 0)
    left = int(np.min(bounds))
    right = int(np.max(bounds))
    bounds = np.where(np.sum(costs, axis=1) > 0)
    top = int(np.min(bounds))
    bottom = int(np.max(bounds))
    boundary = (top, bottom, left, right)

    if output_path:
        height, width = costs.shape

        value = np.iinfo(canvas.dtype).max

        cv2.rectangle(canvas, (left, 0), (left, height), (0, value, value), 1)
        cv2.rectangle(canvas, (right, 0), (right, height), (0, value, value), 1)
        cv2.rectangle(canvas, (left, top), (right, top), (0, value, value), 1)
        cv2.rectangle(canvas, (left, bottom), (right, bottom), (0, value, value), 1)

        cv2.circle(canvas, peak[::-1], 5, (value, 0, 0), -1)
        cv2.circle(canvas, begin[::-1], 5, (0, value, 0), -1)
        cv2.circle(canvas, end[::-1], 5, (0, 0, value), -1)

        write_contour_debug_image(canvas, index, 8, 'endpoints', output_path=output_path)

    costs = segment.astype(np.float32)
    segmentmask_ = segmentmask.astype(np.float32)
    segmentmask_ = cv2.GaussianBlur(
        segmentmask_, (kernel, kernel), sigmaX=4, sigmaY=4, borderType=cv2.BORDER_DEFAULT
    )
    costs *= segmentmask_
    costs = normalize_stft(costs, None, segment.dtype)

    return costs, grid, begin, end, boundary


def extract_contour_path(grid, begin, end, canvas, index, output_path='.'):
    # set allow_diagonal=True to enable 8-connectivity
    path = pyastar2d.astar_path(grid, begin, end, allow_diagonal=False)

    if output_path:
        canvas_ = canvas.copy()
        value_ = np.iinfo(canvas_.dtype).max

        for point in path:
            cv2.circle(canvas_, point[::-1], 1, (value_, 0, 0), -1)

        write_contour_debug_image(canvas_, index, 9, 'path', output_path=output_path)

    return path


def extract_contour_keypoints(
    path, canvas, index, peak, contour_smoothing_sigma=5, output_path='.'
):
    # Smooth the fit line
    path_ = path.astype(np.float32)
    y = path_[:, 0]
    x = path_[:, 1]
    y_ = gaussian_filter1d(y, contour_smoothing_sigma, mode='nearest')
    x_ = gaussian_filter1d(x, contour_smoothing_sigma, mode='nearest')
    path_ = np.around(np.vstack((y_, x_)).T).astype(np.int32)

    # Calculate the first order derivative
    ymax = y.max()
    y_inv = ymax - y
    y_inv_ = ymax - y_
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        der1 = np.gradient(y_inv_, x_)
    der1_ = der1[np.abs(der1) != np.inf]
    der1min = np.nanmin(der1_)
    der1max = np.nanmax(der1_)
    der1 = np.nan_to_num(der1, nan=der1min, posinf=der1max, neginf=der1min)
    der1 = gaussian_filter1d(der1, contour_smoothing_sigma, mode='nearest')

    # Retrieve first (knee) and last (heel) locations where slope (dy/dx) magnitude approaches the median value
    slope_thresh = np.abs(np.median(der1))
    slope_flags = np.abs(der1) <= slope_thresh
    knee_idx, heel_idx = get_slope_islands(slope_flags)
    # Retrieve location of minimum slope magnitude between knee and heel
    fc_idx = knee_idx + int(np.argmin(np.abs(der1[knee_idx:heel_idx])))

    if output_path:
        counter = 10

        # minx_ = x_.min()
        # maxx_ = x_.max()
        # miny_ = y_inv_.min()
        # maxy_ = y_inv_.max()
        # midy_ = 0.5 * (miny_ + maxy_)

        knee_x_ = x_[knee_idx]
        heel_x_ = x_[heel_idx]
        fc_x_ = x_[fc_idx]

        # Plot the histogram
        fig, ax1 = plt.subplots(figsize=(8, 7))
        ax2 = ax1.twinx()
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Frequency')
        ax2.set_ylabel('Slope')
        plt.title('Chirp Line Fit', y=1.32)
        ax2.set_ylim([-4 * slope_thresh, 2 * slope_thresh])

        # Create a twin axes sharing the x-axis
        lines = []
        lines.append(ax1.plot(x, y_inv, label='Original [L]')[0])
        lines.append(ax1.plot(x_, y_inv_, label='Smoothed [L]')[0])
        lines.append(
            ax2.axhline(
                -slope_thresh, color='orange', linestyle='--', alpha=0.5, label='Median Slope'
            )
        )
        ax2.axhspan(-slope_thresh, slope_thresh, color='orange', alpha=0.1)
        ax2.axhline(0, color='grey', alpha=0.5)
        ax2.axhline(slope_thresh, color='orange', linestyle='--', alpha=0.5)
        lines.append(ax2.plot(x_, der1, color='black', label='Smoothed Slope [R]', alpha=0.5)[0])
        lines.append(ax1.axvline(knee_x_, color='green', label='Knee'))
        lines.append(ax1.axvline(heel_x_, color='red', label='Heel'))
        lines.append(ax1.axvline(fc_x_, color='purple', label='Characteristic Frequency'))

        labels = [line.get_label() for line in lines]
        plt.legend(
            lines,
            labels,
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc=3,
            ncol=1,
            mode='expand',
            borderaxespad=0.0,
        )
        plt.savefig(
            join(output_path, f'contour.{index}.{counter:02d}.slope.png'),
            dpi=150,
            bbox_inches='tight',
        )
        plt.close('all')

        value = np.iinfo(canvas.dtype).max

        cv2.circle(canvas, path_[fc_idx][::-1], 5, (0, 0, 0), -1)
        cv2.circle(canvas, path_[knee_idx][::-1], 5, (value, 0, value), -1)
        cv2.circle(canvas, path_[heel_idx][::-1], 5, (value, value, 0), -1)

        for point in path_:
            cv2.circle(canvas, point[::-1], 1, (value, 0, 0), -1)

        write_contour_debug_image(canvas, index, counter, 'keypoints', output_path=output_path)

    # Locate the (best) index for the peak point based on the smoothed path
    delta = path_ - np.array(peak, dtype=np.float32)
    distances = np.linalg.norm(delta, axis=1)
    peak_idx = int(np.argmin(distances))

    points = (
        tuple(path_[knee_idx].tolist()),
        tuple(path_[fc_idx].tolist()),
        tuple(path_[heel_idx].tolist()),
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        slopes = {
            'slope@hi fc:knee.y_px/x_px': float(der1[knee_idx]),
            'slope@fc.y_px/x_px': float(der1[fc_idx]),
            'slope@low fc:heel.y_px/x_px': float(der1[heel_idx]),
            'slope@peak.y_px/x_px': float(der1[peak_idx]),
            'slope[avg].y_px/x_px': float(np.mean(der1)),
            'slope/hi[avg].y_px/x_px': float(np.mean(der1[: knee_idx + 1])),
            'slope/mid[avg].y_px/x_px': float(np.mean(der1[knee_idx : heel_idx + 1])),
            'slope/lo[avg].y_px/x_px': float(np.mean(der1[heel_idx:])),
            'slope[box].y_px/x_px': float(0.5 * (der1[0] + der1[-1])),
            'slope/hi[box].y_px/x_px': float(0.5 * (der1[0] + der1[knee_idx])),
            'slope/mid[box].y_px/x_px': float(0.5 * (der1[knee_idx] + der1[heel_idx])),
            'slope/lo[box].y_px/x_px': float(0.5 * (der1[heel_idx] + der1[-1])),
        }

    return path_, points, slopes


def significant_contour_path(
    begin, end, freq_step, time_step, min_bandwidth_khz=6e3, min_duration_ms=2.0
):
    bandwidth = (end[0] - begin[0]) * freq_step
    duration = (end[1] - begin[1]) * time_step
    significant = bandwidth >= min_bandwidth_khz and duration >= min_duration_ms
    return bandwidth, duration, significant


def scale_pdf_contour(segment, index, output_path='.'):
    segment = normalize_stft(segment, None, segment.dtype)
    med_db, std_db, peak_db = plot_histogram(
        segment,
        smoothing=512,
        ignore_zeros=True,
        csum_threshold=0.9,
        output_path=output_path,
        output_filename=f'contour.{index}.00.histogram.png',
    )

    assert segment.min() == 0
    assert segment.max() == np.iinfo(segment.dtype).max
    dist = scipy.stats.norm(peak_db, std_db)
    steps = segment.max()
    x = np.linspace(0, steps, steps)
    y = dist.pdf(x)

    y[x < peak_db] = y.max()
    y -= y.min()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        y /= y.max()
    scaling = 1.0 - y

    if np.any(np.isnan(y)):
        return segment, None, None

    if output_path:
        # Plot the histogram
        plt.figure(figsize=(7, 7))
        plt.title('Inverse PDF Scaling', y=1.16)
        plt.xlim([segment.min(), segment.max()])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('Frequency')
        plt.ylabel('Probability')

        plt.axvspan(peak_db - 3 * std_db, peak_db + 3 * std_db, color='grey', alpha=0.15)
        plt.axvspan(peak_db - 2 * std_db, peak_db + 2 * std_db, color='grey', alpha=0.15)
        plt.axvspan(
            peak_db - 1 * std_db,
            peak_db + 1 * std_db,
            color='grey',
            alpha=0.15,
            label='Standard Deviations σ={1,2,3}',
        )
        plt.plot(
            [peak_db] * 2, [0, 1], color='orange', linestyle='--', label='Peak Histogram Frequency'
        )
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.axhline(1, color='black', linestyle='--', alpha=0.5)
        plt.plot(x, scaling, label='Weighting')

        plt.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc=3,
            ncol=1,
            mode='expand',
            borderaxespad=0.0,
        )

        plt.savefig(
            join(output_path, f'contour.{index}.00.histogram.scaling.png'),
            dpi=150,
            bbox_inches='tight',
        )
        plt.close('all')

    scaling = np.hstack((scaling, scaling[-1:]))
    mask = scaling[segment - segment.min()]
    temp = segment.astype(np.float32) * mask
    segment = normalize_stft(temp, None, segment.dtype)

    write_contour_debug_image(segment, index, 1, 'cdf', output_path)

    return segment, peak_db, std_db


def morph_open_contour(segment, index, output_path='.'):
    segment = cv2.morphologyEx(segment, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    write_contour_debug_image(segment, index, 3, 'open', output_path=output_path)

    return segment


def find_contour_and_peak(
    segment,
    index,
    max_locations,
    peak_db=None,
    peak_db_std=None,
    threshold_std=2.0,
    sigma=5,
    output_path='.',
    threshold=None,
):

    if not threshold:
        # Apply threshold equal to normalized (and smoothed) segment histogram mode,
        # minus the estimated noise standard deviation scaled by threshold_std
        # (note that these were computed prior to CDF weighting)
        threshold = peak_db - threshold_std * peak_db_std

    contours = measure.find_contours(
        segment, level=threshold, fully_connected='high', positive_orientation='high'
    )

    # Display the image and plot all contours found
    if output_path:
        fig, ax = plt.subplots()
        ax.imshow(segment, cmap=plt.cm.gray)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])

    max_points = [Point(*value) for value in max_locations]
    counter = {}
    segmentmask = np.zeros(segment.shape, dtype=bool)
    for idx, contour in enumerate(contours):
        polygon = Polygon(contour).convex_hull
        found = []
        for max_point, max_location in zip(max_points, max_locations):
            if polygon.contains(max_point):
                found.append(max_location)
        if len(found) > 0:
            x = gaussian_filter1d(contour[:, 1], sigma)
            y = gaussian_filter1d(contour[:, 0], sigma)

            if output_path:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
                ax.plot(x, y, linewidth=1, linestyle='--')

            contour_ = np.vstack((y, x), dtype=contour.dtype).T
            polygon_ = Polygon(contour).convex_hull
            assert idx not in counter
            counter[idx] = (found, polygon_)

            rr, cc = draw.polygon(contour_[:, 0], contour_[:, 1], shape=segment.shape)
            segmentmask[rr, cc] = True

    if output_path:
        plt.savefig(
            join(output_path, f'contour.{index}.05.contour.png'),
            dpi=150,
            pad_inches=0,
            bbox_inches='tight',
        )
        plt.close('all')

    # segmentmask = np.ones(segment.shape, dtype=bool)

    if len(counter) == 0:
        peak = None
    else:
        _, (points, polygon) = sorted(
            list(counter.items()), key=lambda x: len(x[1][0]), reverse=True
        )[0]

        peak = np.around(np.array(points).mean(axis=0)).astype(int).tolist()

    return segmentmask, peak, threshold


def refine_segmentmask(segmentmask, index, output_path='.'):
    segmentmask = ndimage.binary_fill_holes(segmentmask)

    segmentmask_img = segmentmask.astype(np.uint8) * 255
    write_contour_debug_image(segmentmask_img, index, 6, 'masked', output_path)

    return segmentmask


def calculate_harmonic_and_echo_flags(
    original, index, segmentmask, harmonic, echo, canvas, kernel=5, output_path='.'
):
    nonzeros = original > 0
    negative = ~np.logical_or(np.logical_or(harmonic, echo), segmentmask)
    negative_ = negative.astype(np.uint8) * 255
    negative_ = cv2.GaussianBlur(
        negative_, (kernel, kernel), sigmaX=4, sigmaY=4, borderType=cv2.BORDER_DEFAULT
    )
    write_contour_debug_image(negative_, index, 7, 'negative', output_path=output_path)

    negative_skew = scipy.stats.skew(original[np.logical_and(nonzeros, negative)])
    harmonic_skew = scipy.stats.skew(original[np.logical_and(nonzeros, harmonic)]) - negative_skew
    echo_skew = (
        scipy.stats.skew(original[np.logical_and(np.logical_and(nonzeros, echo), ~harmonic)])
        - negative_skew
    )

    skew_thresh = np.abs(negative_skew * 0.1)
    harmonic_flag = harmonic_skew >= skew_thresh
    echo_flag = echo_skew >= skew_thresh

    harmonic_peak = None
    if harmonic_flag:
        if output_path:
            temp = canvas.copy()
            temp[:, :, 2][harmonic] = np.iinfo(original.dtype).max
            canvas = np.around(
                (canvas.astype(np.float32) * 0.5) + (temp.astype(np.float32) * 0.5)
            ).astype(canvas.dtype)
        try:
            temp = original.copy()
            temp[~harmonic] = 0
            harmonic_peak = find_max_locations(temp)[0]
        except Exception:
            harmonic_flag = False
            harmonic_peak = None

    echo_peak = None
    if echo_flag:
        if output_path:
            temp = canvas.copy()
            temp[:, :, 0][echo] = np.iinfo(original.dtype).max
            canvas = np.around(
                (canvas.astype(np.float32) * 0.5) + (temp.astype(np.float32) * 0.5)
            ).astype(canvas.dtype)
        try:
            temp = original.copy()
            temp[~echo] = 0
            echo_peak = find_max_locations(temp)[0]
        except Exception:
            echo_flag = False
            echo_peak = None

    return harmonic_flag, harmonic_peak, echo_flag, echo_peak


# @lp
def compute_wrapper(
    wav_filepath,
    annotations=None,
    output_folder='.',
    bitdepth=16,
    fast_mode=False,
    debug=False,
    **kwargs,
):
    """
    Compute the spectrograms for a given input WAV and saves them to disk.

    If a given spectrogram has already been rendered to disk, it will not be recomputed.

    Args:
        wav_filepath (str): WAV filepath (relative or absolute) to compute spectrograms for.
        ext (str, optional): The file extension of the resulting spectrogram files.  If this value is
            not specified, it will use the same extension as `wav_filepath`.  Passed as input
            to :meth:`batbot.spectrogram.spectrogram_filepath`.  Defaults to :obj:`None`.
        **kwargs: keyword arguments passed to :meth:`batbot.spectrogram.spectrogram_grid`

    Returns:
        tuple ( int, float, tuple (int), list ( str ) ):
            - the original WAV file's sample rate.
            - the original WAV file's duration in seconds.
            - tuple of spectrogram's (width, height) in pixels
            - tuple of spectrogram's (min, max) frequency
            - list of spectrogram filepaths, split by 50k horizontal pixels
    """
    base = splitext(basename(wav_filepath))[0]

    if fast_mode:
        bitdepth = 8
    assert bitdepth in [8, 16]
    dtype = np.uint8 if bitdepth == 8 else np.uint16

    chunksize = int(50e3)

    debug_path = get_debug_path(output_folder, wav_filepath, enabled=debug)

    # Load the spectrogram from a WAV file on disk
    stft_db, waveplot, sr, bands, duration, freq_offset, time_vec = load_stft(
        wav_filepath, fast_mode=fast_mode
    )

    # Apply a dynamic range to a fixed dB range
    stft_db = gain_stft(stft_db, fast_mode=fast_mode)

    # Bin the floating point data to X-bit integers (X=8 or X=16)
    stft_db = normalize_stft(stft_db, None, dtype)

    # Vertically flip the spectrogram, lowest frequencies on the bottom
    # Convert to a C++ contiguous array for OpenCV
    stft_db = np.ascontiguousarray(stft_db[::-1, :])
    bands = bands[::-1]
    y_step_freq = float(bands[0] - bands[1])
    x_step_ms = float(1e3 * (time_vec[1] - time_vec[0]))
    bands = np.around(bands).astype(np.int32).tolist()

    # # Save the spectrogram image to disk
    # cv2.imwrite('debug.tif', stft_db, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

    if not fast_mode:
        # Plot the histogram, ignoring any non-zero values (will no-op if output_path is None)
        global_med_db, global_std_db, global_peak_db = plot_histogram(
            stft_db, ignore_zeros=True, smoothing=512, output_path=debug_path
        )
        # Estimate a global threshold for finding the edges of bat call contours
        global_threshold_std = 2.0
        global_threshold = global_peak_db - global_threshold_std * global_std_db
    else:
        # Fast mode skips bat call segmentation
        global_threshold = 0.0

    # Get a distribution of the max candidate locations
    # Normal mode uses a relatively large window and little overlap
    # Fast mode uses a relatively small window and lots of overlap, since it skips range tightening step
    strides_per_window = 3 if not fast_mode else 16
    window_size_ms = 12 if not fast_mode else 3
    threshold_stddev = 3.0 if not fast_mode else 4.5
    window, stride = calculate_window_and_stride(
        stft_db,
        duration,
        window_size_ms=window_size_ms,
        strides_per_window=strides_per_window,
        time_vec=time_vec,
    )
    candidates, candidate_max_dbs = create_coarse_candidates(
        stft_db, window, stride, threshold_stddev=threshold_stddev
    )

    if fast_mode:
        # combine candidates for efficiency and remove very short candidates (likely noise)
        tmp_ranges = [(x, y) for _, x, y in candidates]
        tmp_ranges = merge_ranges(tmp_ranges, stft_db.shape[1])
        candidate_lengths = np.array([y - x for x, y in tmp_ranges])
        length_thresh = window * 1.5
        idx_remove = candidate_lengths < length_thresh
        candidates = [(ii, x, y) for ii, (x, y) in enumerate(tmp_ranges) if not idx_remove[ii]]
        candidate_max_dbs = []

    # Filter all candidates to the ranges that have a substantial right-side skew
    ranges, reject_idxs = filter_candidates_to_ranges(
        stft_db, candidates, output_path=debug_path, fast_mode=fast_mode
    )

    # Add in user-specified annotations to ranges
    if annotations:
        for start, stop in annotations:
            start_px = int(np.argmin(np.abs(time_vec - start)))
            stop_px = int(np.argmin(np.abs(time_vec - stop)) + 1)
            ranges.append((start_px, stop_px))

    # Merge all range segments into contiguous range blocks
    ranges = merge_ranges(ranges, stft_db.shape[1])

    # Plot the chirp candidates (will no-op if output_path is None)
    plot_chirp_candidates(stft_db, candidate_max_dbs, ranges, reject_idxs, output_path=debug_path)

    if fast_mode:
        # Apply reduced processing without segment refinement or metadata calculation

        segments = {'stft_db': []}
        # Remove a fraction of the window length when not doing call segmentation
        crop_length_l = max(0, int(round(0.15 * window - 1)))
        crop_length_r = max(0, int(round(0.45 * window - 1)))
        for start, stop in ranges:
            segments['stft_db'].append(stft_db[:, start + crop_length_l : stop - crop_length_r])
        metas = {}

    else:

        # Tighten the ranges by looking for substantial right-side skew (use stride for a smaller sampling window)
        ranges = tighten_ranges(stft_db, ranges, stride, duration, output_path=debug_path)

        # Extract chirp metrics and metadata
        segments = {
            'stft_db': [],
            'waveplot': [],
            'costs': [],
            'canvas': [],
        }
        metas = []
        for index, (start, stop) in tqdm.tqdm(list(enumerate(ranges))):
            segment = stft_db[:, start:stop]

            # Step 0.1 - Debugging setup and find peak amplitude (will return None if disabled)
            canvas = create_contour_debug_canvas(segment, index, output_path=debug_path)

            # Step 0.2 - Find the location(s) of peak amplitude
            max_locations = find_max_locations(segment)

            # Step 1 - Scale with PDF
            segment, peak_db, peak_db_std = scale_pdf_contour(
                segment, index, output_path=debug_path
            )
            if None in {peak_db, peak_db_std}:
                continue

            # Step 2 - Apply median filtering to contour
            segment = filter_contour(segment, index, output_path=debug_path)

            # Step 3 - Apply Morphology Open to contour
            segment = morph_open_contour(segment, index, output_path=debug_path)

            # Step 4 - Normalize contour
            segment = normalize_contour(segment, index, output_path=debug_path)

            # # Step 5 (OLD) - Threshold contour
            # segment, med_db, std_db, peak_db = threshold_contour(segment, index, output_path=debug_path)

            # Step 5 - Find primary contour that contains max amplitude
            # (To use a local instead of global threshold, remove the threshold argument here)
            segmentmask, peak, segment_threshold = find_contour_and_peak(
                segment,
                index,
                max_locations,
                peak_db,
                peak_db_std,
                output_path=debug_path,
                threshold=global_threshold,
            )

            if peak is None:
                continue

            # Step 6 - Create final segmentmask
            segmentmask = refine_segmentmask(segmentmask, index, output_path=debug_path)

            # # Step 6 (OLD) - Find the contour with the (most) max amplitude location(s)
            # valid, segmentmask, peak = find_contour_connected_components(segment, index, max_locations, output_path=debug_path)
            # # Step 6 (OLD) - Refine contour by removing any harmonic or echo
            # segmentmask, peak = refine_contour(segment_, index, max_locations, segmentmask, peak, output_path=debug_path)

            # Step 7 - Calculate the first order harmonic and echo region
            harmonic = find_harmonic(segmentmask, index, freq_offset, output_path=debug_path)
            echo = find_echo(segmentmask, index, output_path=debug_path)

            original = stft_db[:, start:stop]
            harmonic_flag, hamonic_peak, echo_flag, echo_peak = calculate_harmonic_and_echo_flags(
                original, index, segmentmask, harmonic, echo, canvas, output_path=debug_path
            )

            # Remove harmonic and echo from segmentation
            segment = remove_harmonic_and_echo(
                segment, index, harmonic, echo, global_threshold, output_path=debug_path
            )

            # Step 8 - Calculate the A* cost grid and bat call start/end points
            costs, grid, call_begin, call_end, boundary = calculate_astar_grid_and_endpoints(
                segment, index, segmentmask, peak, canvas, output_path=debug_path
            )
            top, bottom, left, right = boundary

            # Skip chirp if the extracted path covers a small duration or bandwidth
            bandwidth, duration_, significant = significant_contour_path(
                call_begin, call_end, y_step_freq, x_step_ms
            )
            if not significant:
                continue

            # Step 9 - Extract optimal path from start to end using the cost grid
            path = extract_contour_path(
                grid, call_begin, call_end, canvas, index, output_path=debug_path
            )

            # Step 10 - Extract contour keypoints
            path_smoothed, (knee, fc, heel), slopes = extract_contour_keypoints(
                path, canvas, index, peak, output_path=debug_path
            )

            # Step 11 - Collect chirp metadata
            metadata = {
                'curve.(hz,ms)': [
                    (
                        bands[y],
                        (start + x) * x_step_ms,
                    )
                    for y, x in path_smoothed
                ],
                'start.ms': (start + left) * x_step_ms,
                'end.ms': (start + right) * x_step_ms,
                'duration.ms': (right - left) * x_step_ms,
                'threshold.amp': int(
                    round(255.0 * (segment_threshold / np.iinfo(stft_db.dtype).max))
                ),
                'peak f.ms': (start + peak[1]) * x_step_ms,
                'fc.ms': (start + bands[fc[1]]) * x_step_ms,
                'hi fc:knee.ms': (start + bands[knee[1]]) * x_step_ms,
                'lo fc:heel.ms': (start + bands[heel[1]]) * x_step_ms,
                'bandwidth.hz': bandwidth,
                'hi f.hz': bands[top],
                'lo f.hz': bands[bottom],
                'peak f.hz': bands[peak[0]],
                'fc.hz': bands[fc[0]],
                'hi fc:knee.hz': bands[knee[0]],
                'lo fc:heel.hz': bands[heel[0]],
                'harmonic.flag': harmonic_flag,
                'harmonic peak f.ms': (
                    (start + hamonic_peak[1]) * x_step_ms if harmonic_flag else None
                ),
                'harmonic peak f.hz': bands[hamonic_peak[0]] if harmonic_flag else None,
                'echo.flag': echo_flag,
                'echo peak f.ms': (start + echo_peak[1]) * x_step_ms if echo_flag else None,
                'echo peak f.hz': bands[echo_peak[0]] if echo_flag else None,
            }
            metadata.update(slopes)

            # Normalize values
            for key, value in list(metadata.items()):
                if value is None:
                    continue
                if key.endswith('.ms'):
                    metadata[key] = round(float(value), 3)
                if key.endswith('.hz'):
                    metadata[key] = int(round(value))
                if key.endswith('.flag'):
                    metadata[key] = bool(value)
                if key.endswith('.y_px/x_px'):
                    key_ = key.replace('.y_px/x_px', '.khz/ms')
                    metadata[key_] = round(float(value * ((y_step_freq / 1000.0) / x_step_ms)), 3)
                    metadata.pop(key)
                if key.endswith('.(hz,ms)'):
                    metadata[key] = [
                        (
                            int(round(val1)),
                            round(float(val2), 3),
                        )
                        for val1, val2 in value
                    ]

            metas.append(metadata)

            # Trim segment around the bat call with a small buffer
            buffer_ms = 1.0
            buffer_pix = int(round(buffer_ms / x_step_ms))
            trim_begin = max(0, min(segment.shape[1], call_begin[1] - buffer_pix))
            trim_end = max(0, min(segment.shape[1], call_end[1] + buffer_pix))

            segments['stft_db'].append(stft_db[:, start + trim_begin : start + trim_end])
            segments['waveplot'].append(waveplot[:, start + trim_begin : start + trim_end])
            segments['costs'].append(costs[:, trim_begin:trim_end])
            if debug_path:
                segments['canvas'].append(canvas[:, trim_begin:trim_end])

    # Concatenate extracted, trimmed segments and other matrices
    for key in list(segments.keys()):
        value = segments[key]
        if len(value) == 0:
            segments.pop(key)
            continue
        segments[key] = np.hstack(value)

    if debug_path:
        cv2.imwrite(join(debug_path, 'spectrogram.tif'), stft_db, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        cv2.imwrite(join(debug_path, 'spectrogram.waveplot.png'), waveplot)

        if 'stft_db' in segments:
            cv2.imwrite(
                join(debug_path, 'spectrogram.compressed.tif'),
                segments['stft_db'],
                [cv2.IMWRITE_TIFF_COMPRESSION, 1],
            )

        if 'waveplot' in segments:
            cv2.imwrite(
                join(debug_path, 'spectrogram.compressed.waveplot.png'), segments['waveplot']
            )

        if 'stft_db' in segments and 'waveplot' in segments:
            temp_top = np.stack(
                (segments['stft_db'], segments['stft_db'], segments['stft_db']), axis=2
            )
            temp_bot = cv2.resize(
                segments['waveplot'], temp_top.shape[:2][::-1], interpolation=cv2.INTER_LINEAR
            )
            temp_bot = temp_bot.astype(np.float32) * (
                np.iinfo(temp_top.dtype).max / np.iinfo(temp_bot.dtype).max
            )
            temp_bot = np.around(temp_bot).astype(temp_top.dtype)
            temp = np.vstack((temp_top, temp_bot))
            cv2.imwrite(join(debug_path, 'spectrogram.compressed.combined.png'), temp)

        if 'costs' in segments:
            cv2.imwrite(
                join(debug_path, 'spectrogram.compressed.threshold.tif'),
                segments['costs'],
                [cv2.IMWRITE_TIFF_COMPRESSION, 1],
            )
            temp = segments['costs'].copy()
            flags = segments['costs'] == 0
            temp = normalize_stft(temp, None, np.uint8)
            temp = cv2.applyColorMap(temp, cv2.COLORMAP_JET)
            temp[:, :, 0][flags] = 0
            temp[:, :, 1][flags] = 0
            temp[:, :, 2][flags] = 0
            cv2.imwrite(
                join(debug_path, 'spectrogram.compressed.threshold.jet.tif'),
                temp,
                [cv2.IMWRITE_TIFF_COMPRESSION, 1],
            )

        if 'canvas' in segments:
            cv2.imwrite(
                join(debug_path, 'spectrogram.compressed.keypoints.tif'),
                segments['canvas'],
                [cv2.IMWRITE_TIFF_COMPRESSION, 1],
            )

    output_paths = []
    compressed_paths = []
    if not fast_mode:
        datas = [
            (output_paths, 'jpg', stft_db),
        ]
    else:
        datas = []
    if 'stft_db' in segments:
        datas += [
            (compressed_paths, 'compressed.jpg', segments['stft_db']),
        ]

    for accumulator, tag, data in datas:
        if data.dtype != np.uint8:
            data_ = data.astype(np.float32)
            data_ -= data_.min()
            data_ /= data_.max()
            data_ = np.clip(np.around(data_ * 255.0), 0, 255).astype(np.uint8)
        else:
            data_ = data

        splits = np.arange(chunksize, chunksize + data_.shape[1], chunksize)
        chunks = np.split(data_, splits, axis=1)
        chunks = [chunk for chunk in chunks if chunk.shape[1] > 0]
        total = len(chunks)

        for index, chunk in enumerate(chunks):
            if chunk.shape[1] == 0:
                continue
            output_path = join(output_folder, f'{base}.{index + 1:02d}of{total:02d}.{tag}')
            cv2.imwrite(output_path, chunk, [cv2.IMWRITE_JPEG_QUALITY, 80])
            accumulator.append(output_path)

    log.debug(f'Rendered {len(output_paths)} spectrograms')

    max_value = np.iinfo(stft_db.dtype).max
    metadata = {
        'wav.path': wav_filepath,
        'spectrogram': {
            'uncompressed.path': output_paths,
            'compressed.path': compressed_paths,
        },
        'global_threshold.amp': int(round(255.0 * (global_threshold / max_value))),
        'sr.hz': int(sr),
        'duration.ms': round(duration * 1e3, 3),
        'frequencies': {
            'min.hz': int(FREQ_MIN),
            'max.hz': int(FREQ_MAX),
            'pixels.hz': bands,
        },
        'size': {
            'uncompressed': {
                'width.px': stft_db.shape[1],
                'height.px': stft_db.shape[0],
            },
            'compressed': None,
        },
        'segments': metas,
    }
    if 'stft_db' in segments:
        metadata['size']['compressed'] = (
            {
                'width.px': segments['stft_db'].shape[1],
                'height.px': segments['stft_db'].shape[0],
            },
        )

    metadata_path = join(output_folder, f'{base}.metadata.json')
    with open(metadata_path, 'w') as metafile:
        json.dump(metadata, metafile, indent=4)

    return output_paths, metadata_path, metadata
