"""Shared, testable tutorial utilities; not part of BatBot's stable public API."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import zipfile
from pathlib import Path


def native_path(value):
    return Path(os.path.expandvars(os.path.expanduser(os.fspath(value))))


def runtime_profile(requested):
    """Choose an explicit runtime; auto inspects the driver's supported CUDA major."""
    if requested == 'auto':
        try:
            status = subprocess.run(
                ['nvidia-smi'], capture_output=True, text=True, check=False, timeout=15
            )
        except (OSError, subprocess.TimeoutExpired):
            status = None
        match = re.search(r'CUDA Version:\s*(\d+)\.', status.stdout) if status else None
        major = int(match[1]) if match and status.returncode == 0 else 0
        requested = 'cuda13' if major >= 13 else 'cuda12' if major >= 12 else 'cpu'
    profiles = {
        'cpu': ('CPUExecutionProvider', 'onnxruntime>=1.21'),
        'coreml': ('CoreMLExecutionProvider', 'onnxruntime>=1.21'),
        # Colab drivers may only support CUDA 12; do not install the CUDA 13 extra there.
        'cuda12': ('CUDAExecutionProvider', 'onnxruntime-gpu[cuda,cudnn]==1.26.0'),
        'cuda13': ('CUDAExecutionProvider', 'onnxruntime-gpu[cuda,cudnn]>=1.31,<1.32'),
    }
    if requested not in profiles:
        raise ValueError('Choose cpu, coreml, auto, cuda12, or cuda13')
    provider, requirement = profiles[requested]
    return requested, provider, requirement


def prepare_environment(repo, requested='cpu', install=True):
    """Install into the active kernel, never silently uninstalling another runtime."""
    if sys.version_info < (3, 11):
        raise RuntimeError('Select a Python 3.11 or newer notebook runtime.')
    selected, provider, requirement = runtime_profile(requested)
    opposite = 'onnxruntime' if selected.startswith('cuda') else 'onnxruntime-gpu'
    try:
        importlib.metadata.version(opposite)
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError(
            f'{opposite} is already installed. CPU/GPU distributions share one import. '
            f'Use a fresh runtime, or explicitly uninstall {opposite} and restart the kernel '
            'before changing profiles. Do not install both distributions.'
        )
    if install and os.getenv('BATBOT_NOTEBOOK_SKIP_INSTALL') != '1':
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '-e', f'{Path(repo)}[notebook]', requirement]
        )
        subprocess.check_call([sys.executable, '-m', 'pip', 'check'])
    print(f'Runtime profile: {selected}; required provider: {provider}')
    return selected, provider


def sample_wav(repo, cache):
    """Prefer the LFS fixture; otherwise fetch its integrity-checked public copy."""
    candidate = Path(repo) / 'examples' / 'example1.wav'
    if candidate.is_file():
        with candidate.open('rb') as stream:
            header = stream.read(12)
        if header[:4] == b'RIFF' and header[8:12] == b'WAVE':
            return candidate
    import pooch

    return Path(
        pooch.retrieve(
            url='https://media.githubusercontent.com/media/Kitware/batbot/main/examples/example1.wav',
            known_hash='sha256:391efce5433d1057caddb4ce07b9712c523d6a815e4ee9e64b62973569982925',
            path=native_path(cache),
            fname='example1.wav',
            progressbar=True,
        )
    )


def choose_inputs(directory, repo, cache, maximum=8, allow_example=True, only_wav=False):
    from batbot import classifier

    if maximum <= 0:
        raise ValueError('maximum must be positive')
    directory = native_path(directory)
    paths = []
    if directory.exists():
        paths = classifier.discover_inputs(directory, input_type='wav' if only_wav else 'auto')
    if not paths and allow_example:
        print('No matching user files found; using the public example (not a labeled test).')
        paths = [str(sample_wav(repo, cache))]
    if not paths:
        raise FileNotFoundError(f'Add WAV/JPG/PNG files under {directory}, or enable the example.')
    if len(paths) > maximum:
        print(
            f'Using the first {maximum} of {len(paths)} files. Increase MAX_FILES for a larger run.'
        )
    return [Path(path) for path in paths[:maximum]]


def crop_recording(source, output, start=0.0, seconds=1.0):
    """Create a bounded mono inspection clip; leave the original untouched."""
    import librosa
    import soundfile as sf
    from scipy.io import wavfile

    if not math.isfinite(start) or not math.isfinite(seconds) or start < 0 or not 0 < seconds <= 10:
        raise ValueError('Choose start >= 0 and 0 < seconds <= 10 for an inspection clip')
    source, output = native_path(source), native_path(output)
    if source.resolve() == output.resolve() or (output.exists() and source.samefile(output)):
        raise ValueError('The inspection output must not overwrite the original recording')
    metadata = sf.info(str(source))
    if int(start * metadata.samplerate) >= metadata.frames:
        raise ValueError('Inspection clip is too short or starts past the end of the recording')
    samples, rate = librosa.load(str(source), sr=None, offset=start, duration=seconds)
    if len(samples) < 512:
        raise ValueError('Inspection clip is too short or starts past the end of the recording')
    if rate <= 11_000:
        raise ValueError('Sample rate is too low for BatBot\'s 5–120 kHz analysis band')
    if rate < 240_000:
        print(
            'Warning: low sample rate limits usable ultrasonic bandwidth; resampling cannot restore it.'
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output, rate, samples)
    return samples, rate, output


def result_tables(report):
    import pandas as pd

    rows = []
    scores = []
    for result in report['results']:
        rows.append(
            {
                'path': result['path'],
                'file': Path(result['path']).name,
                'label': result.get('label'),
                'confidence': result.get('confidence'),
                'windows': result.get('window_count'),
                'error': result.get('error'),
            }
        )
        if 'scores' in result:
            scores.append({'path': result['path'], **result['scores']})
    return (
        pd.DataFrame(rows, columns=['path', 'file', 'label', 'confidence', 'windows', 'error']),
        pd.DataFrame(scores).set_index('path') if scores else pd.DataFrame(),
    )


def plot_results(report, destination):
    """Count classified files, not calls or individual bats."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    table, scores = result_tables(report)
    figure, axes = plt.subplots(1, 3, figsize=(17, 4.5), layout='constrained')
    counts = report['summary']['label_counts']
    if counts:
        labels = sorted(counts, key=lambda label: (-counts[label], label))
        axes[0].bar(
            labels,
            [counts[label] for label in labels],
            color=['#8b949e' if label == 'NOISE' else '#087f8c' for label in labels],
        )
        axes[0].tick_params(axis='x', rotation=60)
        axes[0].set_ylabel('Classified files')
        axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes[1].hist(table['confidence'].dropna(), bins=np.linspace(0, 1, 11), color='#087f8c')
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel('Top model score (not calibrated probability)')
        axes[1].set_ylabel('Classified files')
        axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
        top = scores.mean().nlargest(min(8, len(scores.columns))).index
        shown = scores.loc[:, top].iloc[:20]
        matrix = axes[2].imshow(shown, aspect='auto', vmin=0, vmax=1, cmap='viridis')
        axes[2].set_xticks(range(len(top)), top, rotation=60)
        names = [f'{index + 1}. {Path(path).name}' for index, path in enumerate(shown.index)]
        axes[2].set_yticks(range(len(shown)), names)
        figure.colorbar(matrix, ax=axes[2], label='Model score')
    else:
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                'No successful predictions; inspect the error table.',
                ha='center',
                va='center',
                transform=axis.transAxes,
                wrap=True,
            )
    for axis, title in zip(
        axes,
        ['File predictions, including NOISE', 'Score distribution', 'Top classes; first 20 files'],
    ):
        axis.set_title(title)
    destination = native_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=140, bbox_inches='tight')
    return figure


def show_images(paths, destination=None, limit=6, title='Processing outputs'):
    """Read Unicode filenames and preserve 16-bit TIFF intensities for display."""
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    paths = list(paths)[:limit]
    figure, axes = plt.subplots(
        max(1, len(paths)),
        1,
        figsize=(12, max(3, 2.8 * len(paths))),
        squeeze=False,
        layout='constrained',
    )
    for axis, path in zip(axes[:, 0], paths):
        data = cv2.imdecode(
            np.frombuffer(Path(path).read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )
        if data is None:
            raise OSError(f'Unable to decode {path}')
        if data.ndim == 3:
            data = cv2.cvtColor(
                data, cv2.COLOR_BGRA2RGBA if data.shape[2] == 4 else cv2.COLOR_BGR2RGB
            )
            if np.issubdtype(data.dtype, np.integer) and data.dtype != np.uint8:
                data = data.astype(np.float32) / np.iinfo(data.dtype).max
        stride = max(1, int(np.ceil(data.shape[1] / 2000)))
        axis.imshow(data[:, ::stride], aspect='auto', cmap='magma')
        axis.set_title(Path(path).name)
        axis.set_axis_off()
    if not paths:
        axes[0, 0].text(0.5, 0.5, 'No outputs of this kind were produced.', ha='center')
    figure.suptitle(title)
    if destination is not None:
        figure.savefig(destination, dpi=120, bbox_inches='tight')
    return figure


def plot_stages(clip, waveform, sample_rate, destination):
    import matplotlib.pyplot as plt
    import numpy as np

    from batbot.spectrogram import gain_stft, load_stft, normalize_stft

    raw, _, rate, bands, duration, _, times, original_rate, max_band = load_stft(
        str(clip), fast_mode=True
    )
    gained = gain_stft(raw.copy(), max_band_idx=max_band)
    quantized = normalize_stft(gained.copy(), None, np.uint16)
    figure, axes = plt.subplots(4, 1, figsize=(12, 11), layout='constrained')
    stride = max(1, len(waveform) // 15000)
    axes[0].plot(np.arange(len(waveform))[::stride] / sample_rate, waveform[::stride], lw=0.5)
    axes[0].set(xlabel='Clip time (s)', ylabel='Amplitude', title='1. Mono inspection waveform')
    stride = max(1, int(np.ceil(raw.shape[1] / 3000)))
    for axis, data, title, unit in zip(
        axes[1:],
        [raw, gained, quantized],
        ['2. Band-limited STFT', '3. Median subtraction, gain, clipping', '4. 16-bit quantization'],
        ['Relative power (dB)', 'Shifted/clipped intensity', 'Integer intensity'],
    ):
        artist = axis.imshow(
            data[:, ::stride],
            origin='lower',
            aspect='auto',
            cmap='magma',
            extent=[times[0], times[-1], bands[0] / 1000, bands[-1] / 1000],
        )
        axis.set(xlabel='Clip time (s)', ylabel='Frequency (kHz)', title=title)
        figure.colorbar(artist, ax=axis, label=unit)
    figure.savefig(destination, dpi=120, bbox_inches='tight')
    return figure, {
        'analysis_sample_rate': rate,
        'original_sample_rate': original_rate,
        'duration_seconds': duration,
        'stft_shape': raw.shape,
        'quantized_dtype': str(quantized.dtype),
    }


def window_scores(runner, windows):
    import numpy as np

    session = runner.session
    tensor = session.get_inputs()[0]
    step = 1 if tensor.shape[0] == 1 else runner.batch_size
    return np.vstack(
        [
            session.run(None, {tensor.name: windows[start : start + step]})[0]
            for start in range(0, len(windows), step)
        ]
    )


def export_report(report, runner, destination):
    import onnxruntime as ort

    import batbot

    destination = native_path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    provenance = {
        'batbot_version': batbot.__version__,
        'python': platform.python_version(),
        'platform': platform.platform(),
        'onnxruntime_version': ort.__version__,
        'providers': runner.session.get_providers(),
        'model': runner.config.filename,
        'model_sha256': runner.config.sha256,
        'batch_size': runner.batch_size,
        'inference_workers': runner.num_workers,
    }
    table, scores = result_tables(report)
    (destination / 'results.json').write_text(
        json.dumps({**report, 'run_info': provenance}, indent=2), encoding='utf8'
    )
    table.to_csv(destination / 'results.csv', index=False)
    scores.to_csv(destination / 'class_scores.csv')
    return provenance


def archive_outputs(directory):
    """Archive generated outputs only, ignoring links outside the output tree."""
    directory = native_path(directory).resolve()
    archive = directory.parent / f'{directory.name}.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(directory.rglob('*')):
            if (
                path.is_file()
                and not path.is_symlink()
                and path.resolve().is_relative_to(directory)
            ):
                bundle.write(path, path.relative_to(directory))
    return archive
