"""Exercise an installed BatBot package outside the source checkout.

Run with ``python -I scripts/smoke_install.py``. CUDA checks also verify kernel
execution on the GPU through ONNX Runtime profiling.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

import numpy as np

from batbot import classifier
from batbot._paths import write_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=['cpu', 'cuda', 'coreml'], default='cpu')
    parser.add_argument('--runtime', choices=['cpu', 'gpu'], default='cpu')
    args = parser.parse_args()
    expected = 'onnxruntime-gpu' if args.runtime == 'gpu' else 'onnxruntime'
    excluded = 'onnxruntime' if args.runtime == 'gpu' else 'onnxruntime-gpu'
    print(
        f'{sys.platform}: Python {sys.version.split()[0]}, {expected} '
        f'{importlib.metadata.version(expected)}'
    )
    try:
        importlib.metadata.version(excluded)
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise AssertionError('Install only one ONNX Runtime distribution')
    provider = {
        'cpu': 'CPUExecutionProvider',
        'cuda': 'CUDAExecutionProvider',
        'coreml': 'CoreMLExecutionProvider',
    }[args.provider]
    with tempfile.TemporaryDirectory(prefix='batbot-smoke-') as temporary:
        root = Path(temporary) / 'résultats 空間 [batbot]'
        root.mkdir()
        image = root / 'spectrogramme.jpg'
        write_image(image, np.zeros((300, 700, 3), dtype=np.uint8))
        for reduced in (False, True):
            os.environ['BATBOT_REDUCED'] = '1' if reduced else '0'
            runner = classifier.Classifier(providers=[provider], num_workers=2)
            results = runner.classify([image, image], progress=False)
            assert results[0]['label'] == results[1]['label']
            np.testing.assert_allclose(
                list(results[0]['scores'].values()),
                list(results[1]['scores'].values()),
                rtol=1e-5,
                atol=1e-7,
            )
            assert results[0]['label'] in classifier.CLASSES
            assert len(results[0]['scores']) == len(classifier.CLASSES)
            assert provider in runner.session.get_providers()
            if args.provider == 'coreml':
                assert runner.session.get_inputs()[0].shape[0] == 1
            options = runner.session.get_session_options()
            assert options.intra_op_num_threads == (2 if reduced else 0)
            assert options.inter_op_num_threads == (2 if reduced else 0)

        from scipy.io import wavfile

        rate = 250_000
        random = np.random.default_rng(0)
        samples = random.integers(-300, 300, size=rate // 5, dtype=np.int16)
        wav = root / 'recording.WAV'
        wavfile.write(wav, rate, samples)
        result = runner.classify_bulk([wav], input_type='wav', preprocess_workers=2, progress=False)
        assert result['summary']['classified'] == 1, result
        assert result['summary']['failed'] == 0, result
        del runner  # Release native accelerator resources before launching another process.

        executable = Path(sysconfig.get_path('scripts')) / (
            'batbot.exe' if os.name == 'nt' else 'batbot'
        )
        command = subprocess.run(
            [str(executable), 'classify', str(image), '--provider', args.provider, '--no-progress'],
            check=False,
            capture_output=True,
            text=True,
            encoding='utf8',
        )
        assert command.returncode == 0, command.stderr
        assert json.loads(command.stdout)['summary']['classified'] == 1

        if args.provider == 'cuda':
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.enable_profiling = True
            options.profile_file_prefix = str(Path(temporary) / 'cuda-profile')
            session = ort.InferenceSession(
                classifier.fetch(),
                sess_options=options,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            )
            windows, _ = next(classifier.pre([image]))
            session.run(None, {session.get_inputs()[0].name: windows[:1]})
            profile = json.loads(Path(session.end_profiling()).read_text(encoding='utf8'))
            assert any(
                event.get('args', {}).get('provider') == 'CUDAExecutionProvider'
                for event in profile
            ), 'No model kernels executed on CUDA'
    print('Installed package, CLI, Unicode paths, WAV workers, and inference: OK')


if __name__ == '__main__':
    main()
