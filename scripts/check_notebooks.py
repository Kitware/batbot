"""Validate clean tutorial sources, or execute them against local fixtures.

Execution never modifies source notebooks. Generated outputs go to an ignored
run directory; package installation, cloud uploads, and downloads are disabled.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / 'examples' / 'notebooks'


def validate(path, clear=False):
    import nbformat

    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source, filename=f'{path.name}:{cell.id}')
            if clear:
                cell.outputs = []
                cell.execution_count = None
            elif cell.outputs or cell.execution_count is not None:
                raise ValueError(f'{path}: clear outputs/execution counts before committing')
        elif cell.get('attachments'):
            raise ValueError(f'{path}: embedded attachments are not allowed in tutorial sources')
    if clear:
        nbformat.write(notebook, path)
    return notebook


def execute(notebook, path, destination):
    import nbformat
    import numpy as np
    from nbclient import NotebookClient
    from scipy.io import wavfile

    from batbot._paths import write_image

    fixture = ROOT / 'examples' / 'example1.wav'
    with fixture.open('rb') as stream:
        if stream.read(4) != b'RIFF':
            raise ValueError('Notebook execution requires real WAV fixtures: run git lfs pull')
    work = destination / path.stem
    inputs = work / 'inputs 空間'
    inputs.mkdir(parents=True)
    rate, samples = wavfile.read(fixture)
    wavfile.write(inputs / 'recording.WAV', rate, samples[: rate // 2])
    write_image(inputs / 'spectrogram.JPG', np.zeros((300, 700, 3), dtype=np.uint8))
    environment = {
        **os.environ,
        'BATBOT_NOTEBOOK_REPO': str(ROOT),
        'BATBOT_NOTEBOOK_SKIP_INSTALL': '1',
        'BATBOT_NOTEBOOK_INPUT_DIR': str(inputs),
        'BATBOT_NOTEBOOK_OUTPUT_DIR': str(work / 'outputs'),
        'BATBOT_NOTEBOOK_CLIP_SECONDS': '0.5',
        'MPLBACKEND': 'Agg',
        'MPLCONFIGDIR': str(work / 'matplotlib'),
        'ORT_DISABLE_TELEMETRY': '1',
    }
    client = NotebookClient(notebook, timeout=600, kernel_name='python3')
    try:
        client.execute(cwd=str(work), env=environment)
    finally:
        nbformat.write(notebook, work / path.name)
    charts = sum(
        'image/png' in output.get('data', {})
        for cell in notebook.cells
        if cell.cell_type == 'code'
        for output in cell.outputs
    )
    if charts < 3:
        raise AssertionError(f'{path.name}: expected rendered charts, got {charts}')
    reports = list(work.glob('outputs/run-*/results.json'))
    if len(reports) != 1:
        raise AssertionError(f'{path.name}: missing final classification report')
    report = json.loads(reports[0].read_text(encoding='utf8'))
    expected = 2 if path.name.startswith('01_') else 1
    assert report['summary']['classified'] == expected, report['summary']
    assert report['summary']['failed'] == 0, report['results']
    assert report['run_info']['providers'] == ['CPUExecutionProvider']
    print(f'{path.name}: {expected} predictions, {charts} rendered charts, exports OK')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clear', action='store_true', help='Clear source outputs mechanically')
    parser.add_argument(
        '--execute', action='store_true', help='Execute using local fixtures and CPU'
    )
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/notebooks/validation')
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    paths = sorted(NOTEBOOKS.glob('*.ipynb'))
    if not paths:
        parser.error('No tutorial notebooks found')
    destination = None
    if args.execute:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        destination = Path(tempfile.mkdtemp(prefix='validation-', dir=args.output_dir))
        print(f'Executed notebooks and artifacts: {destination}', flush=True)
    for path in paths:
        notebook = validate(path, clear=args.clear)
        print(f'{path.name}: clean source/schema/Python OK', flush=True)
        if destination is not None:
            execute(notebook, path, destination)


if __name__ == '__main__':
    main()
