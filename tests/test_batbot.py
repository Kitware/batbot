import json
import subprocess
import sys
from pathlib import Path

import batbot


def test_import_is_lightweight():
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                'import sys; import batbot; '
                'assert "batbot.classifier" not in sys.modules; '
                'assert "batbot.spectrogram" not in sys.modules'
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_example(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    batbot.example()
    assert 'Example pipeline completed' in capsys.readouterr().out
    metadata = json.loads((tmp_path / 'output/example1.metadata.json').read_text())
    assert metadata['segments'] and metadata['metadata_origsr']['sr.hz'] == 500_000
    for category in ['uncompressed.path', 'compressed.path', 'mask.path', 'masked.path']:
        paths = metadata['spectrogram'][category]
        assert len(paths) == (4 if category == 'uncompressed.path' else 2)
        assert all(Path(path).is_file() for path in paths)
