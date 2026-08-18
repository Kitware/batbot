import subprocess
import sys

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


def test_example():
    batbot.example()
