import logging
from logging.handlers import TimedRotatingFileHandler

import rich

from batbot import utils


def test_init_logging_configures_file_and_rich_handlers(monkeypatch, tmp_path):
    configured = {}
    root = logging.getLogger()
    original_level = root.level
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rich, 'reconfigure', lambda **kwargs: configured.update(theme=kwargs))
    monkeypatch.setattr(logging, 'basicConfig', lambda **kwargs: configured.update(logging=kwargs))

    try:
        logger = utils.init_logging()
    finally:
        root.setLevel(original_level)

    handlers = configured['logging']['handlers']
    try:
        assert logger.name == 'batbot'
        assert configured['logging']['level'] == utils.DEFAULT_LOG_LEVEL
        assert any(isinstance(handler, TimedRotatingFileHandler) for handler in handlers)
        assert configured['theme']['theme'].styles['logging.level.error'].bold is True
    finally:
        for handler in handlers:
            handler.close()
