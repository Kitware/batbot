"""Process-wide BatBot settings that are safe to import from any submodule."""

import logging
import os

VERBOSE = os.getenv('BATBOT_VERBOSE', os.getenv('VERBOSE')) is not None
QUIET = not VERBOSE

log = logging.getLogger('batbot')
log.addHandler(logging.NullHandler())
