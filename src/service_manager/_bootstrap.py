from __future__ import annotations

import logging

from ..logging_utils import configure_logging
from ..runtime_encoding import ensure_utf8_stdio


ensure_utf8_stdio()
configure_logging()
logger = logging.getLogger("src.service_manager")
