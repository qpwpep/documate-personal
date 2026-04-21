from __future__ import annotations

import logging

from src.infra.logging_utils import configure_logging
from src.infra.runtime_encoding import ensure_utf8_stdio


ensure_utf8_stdio()
configure_logging()
logger = logging.getLogger("src.app.service_manager")
