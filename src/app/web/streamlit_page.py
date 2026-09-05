from __future__ import annotations

import logging
import sys
from functools import lru_cache

from src.infra.logging_utils import log_event


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def warn_if_utf8_mode_disabled_once() -> None:
    if sys.flags.utf8_mode == 1:
        return

    log_event(
        logger,
        logging.WARNING,
        "utf8_mode_disabled",
        suggested_command=(
            "uv run python -X utf8 -m streamlit run src/app/web/streamlit_app.py --server.port 8501"
        ),
    )
