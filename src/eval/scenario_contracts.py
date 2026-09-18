"""Execution prerequisites shared by authored specifications and release validation."""

from .config_models import BenchmarkCase


def validate_execution_prerequisites(case: BenchmarkCase) -> list[str]:
    """A fresh scenario can search only the attachments declared before its turns.

    Setup turns contain text requests, not upload operations. They cannot supply
    an undeclared file or inherit one from a different scenario's session.
    """
    if "upload_search" in case.expected_tools and not case.resolved_upload_fixtures:
        return ["upload_search requires a declared attachment in an isolated scenario"]
    return []
