from src.infra.settings import AppSettings
from src.runtime.nodes.synthesis.budgets import ExcerptLimits


def synthesis_excerpt_limits(*, normal_chars: int | None = None, compact_chars: int | None = None) -> ExcerptLimits:
    """Build test inputs from the application's declared defaults unless a case overrides them."""
    return ExcerptLimits(
        normal_chars=(AppSettings.model_fields["synthesis_prompt_snippet_chars"].default
                      if normal_chars is None else normal_chars),
        compact_chars=(AppSettings.model_fields["synthesis_compact_prompt_snippet_chars"].default
                       if compact_chars is None else compact_chars),
    )
