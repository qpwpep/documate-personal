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


def prepare_synthesis_state(state, *, limits=None, decisions=None):
    """Run real preparation, replacing only the Decisions provider HTTP boundary."""
    from src.runtime.nodes.validation import make_pre_synthesis_validation_node
    from tests.core.helpers import make_decisions_client

    state.update(make_pre_synthesis_validation_node(
        False, decisions_client=decisions or make_decisions_client(),
        excerpt_limits=limits or synthesis_excerpt_limits(),
    )(state))
    return state
