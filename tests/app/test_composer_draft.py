"""Draft events are scoped to a conversation and cannot resurrect sent text."""
from src.app.web.composer_draft import ComposerDraft


def test_drafts_keep_exact_text_and_remain_independent():
    first, second = ComposerDraft(), ComposerDraft()
    first.record("  첫 질문\n두 번째 줄  ", revision=0)
    second.record("다른 대화", revision=0)
    assert first.text == "  첫 질문\n두 번째 줄  "
    assert second.text == "다른 대화"


def test_submission_does_not_restore_a_delayed_draft_event():
    draft = ComposerDraft()
    draft.record("보낸 질문", revision=0)
    draft.consume()
    draft.record("보낸 질문", revision=0)
    assert draft.text == ""
    draft.record("다음 질문", revision=draft.revision)
    assert draft.text == "다음 질문"


def test_edit_replaces_draft_without_accepting_an_older_browser_event():
    draft = ComposerDraft()
    draft.record("이전 초안", revision=0)
    draft.edit("실패했던 질문")
    draft.record("이전 초안", revision=0)
    assert draft.text == "실패했던 질문"
    assert draft.needs_restore
