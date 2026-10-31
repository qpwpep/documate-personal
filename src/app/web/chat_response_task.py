"""Receive a chat response independently of Streamlit reruns and navigation."""
from __future__ import annotations

import logging
from contextlib import closing
from dataclasses import dataclass, field, replace
from queue import Empty, Queue
from threading import Event, Thread
from time import monotonic

from src.app.client import AgentSessionClient, AgentStreamEvent
from src.core.contracts.outcome import AgentCallResult
from src.app.web.streamlit_chat import progress_message_for_event
from src.app.web.streamlit_state import Conversation
from src.core.contracts.outcome import TurnResult


logger = logging.getLogger(__name__)


@dataclass
class ChatResponseTask:
    operation_id: str
    client: AgentSessionClient
    events: Queue[AgentStreamEvent] = field(default_factory=Queue)
    finished: Event = field(default_factory=Event)
    stopped: Event = field(default_factory=Event)
    started_at: float = field(default_factory=monotonic)
    progress: str = "요청을 접수했습니다."
    request_id: str = ""
    errors: list[str] = field(default_factory=list)
    result: AgentCallResult | None = None

    def start(self, prompt: str) -> None:
        Thread(target=self._receive, args=(prompt,), name="documate-chat-response", daemon=True).start()

    def _receive(self, prompt: str) -> None:
        try:
            with closing(self.client.stream(prompt)) as stream:
                for event in stream:
                    if self.stopped.is_set():
                        return
                    self.events.put(event)
        except Exception:
            # Unexpected worker failures must release the UI and remain visible
            # in the server log instead of leaving an endless pending response.
            logger.exception("chat_response_receiver_failed")
        finally:
            self.finished.set()


def start_chat_response(conversation: Conversation, prompt: str, operation_id: str) -> None:
    manifest = conversation.client.manifest
    task = ChatResponseTask(
        operation_id=operation_id,
        client=AgentSessionClient(
            replace(conversation.client.context),
            manifest=manifest.model_copy(deep=True) if manifest is not None else None,
        ),
    )
    conversation.messages.append({"role": "user", "content": prompt})
    conversation.chat_task = task
    task.start(prompt)


def collect_chat_response(conversation: Conversation) -> bool:
    """Apply received events on the UI thread; return whether an answer ended."""
    task = conversation.chat_task
    if task is None:
        return False
    submission = conversation.active_submission
    if submission is None or submission.operation_id != task.operation_id:
        task.stopped.set()
        conversation.chat_task = None
        return False
    finished = task.finished.is_set()
    while True:
        try:
            event = task.events.get_nowait()
        except Empty:
            break
        task.request_id = event.observation.request_id or str(event.data.get("request_id") or task.request_id)
        if event.event == "final_response" and event.result is not None:
            task.result = event.result
        elif event.event == "error":
            message = str(event.data.get("message") or "응답 처리 중 오류가 발생했습니다.")
            if message not in task.errors:
                task.errors.append(message)
            task.progress = message
        else:
            task.progress = progress_message_for_event(event) or task.progress
    if not finished:
        return False
    if task.result is None:
        conversation.messages.append({
            "role": "assistant", "result": None,
            "transport_errors": task.errors or ["응답을 받지 못했습니다. 질문을 수정하거나 잠시 후 다시 시도해 주세요."],
            "request_id": task.request_id,
        })
    else:
        conversation.messages.append({
            "role": "assistant",
            "result": TurnResult.model_validate(task.result.model_dump(exclude={"upload_manifest"})),
            "transport_errors": task.errors,
        })
    conversation.client.manifest = task.client.manifest
    conversation.active_submission = None
    conversation.operation_in_progress = False
    conversation.chat_task = None
    # Keep the mounted composer intact: browser typing can be newer than the
    # last draft event Python received while this response was pending.
    return True
