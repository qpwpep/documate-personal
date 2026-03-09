from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import requests


@dataclass
class AgentRequestContext:
    fastapi_url: str
    session_id: str
    slack_user_id: str = ""
    slack_email: str = ""
    slack_channel_id: str = ""
    upload_file_path: str | None = None


@dataclass
class AgentCallResult:
    answer: str
    file_path: str | None = None
    evidence_items: list[Any] = field(default_factory=list)


def get_agent_response(user_input: str, context: AgentRequestContext) -> AgentCallResult:
    endpoint = f"{context.fastapi_url}/agent"
    try:
        payload = {
            "query": user_input,
            "session_id": context.session_id,
        }

        if context.slack_user_id:
            payload["slack_user_id"] = context.slack_user_id
        if context.slack_email:
            payload["slack_email"] = context.slack_email
        if context.slack_channel_id:
            payload["slack_channel_id"] = context.slack_channel_id
        if context.upload_file_path:
            payload["upload_file_path"] = context.upload_file_path

        resp = requests.post(endpoint, json=payload, timeout=60)

        if resp.status_code == 200:
            data = resp.json()
            response_payload = data.get("response") or {}
            if isinstance(response_payload, dict):
                answer = str(response_payload.get("answer", "") or "")
                evidence = response_payload.get("evidence")
                evidence_items = evidence if isinstance(evidence, list) else []
            else:
                answer = str(response_payload)
                evidence_items = []
            return AgentCallResult(
                answer=answer,
                file_path=data.get("file_path"),
                evidence_items=evidence_items,
            )

        return AgentCallResult(
            answer=(
                f"Agent 호출 실패: 상태 코드 {resp.status_code}\n"
                f"응답: {resp.text}"
            ),
        )

    except requests.exceptions.Timeout:
        return AgentCallResult(
            answer="요청이 타임아웃되었습니다. 서버 상태를 확인해 주세요.",
        )
    except requests.exceptions.ConnectionError:
        return AgentCallResult(
            answer="FastAPI 서버에 연결할 수 없습니다. 서버(8000번 포트) 실행 여부를 확인해 주세요.",
        )
    except Exception as exc:
        return AgentCallResult(
            answer=f"요청 중 예기치 않은 오류가 발생했습니다: {exc}",
        )
