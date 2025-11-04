# new_src/slack_bot_notifier.py
import os
from typing import Optional, Dict, Tuple
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
DEFAULT_CHANNEL = os.getenv("SLACK_DEFAULT_CHANNEL")
POST_AS_THREAD = os.getenv("SLACK_POST_AS_THREAD", "true").lower() == "true"

_client: Optional[WebClient] = WebClient(token=SLACK_BOT_TOKEN) if SLACK_BOT_TOKEN else None

# session_id → (channel, thread_ts)
THREAD_TS_CACHE: Dict[str, Tuple[str, str]] = {}

def enabled() -> bool:
    return _client is not None

def open_im_channel(user_id: str) -> Optional[str]:
    """유저 ID(Uxxxx)로 DM 채널을 개설하고 채널 ID를 반환"""
    if not enabled():
        return None
    try:
        resp = _client.conversations_open(users=[user_id])
        return resp["channel"]["id"]
    except SlackApiError as e:
        print(f"[slack] open_im_channel failed: {e.response['error']}")
    except Exception as e:
        print(f"[slack] open_im_channel failed: {e}")
    return None

def _resolve_destination(channel: Optional[str], session_id: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    반환: (channel_id, thread_ts or None)
    - session_id가 있고 캐시에 thread_ts가 있으면 그 스레드로
    - 없으면 새 포스트(스레드 생성)
    """
    # 목적 채널 결정
    channel_id = channel or DEFAULT_CHANNEL
    if not channel_id:
        return None, None

    # 스레드 재사용
    if POST_AS_THREAD and session_id and session_id in THREAD_TS_CACHE:
        cached_channel, thread_ts = THREAD_TS_CACHE[session_id]
        # 채널이 바뀌면 새로 시작
        if cached_channel == channel_id:
            return channel_id, thread_ts

    return channel_id, None  # 새 포스트

def post_message(
    text: str,
    *,
    session_id: Optional[str] = None,
    origin: str = "web",
    channel: Optional[str] = None,
    mrkdwn: bool = True
) -> None:
    """Slack에 메시지 게시. 세션별 스레딩을 자동 처리."""
    if not enabled():
        return

    # 헤더 + 본문
    header = f"*Source*: `{origin}`"
    sid = f"\n*Session*: `{session_id}`" if session_id else ""
    final_text = f"{header}{sid}\n\n{text}"

    channel_id, thread_ts = _resolve_destination(channel, session_id)
    if not channel_id:
        print("[slack] No channel resolved; set SLACK_DEFAULT_CHANNEL or pass channel explicitly.")
        return

    try:
        resp = _client.chat_postMessage(
            channel=channel_id,
            text=final_text,
            mrkdwn=mrkdwn,
            thread_ts=thread_ts,
            unfurl_links=False,
            unfurl_media=False,
        )
        # 첫 포스트면 thread_ts 저장 (세션 스레딩)
        if POST_AS_THREAD and session_id and not thread_ts:
            THREAD_TS_CACHE[session_id] = (channel_id, resp["ts"])
    except SlackApiError as e:
        print(f"[slack] chat.postMessage failed: {e.response['error']}")
    except Exception as e:
        print(f"[slack] chat.postMessage failed: {e}")
