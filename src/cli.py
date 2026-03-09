from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from langchain_core.messages import AIMessage

from .graph_builder import build_agent_graph
from .logging_utils import configure_logging, log_event
from .runtime_encoding import ensure_utf8_stdio, maybe_reexec_with_utf8
from .settings import ConfigurationError, get_settings, validate_required_keys


ensure_utf8_stdio()
configure_logging()
logger = logging.getLogger(__name__)


def _write_line(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _load_validated_settings(context: str):
    settings = get_settings()
    validate_required_keys(settings, context=context)
    return settings


def _maybe_dump_mermaid_png(graph, output_path: str | None) -> None:
    if not output_path:
        return

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    png_bytes = graph.get_graph().draw_mermaid_png()
    target_path.write_bytes(png_bytes)
    log_event(logger, logging.INFO, "graph_dumped", output_path=target_path)


def run_cli(*, dump_graph: str | None = None) -> int:
    try:
        settings = _load_validated_settings("cli")
    except ConfigurationError as exc:
        log_event(logger, logging.ERROR, "configuration_error", context="cli", error=exc)
        return 1

    graph = build_agent_graph(settings=settings)
    if dump_graph:
        try:
            _maybe_dump_mermaid_png(graph, dump_graph)
        except Exception as exc:
            log_event(logger, logging.WARNING, "graph_dump_failed", output_path=dump_graph, error=exc)

    verbose = settings.verbose
    messages = []
    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in {"quit", "exit", "q"}:
                _write_line("Goodbye!")
                return 0

            state = {
                "user_input": user_input,
                "messages": messages,
            }
            response = graph.invoke(state)
            messages = response["messages"]

            if verbose:
                for message in response["messages"]:
                    _write_line(repr(message))
            else:
                for message in reversed(messages):
                    if isinstance(message, AIMessage):
                        _write_line(str(message.content))
                        break
        except KeyboardInterrupt:
            _write_line("")
            _write_line("Goodbye!")
            return 0
        except Exception as exc:
            log_event(logger, logging.ERROR, "cli_run_failed", error=exc)
            return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dump-graph",
        type=str,
        default=None,
        help="Optional PNG output path for the Mermaid graph dump.",
    )
    return parser


def main() -> int:
    maybe_reexec_with_utf8("src.cli", sys.argv[1:])
    parser = build_parser()
    args = parser.parse_args()
    return run_cli(dump_graph=args.dump_graph)


if __name__ == "__main__":
    raise SystemExit(main())
