from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from src import service_manager


class ServiceManagerTest(unittest.TestCase):
    def test_start_web_services_rejects_occupied_ports(self) -> None:
        with patch("src.service_manager._is_port_open", side_effect=[True, False]):
            result = service_manager._start_web_services()

        self.assertEqual(result, 1)

    def test_start_web_services_writes_runtime_state(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "output" / "runtime"
            runtime_dir.mkdir(parents=True)
            state_path = runtime_dir / "web_services_state.json"
            fastapi_log = runtime_dir / "fastapi.log"
            streamlit_log = runtime_dir / "streamlit.log"

            fake_fastapi_proc = SimpleNamespace(pid=101)
            fake_streamlit_proc = SimpleNamespace(pid=202)

            with patch("src.service_manager.get_project_root_path", return_value=root), patch(
                "src.service_manager.get_service_state_path",
                return_value=state_path,
            ), patch(
                "src.service_manager.get_runtime_log_path",
                side_effect=[fastapi_log, streamlit_log],
            ), patch("src.service_manager._is_port_open", return_value=False), patch(
                "src.service_manager._start_background_process",
                side_effect=[fake_fastapi_proc, fake_streamlit_proc],
            ), patch("src.service_manager._wait_for_port_open", return_value=True), patch(
                "src.service_manager._get_process_create_time",
                side_effect=[1.1, 2.2],
            ):
                result = service_manager._start_web_services()

            self.assertEqual(result, 0)
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["fastapi_pid"], 101)
            self.assertEqual(payload["streamlit_pid"], 202)
            self.assertEqual(payload["fastapi_log"], str(fastapi_log))
            self.assertEqual(payload["streamlit_log"], str(streamlit_log))
            self.assertEqual(state_path.parent.name, "runtime")

    def test_stop_web_services_falls_back_to_cmdline_discovery(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "output" / "runtime"
            runtime_dir.mkdir(parents=True)
            state_path = runtime_dir / "web_services_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "fastapi_pid": 1,
                        "fastapi_create_time": 11.0,
                        "streamlit_pid": 2,
                        "streamlit_create_time": 22.0,
                    }
                ),
                encoding="utf-8",
            )

            with patch("src.service_manager.get_service_state_path", return_value=state_path), patch(
                "src.service_manager._is_process_alive",
                return_value=False,
            ), patch(
                "src.service_manager._find_process_pid_by_tokens",
                side_effect=[111, 222, None, None],
            ), patch(
                "src.service_manager._get_process_create_time",
                side_effect=[111.0, 222.0],
            ), patch(
                "src.service_manager._terminate_process_tree",
                return_value=True,
            ) as mock_terminate:
                result = service_manager._stop_web_services()

            self.assertEqual(result, 0)
            terminate_pids = [call.kwargs["pid"] for call in mock_terminate.call_args_list]
            self.assertEqual(terminate_pids, [111, 222])
            self.assertFalse(state_path.exists())


if __name__ == "__main__":
    unittest.main()
