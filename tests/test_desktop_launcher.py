import unittest
from unittest.mock import MagicMock, patch

from pathlib import Path

from desktop_launcher import available_port, run_streamlit, streamlit_options


class DesktopLauncherTests(unittest.TestCase):
    def test_available_port_can_be_bound_on_loopback(self):
        listener = MagicMock()
        listener.__enter__.return_value = listener
        listener.getsockname.return_value = ("127.0.0.1", 49152)
        with patch("desktop_launcher.socket.socket", return_value=listener):
            port = available_port()

        listener.bind.assert_called_once_with(("127.0.0.1", 0))
        self.assertEqual(port, 49152)

    def test_streamlit_options_are_local_headless_and_without_watcher(self):
        options = streamlit_options(49152)
        self.assertFalse(options["global.developmentMode"])
        self.assertEqual(options["server.address"], "127.0.0.1")
        self.assertEqual(options["server.port"], 49152)
        self.assertTrue(options["server.headless"])
        self.assertEqual(options["server.fileWatcherType"], "none")
        self.assertFalse(options["browser.gatherUsageStats"])
        self.assertEqual(options["logger.level"], "info")

    def test_options_are_loaded_before_streamlit_server_runs(self):
        options = streamlit_options(49152)
        with patch("desktop_launcher.bootstrap.load_config_options") as load:
            with patch("desktop_launcher.bootstrap.run") as run:
                calls = MagicMock()
                calls.attach_mock(load, "load")
                calls.attach_mock(run, "run")
                run_streamlit(Path("app.py"), options)

        load.assert_called_once_with(flag_options=options)
        run.assert_called_once_with("app.py", False, [], options)
        self.assertEqual(
            [item[0] for item in calls.mock_calls],
            ["load", "run"],
        )


if __name__ == "__main__":
    unittest.main()
