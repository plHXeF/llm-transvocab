import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from prefetch import PrefetchKey, PrefetchManager


class PrefetchManagerTests(unittest.TestCase):
    def setUp(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.manager = PrefetchManager(self.executor)
        self.key = PrefetchKey("card", 1, "batch", "prompt-v1")

    def tearDown(self):
        self.manager.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)

    def test_duplicate_submission_reuses_future(self):
        calls = []

        def work():
            calls.append(1)
            return "ready"

        first = self.manager.submit(self.key, work)
        second = self.manager.submit(self.key, work)
        self.assertIs(first, second)
        self.assertEqual(self.manager.consume(self.key), "ready")
        self.assertEqual(calls, [1])

    def test_revision_change_makes_old_result_stale(self):
        release = threading.Event()

        def slow():
            release.wait(timeout=1)
            return "old"

        self.manager.submit(self.key, slow)
        new_key = PrefetchKey("card", 2, "batch", "prompt-v1")
        new_future = self.manager.submit(new_key, lambda: "new")
        release.set()
        self.assertEqual(new_future.result(timeout=2), "new")
        with self.assertRaises(KeyError):
            self.manager.consume(self.key)
        self.assertEqual(self.manager.consume(new_key), "new")

    def test_worker_error_is_not_replaced(self):
        def fail():
            raise RuntimeError("boom")

        self.manager.submit(self.key, fail)
        with self.assertRaisesRegex(RuntimeError, "boom"):
            self.manager.consume(self.key)
        self.assertIsNone(self.manager.key)


if __name__ == "__main__":
    unittest.main()
