from datetime import datetime, timedelta, timezone
import unittest

from domain import Progress
from learning_charts import baseline_figure, forgetting_figure
from scheduler import retention_for


class LearningChartsTests(unittest.TestCase):
    def test_forgetting_matches_scheduler_and_is_monotonic(self):
        now = datetime(2026, 10, 8, tzinfo=timezone.utc)
        progress = Progress(
            card_id="test", attempts=3, stability_days=4,
            last_reviewed_at=now - timedelta(days=2),
        )
        for days in (7, 30, 90):
            figure = forgetting_figure(progress, days, now=now)
            curve = figure.data[0]
            self.assertEqual(curve.x[0], 0)
            self.assertEqual(curve.x[-1], days)
            for offset, value in zip(curve.x, curve.y):
                self.assertAlmostEqual(value, 100 * retention_for(
                    progress, now=now + timedelta(days=offset),
                ))
                self.assertTrue(0 <= value <= 100)
            self.assertTrue(all(a >= b for a, b in zip(curve.y, curve.y[1:])))
            self.assertEqual(figure.data[1].y[0], curve.y[0])

    def test_missing_and_future_review_times_follow_scheduler(self):
        now = datetime(2026, 10, 8, tzinfo=timezone.utc)
        empty = forgetting_figure(Progress(card_id="empty"), 7, now=now)
        self.assertTrue(all(value == 0 for value in empty.data[0].y))
        future = forgetting_figure(Progress(
            card_id="future", attempts=1, last_reviewed_at=now + timedelta(days=1),
        ), 7, now=now)
        self.assertEqual(future.data[0].y[0], 100)
        self.assertTrue(all(0 <= value <= 100 for value in future.data[0].y))

    def test_baseline_axes_reference_and_hover_details(self):
        figure = baseline_figure([{
            "word": "appeal", "pos": "n", "meaning": "呼吁",
            "reviewed_at": "2026-10-08T08:00:00Z",
            "expected_performance": 0.7, "target_performance": 0.9,
        }])
        self.assertEqual(list(figure.data[0].x), [70])
        self.assertEqual(list(figure.data[0].y), [90])
        self.assertIn("appeal · n · 呼吁", figure.data[0].customdata[0])
        self.assertEqual(list(figure.layout.xaxis.range), [0, 100])
        self.assertEqual(list(figure.layout.yaxis.range), [0, 100])
        self.assertEqual(list(figure.data[1].x), list(figure.data[1].y))
