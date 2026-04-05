from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bot_training.data.preprocessing import Phase1Result, RejectionBreakdown
from bot_training.data.threshold_sweep import (
    ScoreWeights,
    SweepRunResult,
    compute_quality_score,
    parse_float_grid,
    parse_int_grid,
    parse_score_weights,
    sample_csv_files,
    to_report_rows,
)


class Phase1ThresholdSweepTest(unittest.TestCase):
    def test_parse_numeric_grid(self) -> None:
        self.assertEqual(parse_int_grid("400, 700,1000"), [400, 700, 1000])
        self.assertEqual(parse_float_grid("0.2,0.4"), [0.2, 0.4])

    def test_parse_score_weights(self) -> None:
        weights = parse_score_weights("0.1,0.2,0.3,0.4")
        self.assertEqual(weights.min_frames, 0.1)
        self.assertEqual(weights.damage, 0.2)
        self.assertEqual(weights.accuracy, 0.3)
        self.assertEqual(weights.sprint, 0.4)

    def test_sample_csv_files_deterministic_fraction(self) -> None:
        files = [Path(f"/tmp/file_{i}.csv") for i in range(20)]
        sample_a = sample_csv_files(files, 0.1, 123)
        sample_b = sample_csv_files(files, 0.1, 123)
        self.assertEqual(sample_a, sample_b)
        self.assertEqual(len(sample_a), 2)

    def test_compute_quality_score(self) -> None:
        result = Phase1Result(
            candidate_matches=100,
            kept_matches=25,
            rejected_matches=75,
            rejection_reasons=RejectionBreakdown(min_frames=10, damage=20, accuracy=30, sprint=40),
        )
        keep_rate, score = compute_quality_score(result, ScoreWeights(0.25, 0.25, 0.25, 0.25))
        self.assertAlmostEqual(keep_rate, 0.25)
        self.assertAlmostEqual(score, 0.0)

    def test_report_rows_include_expected_fields(self) -> None:
        run = SweepRunResult(
            min_frames=700,
            max_damage_taken=50.0,
            min_attack_accuracy=0.3,
            min_sprint_uptime=0.45,
            result=Phase1Result(
                candidate_matches=10,
                kept_matches=4,
                rejected_matches=6,
                rejection_reasons=RejectionBreakdown(min_frames=2, damage=3, accuracy=1, sprint=4),
            ),
            keep_rate=0.4,
            quality_score=0.1,
            rank=1,
        )
        rows = to_report_rows([run])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["rank"], "1")
        self.assertEqual(row["min_frames"], "700")
        self.assertEqual(row["rej_damage"], "3")


if __name__ == "__main__":
    unittest.main()


