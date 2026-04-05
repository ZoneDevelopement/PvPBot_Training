from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bot_training.data.preprocessing import Phase1Config, process_phase1_csv_file, process_phase1_csv_files


class Phase1PreprocessingTest(unittest.TestCase):
    def test_keeps_only_matches_passing_all_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "raw"
            output_file = tmp_path / "clean" / "phase1_clean_matches.csv"
            input_dir.mkdir(parents=True, exist_ok=True)

            rows = []
            for i in range(1005):
                rows.append(
                    {
                        "timestamp": str(i),
                        "playerName": "player_one",
                        "damageTaken": "0.0",
                        "damageDealt": "1.0" if i % 2 == 0 else "0.0",
                        "inputLmb": "True",
                        "inputSprint": "True" if i % 10 < 7 else "False",
                    }
                )
            for i in range(850):
                rows.append(
                    {
                        "timestamp": str(10_000 + i),
                        "playerName": "player_one",
                        "damageTaken": "0.0",
                        "damageDealt": "1.0",
                        "inputLmb": "True",
                        "inputSprint": "True",
                    }
                )
            for i in range(1_010):
                rows.append(
                    {
                        "timestamp": str(20_000 + i),
                        "playerName": "player_one",
                        "damageTaken": "0.05",
                        "damageDealt": "1.0",
                        "inputLmb": "True",
                        "inputSprint": "True",
                    }
                )

            with (input_dir / "matches.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["timestamp", "playerName", "damageTaken", "damageDealt", "inputLmb", "inputSprint"],
                )
                writer.writeheader()
                writer.writerows(rows)

            config = Phase1Config(
                chunksize=250,
                timestamp_gap=1_000,
                min_frames=1_000,
                max_damage_taken=40.0,
                min_attack_accuracy=0.40,
                min_sprint_uptime=0.60,
            )
            result = process_phase1_csv_files(input_dir, output_file, config)

            self.assertEqual(result.files_processed, 1)
            self.assertEqual(result.kept_matches, 1)
            self.assertGreaterEqual(result.candidate_matches, 3)
            self.assertTrue(output_file.exists())
            self.assertEqual(result.rejection_reasons.min_frames, 1)
            self.assertEqual(result.rejection_reasons.damage, 1)
            self.assertEqual(result.rejection_reasons.accuracy, 0)
            self.assertEqual(result.rejection_reasons.sprint, 0)

            with output_file.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                output_rows = list(reader)

            self.assertEqual({row["match_id"] for row in output_rows}, {"1"})
            self.assertEqual(len(output_rows), 1005)
            self.assertEqual(output_rows[0]["playerName"], "player_one")

    def test_alternating_players_do_not_create_one_row_matches_and_skip_empty_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_file = tmp_path / "raw" / "alternating.csv"
            output_file = tmp_path / "clean" / "alternating_clean.csv"
            input_file.parent.mkdir(parents=True, exist_ok=True)

            rows = []
            for i in range(120):
                rows.append(
                    {
                        "timestamp": str(1_000_000 + i),
                        "playerName": "alpha" if i % 2 == 0 else "beta",
                        "damageTaken": "0",
                        "damageDealt": "0",
                        "inputLmb": "False",
                        "inputSprint": "False",
                    }
                )

            with input_file.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["timestamp", "playerName", "damageTaken", "damageDealt", "inputLmb", "inputSprint"],
                )
                writer.writeheader()
                writer.writerows(rows)

            config = Phase1Config(chunksize=25, min_frames=1_000)
            result, _ = process_phase1_csv_file(
                input_file,
                output_file,
                config,
                append=False,
                start_match_id=1,
            )

            self.assertEqual(result.rows_read, 120)
            self.assertEqual(result.candidate_matches, 1)
            self.assertEqual(result.kept_matches, 0)
            self.assertEqual(result.rejection_reasons.min_frames, 1)
            self.assertEqual(result.rejection_reasons.accuracy, 1)
            self.assertEqual(result.rejection_reasons.sprint, 1)
            self.assertFalse(output_file.exists())


if __name__ == "__main__":
    unittest.main()


