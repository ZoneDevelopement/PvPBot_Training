from pathlib import Path
import unittest


class ScaffoldSmokeTest(unittest.TestCase):
    def test_expected_directories_exist(self) -> None:
        root = Path(__file__).resolve().parents[1]
        expected = [
            root / "src" / "bot_training",
            root / "data" / "interim",
            root / "data" / "processed",
            root / "data" / "splits",
            root / "models" / "checkpoints",
            root / "models" / "exports",
            root / "reports" / "metrics",
            root / "reports" / "figures",
        ]
        for path in expected:
            self.assertTrue(path.exists(), f"Missing scaffold path: {path}")


if __name__ == "__main__":
    unittest.main()

