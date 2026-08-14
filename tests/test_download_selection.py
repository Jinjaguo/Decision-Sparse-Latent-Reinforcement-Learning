from pathlib import Path
import json
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts" / "exp1"))

from download_selected_dataset import selected_suite  # noqa: E402


class SelectedSuiteTest(unittest.TestCase):
    def test_accepts_one_declared_suite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(
                json.dumps(
                    {
                        "suite_download_scope": "suite_a",
                        "tasks": [{"suite": "suite_a"}, {"suite": "suite_a"}],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(selected_suite(path), "suite_a")

    def test_rejects_mixed_or_mismatched_suites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(
                json.dumps(
                    {
                        "suite_download_scope": "suite_a",
                        "tasks": [{"suite": "suite_a"}, {"suite": "suite_b"}],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                selected_suite(path)


if __name__ == "__main__":
    unittest.main()
