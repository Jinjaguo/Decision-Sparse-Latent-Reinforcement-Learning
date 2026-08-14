from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from decision_sparse_rl.utils.environment_audit import parse_pinned_requirement  # noqa: E402


class RequirementParsingTest(unittest.TestCase):
    def test_reads_exact_pin_with_whitespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requirements.txt"
            path.write_text(" numpy==1.22.4 \n robosuite==1.4.0 # expected\n", encoding="utf-8")
            self.assertEqual(parse_pinned_requirement(path, "robosuite"), "1.4.0")

    def test_missing_pin_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requirements.txt"
            path.write_text("robosuite>=1.4\n", encoding="utf-8")
            self.assertIsNone(parse_pinned_requirement(path, "robosuite"))


if __name__ == "__main__":
    unittest.main()
