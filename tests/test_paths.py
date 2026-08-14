from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from decision_sparse_rl.paths import project_root  # noqa: E402


class ProjectRootTest(unittest.TestCase):
    def test_project_root_contains_project_specification(self) -> None:
        self.assertEqual(project_root(), REPOSITORY_ROOT)
        self.assertTrue((project_root() / "PROJECT.md").is_file())


if __name__ == "__main__":
    unittest.main()
