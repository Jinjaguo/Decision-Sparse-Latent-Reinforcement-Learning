from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from decision_sparse_rl.logging.run_directory import (  # noqa: E402
    REQUIRED_RUN_FILES,
    create_run_directory,
    write_run_record,
)


class RunDirectoryTest(unittest.TestCase):
    def test_refuses_to_overwrite_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_run_directory(root, "run_a")
            with self.assertRaises(FileExistsError):
                create_run_directory(root, "run_a")

    def test_writes_required_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = create_run_directory(Path(directory), "run_a")
            write_run_record(
                run_dir,
                config={},
                command="test",
                environment={},
                git_state={},
                metrics={},
            )
            self.assertTrue((run_dir / "artifacts").is_dir())
            self.assertTrue(all((run_dir / name).is_file() for name in REQUIRED_RUN_FILES))

    def test_rejects_nested_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                create_run_directory(Path(directory), "nested/run")


if __name__ == "__main__":
    unittest.main()
