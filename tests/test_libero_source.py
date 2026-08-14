from pathlib import Path
import hashlib
import json
from collections import namedtuple
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from decision_sparse_rl.envs.libero_source import (  # noqa: E402
    enumerate_registered_benchmarks,
    write_libero_config,
)


Task = namedtuple(
    "Task", "name language problem problem_folder bddl_file init_states_file"
)


class WorkingBenchmark:
    def __init__(self) -> None:
        self.task = Task("task_name", "task language", "Problem", "suite", "task.bddl", "task.init")

    def get_num_tasks(self) -> int:
        return 1

    def get_task(self, task_id: int) -> Task:
        return self.task

    def get_task_bddl_file_path(self, task_id: int) -> str:
        return "suite/task.bddl"

    def get_task_demonstration(self, task_id: int) -> str:
        return "suite/task_demo.hdf5"


class BrokenBenchmark:
    def __init__(self) -> None:
        raise KeyError("missing_suite_map")


class FakeBenchmarkModule:
    @staticmethod
    def get_benchmark_dict():
        return {"working": WorkingBenchmark, "broken": BrokenBenchmark}


class LiberoEnumerationTest(unittest.TestCase):
    def test_enumerates_runtime_fields_and_preserves_registration_error(self) -> None:
        suites, errors = enumerate_registered_benchmarks(FakeBenchmarkModule)
        self.assertEqual(suites["working"]["task_count"], 1)
        self.assertEqual(suites["working"]["tasks"][0]["language"], "task language")
        self.assertEqual(errors["broken"]["exception_type"], "KeyError")

    def test_writes_verified_config_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package_root = root / "source" / "libero" / "libero"
            for name in ("bddl_files", "init_files", "assets"):
                (package_root / name).mkdir(parents=True, exist_ok=True)
            config_file = write_libero_config(root / "config", root / "source", root / "data")
            config = json.loads(config_file.read_text(encoding="utf-8"))
            self.assertEqual(Path(config["benchmark_root"]), package_root.resolve())
            self.assertEqual(Path(config["datasets"]), (root / "data").resolve())

    def test_frozen_selection_matches_generated_manifest(self) -> None:
        manifests = REPOSITORY_ROOT / "experiments" / "exp1_decision_sparsity" / "manifests"
        source_path = manifests / "tasks.json"
        selection = json.loads(
            (manifests / "selected_tasks_pilot.json").read_text(encoding="utf-8")
        )
        source = json.loads(source_path.read_text(encoding="utf-8"))
        self.assertEqual(
            selection["source_manifest_sha256"],
            hashlib.sha256(source_path.read_bytes()).hexdigest().upper(),
        )
        for selected in selection["tasks"]:
            runtime_task = source["suites"][selected["suite"]]["tasks"][selected["task_id"]]
            for field in ("suite", "task_id", "name", "language", "demonstration_relative_path"):
                self.assertEqual(selected[field], runtime_task[field])


if __name__ == "__main__":
    unittest.main()
