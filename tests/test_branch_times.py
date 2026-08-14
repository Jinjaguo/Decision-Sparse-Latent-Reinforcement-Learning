import sys
import unittest
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from decision_sparse_rl.envs.branch_times import select_branch_times  # noqa: E402


class BranchTimeTest(unittest.TestCase):
    def test_selection_is_unique_and_deterministic(self):
        actions = np.zeros((20, 7), dtype=float)
        actions[:10, -1] = -1
        actions[10:, -1] = 1
        contacts = [3] * 8 + [7] * 12
        first = select_branch_times(actions, contacts)
        second = select_branch_times(actions.copy(), list(contacts))
        self.assertEqual(first, second)
        self.assertEqual(12, len({item["action_index"] for item in first}))
        self.assertTrue(all(0 <= item["action_index"] < 20 for item in first))

    def test_constant_gripper_uses_documented_fallback(self):
        actions = np.zeros((20, 7), dtype=float)
        actions[:, -1] = -1
        contacts = [3] * 8 + [7] * 12
        selected = select_branch_times(actions, contacts)
        gripper = next(item for item in selected if item["kind"] == "gripper_event_fallback")
        self.assertFalse(gripper["event_valid"])
        self.assertIn("constant", gripper["fallback_rule"])


if __name__ == "__main__":
    unittest.main()
