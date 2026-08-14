from pathlib import Path

import pytest

from decision_sparse_rl.envs.libero_replay import (
    evaluate_replay_gate,
    numeric_demo_sort,
    rewrite_episode_model_paths,
    summarize_replay_rows,
)


def test_numeric_demo_sort_rejects_unknown_layout():
    assert numeric_demo_sort(["demo_10", "demo_2", "demo_0"]) == ["demo_0", "demo_2", "demo_10"]
    with pytest.raises(ValueError):
        numeric_demo_sort(["episode_0"])


def test_rewrite_episode_model_paths_verifies_targets(tmp_path: Path):
    robosuite_root = tmp_path / "robosuite"
    libero_assets = tmp_path / "assets"
    robot_mesh = robosuite_root / "models" / "panda.stl"
    texture = libero_assets / "textures" / "table.png"
    robot_mesh.parent.mkdir(parents=True)
    texture.parent.mkdir(parents=True)
    robot_mesh.write_bytes(b"mesh")
    texture.write_bytes(b"png")
    xml = """<mujoco><asset>
    <mesh name="panda" file="/old/robosuite/models/panda.stl"/>
    <texture name="table" file="/old/chiliocosm/assets/textures/table.png"/>
    </asset></mujoco>"""
    rewritten, record = rewrite_episode_model_paths(
        xml,
        robosuite_package_root=robosuite_root,
        libero_assets_root=libero_assets,
    )
    assert record["rewritten_path_count"] == 2
    assert robot_mesh.resolve().as_posix() in rewritten
    assert texture.resolve().as_posix() in rewritten


def test_replay_summary_and_gate():
    rows = [
        {"state_l2_error": error, "normalized_time": time}
        for error, time in [(0.001, 0.0), (0.002, 0.5), (0.003, 1.0)]
    ]
    summary = summarize_replay_rows(rows)
    episodes = [
        {"task": task, "final_success": True, "repeat_restore_l2_error": 0.0}
        for task in ["a", "a", "a", "b", "b", "b", "c", "c", "c"]
    ]
    gate = evaluate_replay_gate(summary=summary, episode_results=episodes, selected_task_count=3)
    assert summary["median_state_l2_error"] == pytest.approx(0.002)
    assert gate["passed"] is True
