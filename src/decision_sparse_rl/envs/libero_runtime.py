"""Shared exact-runtime loading helpers for the frozen LIBERO pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import numpy as np

from .libero_replay import numeric_demo_sort, rewrite_episode_model_paths
from .libero_source import import_benchmark_from_source, import_module_from_source, write_libero_config


def environment_kwargs(bddl_path: Path) -> Dict[str, Any]:
    return {
        "bddl_file_name": str(bddl_path.resolve()),
        "robots": ["Panda"],
        "controller": "OSC_POSE",
        "initialization_noise": None,
        "use_camera_obs": False,
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "ignore_done": True,
        "reward_shaping": True,
        "control_freq": 20,
    }


def task_source_record(task_manifest: Dict[str, Any], suite: str, task_id: int) -> Dict[str, Any]:
    for task in task_manifest["suites"][suite]["tasks"]:
        if int(task["task_id"]) == int(task_id):
            return task
    raise KeyError(f"task absent from source manifest: {suite}:{task_id}")


def bootstrap_runtime(libero_root: Path, dataset_root: Path, config_directory: Path) -> Tuple[Any, Path, Path]:
    write_libero_config(config_directory, libero_root, dataset_root)
    import_benchmark_from_source(libero_root, config_directory)
    wrapper = import_module_from_source(libero_root, config_directory, "libero.libero.envs.env_wrapper")
    import robosuite

    return wrapper, Path(robosuite.__file__).resolve().parent, libero_root / "libero/libero/assets"


def load_episode(
    env: Any,
    *,
    dataset_path: Path,
    episode_index: int,
    robosuite_package_root: Path,
    libero_assets_root: Path,
) -> Dict[str, Any]:
    with h5py.File(dataset_path, "r") as handle:
        names = numeric_demo_sort(handle["data"].keys())
        name = names[episode_index]
        episode = handle[f"data/{name}"]
        states = np.asarray(episode["states"], dtype=np.float64)
        actions = np.asarray(episode["actions"], dtype=np.float64)
        xml, rewrites = rewrite_episode_model_paths(
            episode.attrs["model_file"],
            robosuite_package_root=robosuite_package_root,
            libero_assets_root=libero_assets_root,
        )
    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.set_init_state(states[0])
    # set_init_state updates MuJoCo and observations but does not synchronize the
    # already-constructed robosuite controller caches. Establish one explicit,
    # reproducible pre-policy controller boundary from the verified physical state.
    for robot in env.robots:
        robot.controller.update(force=True)
        robot.controller.reset_goal()
        robot.controller.new_update = True
    return {"episode": name, "states": states, "actions": actions, "xml": xml, "path_rewrites": rewrites}


def load_selection(selection: Path, task_manifest: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return json.loads(selection.read_text(encoding="utf-8")), json.loads(task_manifest.read_text(encoding="utf-8"))
