"""Generic HDF5 schema inspection for EXP1 demonstration files."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable


def json_value(value: Any) -> Any:
    """Convert HDF5 / NumPy attribute values to JSON-safe structures."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def audit_hdf5(path: Path) -> Dict[str, Any]:
    """Inspect every group, dataset, and attribute without assuming a root layout."""

    import h5py

    path = path.resolve()
    groups: Dict[str, Any] = {}
    datasets: Dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        groups["/"] = {
            "attributes": {key: json_value(value) for key, value in handle.attrs.items()},
            "members": sorted(handle.keys()),
        }

        def visitor(name: str, obj: Any) -> None:
            absolute_name = f"/{name}"
            attributes = {key: json_value(value) for key, value in obj.attrs.items()}
            if isinstance(obj, h5py.Group):
                groups[absolute_name] = {
                    "attributes": attributes,
                    "members": sorted(obj.keys()),
                }
            elif isinstance(obj, h5py.Dataset):
                datasets[absolute_name] = {
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "chunks": list(obj.chunks) if obj.chunks else None,
                    "compression": obj.compression,
                    "compression_opts": json_value(obj.compression_opts),
                    "attributes": attributes,
                }

        handle.visititems(visitor)

    parents: Dict[str, Dict[str, str]] = {}
    for dataset_path in datasets:
        parent, _, basename = dataset_path.rpartition("/")
        parents.setdefault(parent or "/", {})[basename] = dataset_path
    episode_groups = []
    for parent, direct_datasets in sorted(parents.items()):
        if "states" in direct_datasets and "actions" in direct_datasets:
            state_shape = datasets[direct_datasets["states"]]["shape"]
            action_shape = datasets[direct_datasets["actions"]]["shape"]
            episode_groups.append(
                {
                    "group": parent,
                    "states_path": direct_datasets["states"],
                    "actions_path": direct_datasets["actions"],
                    "states_length": state_shape[0] if state_shape else None,
                    "actions_length": action_shape[0] if action_shape else None,
                    "direct_datasets": direct_datasets,
                }
            )
    semantic_basenames = {
        "states": "states",
        "actions": "actions",
        "joint_states": "joint_states",
        "gripper_states": "gripper_states",
        "ee_states": "ee_states",
    }
    semantic_matches = {
        key: sorted(
            dataset_path
            for dataset_path in datasets
            if dataset_path.rpartition("/")[2] == basename
        )
        for key, basename in semantic_basenames.items()
    }
    return {
        "file": {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
        "group_count": len(groups),
        "dataset_count": len(datasets),
        "groups": groups,
        "datasets": datasets,
        "episode_detection": {
            "rule": "parent groups containing direct datasets named states and actions",
            "episode_count": len(episode_groups),
            "episodes": episode_groups,
        },
        "semantic_dataset_matches": semantic_matches,
        "presence": {key: bool(matches) for key, matches in semantic_matches.items()},
    }
