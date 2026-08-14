"""Verified LIBERO demonstration replay primitives for EXP1."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def numeric_demo_sort(names: Iterable[str]) -> List[str]:
    """Sort verified ``demo_<integer>`` episode names numerically."""

    parsed: List[Tuple[int, str]] = []
    for name in names:
        prefix, separator, suffix = name.rpartition("_")
        if not separator or prefix != "demo" or not suffix.isdigit():
            raise ValueError(f"unexpected episode name: {name!r}")
        parsed.append((int(suffix), name))
    return [name for _, name in sorted(parsed)]


def rewrite_episode_model_paths(
    xml_string: str,
    *,
    robosuite_package_root: Path,
    libero_assets_root: Path,
) -> Tuple[str, Dict[str, Any]]:
    """Resolve legacy absolute asset paths in an episode's stored MuJoCo XML.

    Public LIBERO files preserve XML created before the repository rename from
    ``chiliocosm``. Paths are mapped only through explicit source-tree anchors;
    every mapped target must exist before the XML is accepted.
    """

    root = ET.fromstring(xml_string)
    asset = root.find("asset")
    if asset is None:
        raise ValueError("stored model XML has no <asset> element")

    robosuite_package_root = robosuite_package_root.resolve()
    libero_assets_root = libero_assets_root.resolve()
    records: List[Dict[str, str]] = []
    unresolved: List[str] = []

    for element in list(asset.findall("mesh")) + list(asset.findall("texture")):
        old_path = element.get("file")
        if old_path is None:
            continue
        normalized = old_path.replace("\\", "/")
        target: Path
        source_anchor: str
        if "/robosuite/" in normalized:
            suffix = normalized.rsplit("/robosuite/", 1)[1]
            target = robosuite_package_root / Path(suffix)
            source_anchor = "robosuite"
        elif "/chiliocosm/assets/" in normalized:
            suffix = normalized.rsplit("/chiliocosm/assets/", 1)[1]
            target = libero_assets_root / Path(suffix)
            source_anchor = "chiliocosm/assets"
        elif "/libero/libero/assets/" in normalized:
            suffix = normalized.rsplit("/libero/libero/assets/", 1)[1]
            target = libero_assets_root / Path(suffix)
            source_anchor = "libero/libero/assets"
        else:
            candidate = Path(old_path)
            if candidate.is_file():
                continue
            unresolved.append(old_path)
            continue

        target = target.resolve()
        if not target.is_file():
            unresolved.append(f"{old_path} -> {target}")
            continue
        new_path = target.as_posix()
        element.set("file", new_path)
        records.append(
            {
                "element": element.tag,
                "name": element.get("name", ""),
                "old_path": old_path,
                "new_path": new_path,
                "source_anchor": source_anchor,
            }
        )

    if unresolved:
        raise FileNotFoundError(
            f"{len(unresolved)} stored XML asset paths could not be verified: {unresolved[:10]}"
        )
    return ET.tostring(root, encoding="unicode"), {
        "rewritten_path_count": len(records),
        "rewrites": records,
        "unresolved_path_count": 0,
    }


def summarize_replay_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize complete per-step replay rows without discarding raw curves."""

    if not rows:
        raise ValueError("cannot summarize an empty replay result")
    errors = np.asarray([row["state_l2_error"] for row in rows], dtype=np.float64)
    normalized_time = np.asarray([row["normalized_time"] for row in rows], dtype=np.float64)
    finite = np.isfinite(errors)
    if not np.all(finite):
        correlation = None
    elif errors.size < 2 or np.std(errors) == 0 or np.std(normalized_time) == 0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(normalized_time, errors)[0, 1])
    first = errors[normalized_time <= 0.25]
    last = errors[normalized_time >= 0.75]
    return {
        "comparison_count": int(errors.size),
        "all_finite": bool(np.all(finite)),
        "median_state_l2_error": float(np.median(errors)),
        "p95_state_l2_error": float(np.percentile(errors, 95)),
        "maximum_state_l2_error": float(np.max(errors)),
        "fraction_above_0_01": float(np.mean(errors > 0.01)),
        "normalized_time_error_pearson_r": correlation,
        "first_quartile_median_error": float(np.median(first)) if first.size else None,
        "last_quartile_median_error": float(np.median(last)) if last.size else None,
    }


def evaluate_replay_gate(
    *,
    summary: Dict[str, Any],
    episode_results: Sequence[Dict[str, Any]],
    selected_task_count: int,
    min_demos_per_task: int = 3,
    minimum_success_rate: float = 0.9,
    maximum_restore_error: float = 1e-10,
    maximum_p95_replay_error: float = 0.01,
) -> Dict[str, Any]:
    """Apply the predeclared pilot replay gate and return every subcriterion."""

    demos_by_task: Dict[str, int] = {}
    for result in episode_results:
        task_name = str(result["task"])
        demos_by_task[task_name] = demos_by_task.get(task_name, 0) + 1
    enough_demos = (
        len(demos_by_task) == selected_task_count
        and all(count >= min_demos_per_task for count in demos_by_task.values())
    )
    successes = [bool(result["final_success"]) for result in episode_results]
    success_rate = float(np.mean(successes)) if successes else 0.0
    restore_max = max(
        (float(result["repeat_restore_l2_error"]) for result in episode_results),
        default=float("inf"),
    )
    criteria = {
        "enough_demos_and_initial_conditions": enough_demos,
        "all_errors_finite": bool(summary["all_finite"]),
        "success_rate_at_least_minimum": success_rate >= minimum_success_rate,
        "repeat_restore_error_at_most_maximum": restore_max <= maximum_restore_error,
        "p95_replay_error_at_most_maximum": float(summary["p95_state_l2_error"])
        <= maximum_p95_replay_error,
        "source_indexing_verified": True,
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "observed": {
            "demos_by_task": demos_by_task,
            "success_rate": success_rate,
            "maximum_repeat_restore_l2_error": restore_max,
            "p95_replay_error": summary["p95_state_l2_error"],
        },
        "thresholds": {
            "min_demos_per_task": min_demos_per_task,
            "minimum_success_rate": minimum_success_rate,
            "maximum_restore_error": maximum_restore_error,
            "maximum_p95_replay_error": maximum_p95_replay_error,
        },
    }
