from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from decision_sparse_rl.metrics.exp15 import monotone_window, recovery_metrics, standardized_distance, weighted_chunk


def test_standardized_distance_and_zero_scale():
    d=standardized_distance([0,0],np.asarray([[0,0],[2,0]],float),[0,2])
    assert np.allclose(d,[0,2])


def test_weighted_chunk_preserves_exact_gripper_sign():
    chunks=np.zeros((2,3,7));chunks[0,:,0]=1;chunks[1,:,0]=3;chunks[0,:,6]=-1;chunks[1,:,6]=1
    out=weighted_chunk(chunks,[1,3])
    assert np.allclose(out[:,0],1.5) and set(out[:,6])=={-1}


def test_weighted_chunk_rejects_shape():
    with pytest.raises(ValueError):weighted_chunk(np.zeros((2,3,6)),[1,2])


def test_monotone_window_never_moves_backward():
    selected=monotone_window(["a"]*4+["b"],[0,5,10,30,7],"a",6,20)
    assert selected.tolist()==[2]


def test_recovery_metrics_separates_demand_and_availability():
    rows=[
        {"branch_id":"a","route":"default","success":False,"safety_stop":False},
        {"branch_id":"a","route":"rescue","success":True,"safety_stop":False},
        {"branch_id":"b","route":"default","success":True,"safety_stop":False},
        {"branch_id":"b","route":"rescue","success":False,"safety_stop":False},
    ]
    result=recovery_metrics(rows,"default")
    assert result=={"group_count":2,"safe_candidate_availability":1.0,"decision_demand_rate":.5,"demand_recovery_rate":1.0}


def test_exp22_routes_are_target_independent_and_structurally_diverse():
    from scripts.exp15.run_recovery_stage import EXP22_MODES, EXP22_ROUTES

    assert EXP22_ROUTES[0]["route"] == "D_physical_chunk"
    assert len(EXP22_ROUTES) == 9
    assert {mode.get("selection", "distance") for mode in EXP22_MODES.values()} >= {
        "distance",
        "goal_effect",
        "progress",
        "response",
    }
    adaptive = [route for route in EXP22_ROUTES if "modes" in route]
    assert len(adaptive) >= 4
    assert all(route["max_steps"] <= 280 for route in EXP22_ROUTES)
    assert all("target_future" not in route for route in EXP22_ROUTES)
