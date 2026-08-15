"""Run reference-free feedback candidates from corrected-D branch states."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))

from decision_sparse_rl.envs.libero_runtime import bootstrap_runtime,environment_kwargs,load_episode,load_selection,task_source_record
from decision_sparse_rl.metrics.exp15 import monotone_window,standardized_distance,weighted_chunk
from scripts.exp3.run_criticality import restore_d
from scripts.exp7.contact_geometry import load_schema
import scripts.exp11.run_replacement_stage as engine

TASKS=engine.TASKS
ROUTES=[
    {"route":"D_physical_chunk","view":"physical","k":1,"replan":10,"monotone":False,"retarget":0.0,"aggregate":"nearest"},
    {"route":"R1_object_feedback","view":"object","k":1,"replan":1,"monotone":False,"retarget":0.0,"aggregate":"nearest"},
    {"route":"R1_contact_feedback","view":"full","k":3,"replan":1,"monotone":False,"retarget":0.0,"aggregate":"weighted"},
    {"route":"R2_monotone_feedback","view":"full","k":1,"replan":1,"monotone":True,"retarget":0.0,"aggregate":"nearest"},
    {"route":"R2_monotone_chunk","view":"full","k":1,"replan":10,"monotone":True,"retarget":0.0,"aggregate":"nearest"},
    {"route":"R3_weighted_feedback","view":"full","k":5,"replan":1,"monotone":False,"retarget":0.0,"aggregate":"weighted"},
    {"route":"R4_retarget_low","view":"object","k":1,"replan":1,"monotone":True,"retarget":0.25,"aggregate":"nearest"},
    {"route":"R4_retarget_high","view":"object","k":1,"replan":1,"monotone":True,"retarget":0.50,"aggregate":"nearest"},
    {"route":"R7_conservative","view":"full","k":7,"replan":1,"monotone":False,"retarget":0.0,"aggregate":"median"},
]
EXP16_ROUTES=[
    {"route":"D_physical_chunk","view":"physical","k":1,"replan":10,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"nearest","smooth":0.0},
    {"route":"S2_task_weighted","view":"full","k":5,"replan":1,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"weighted","smooth":0.0},
    {"route":"S3_progress_1","view":"full","k":3,"replan":1,"monotone":True,"advance":1,"retarget":0.0,"aggregate":"weighted","smooth":0.0},
    {"route":"S3_progress_3","view":"full","k":3,"replan":1,"monotone":True,"advance":3,"retarget":0.0,"aggregate":"weighted","smooth":0.0},
    {"route":"S4_persistent_chunk","view":"full","k":1,"replan":5,"monotone":True,"advance":5,"retarget":0.0,"aggregate":"nearest","smooth":0.0},
    {"route":"S5_conservative_median","view":"full","k":7,"replan":1,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"median","smooth":0.0},
    {"route":"S5_medoid","view":"full","k":7,"replan":1,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"medoid","smooth":0.0},
    {"route":"S7_smooth_weighted","view":"full","k":5,"replan":1,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"weighted","smooth":0.5},
    {"route":"S8_contact_smooth","view":"full","k":3,"replan":2,"monotone":True,"advance":2,"retarget":0.0,"aggregate":"weighted","smooth":0.35},
]
EXP17_ROUTES=[
    {"route":"D_physical_chunk","view":"physical","k":1,"replan":10,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"nearest","smooth":0.0},
    {"route":"H1_weighted_k3","view":"full","k":3,"replan":1,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"weighted","smooth":0.0},
    {"route":"H1_weighted_k9","view":"full","k":9,"replan":1,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"weighted","smooth":0.0},
    {"route":"H2_median_k9","view":"full","k":9,"replan":1,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"median","smooth":0.0},
    {"route":"H2_medoid_k9","view":"full","k":9,"replan":1,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"medoid","smooth":0.0},
    {"route":"H3_smooth_low","view":"full","k":5,"replan":1,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"weighted","smooth":0.25},
    {"route":"H3_smooth_high","view":"full","k":5,"replan":1,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"weighted","smooth":0.75},
    {"route":"H4_progress_persistent","view":"full","k":3,"replan":2,"monotone":True,"advance":2,"retarget":0.0,"aggregate":"weighted","smooth":0.25},
    {"route":"H5_short_chunk","view":"full","k":1,"replan":2,"monotone":True,"advance":2,"retarget":0.0,"aggregate":"nearest","smooth":0.0},
]
EXP17_BASE={x["route"]:x for x in EXP17_ROUTES}
EXP21_ROUTES=[
    {"route":"D_physical_chunk","stages":["D_physical_chunk"],"switch_steps":[],"max_steps":140},
    {"route":"Q0_fixed_k9_200","stages":["H1_weighted_k9"],"switch_steps":[],"max_steps":200},
    {"route":"Q1_default_to_k9","stages":["D_physical_chunk","H1_weighted_k9"],"switch_steps":[70],"max_steps":200},
    {"route":"Q2_k9_to_smooth","stages":["H1_weighted_k9","H3_smooth_low"],"switch_steps":[70],"max_steps":200},
    {"route":"Q3_smooth_to_k9","stages":["H3_smooth_low","H1_weighted_k9"],"switch_steps":[70],"max_steps":200},
    {"route":"Q4_median_to_k9","stages":["H2_median_k9","H1_weighted_k9"],"switch_steps":[70],"max_steps":200},
    {"route":"Q5_default_to_smooth","stages":["D_physical_chunk","H3_smooth_low"],"switch_steps":[70],"max_steps":200},
    {"route":"Q6_three_stage","stages":["D_physical_chunk","H1_weighted_k9","H3_smooth_low"],"switch_steps":[50,110],"max_steps":200},
    {"route":"Q7_k9_to_medoid","stages":["H1_weighted_k9","H2_medoid_k9"],"switch_steps":[90],"max_steps":200}
]
EXP22_MODES={
    "C0_goal_consequence":{"view":"full","k":7,"replan":1,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"weighted","smooth":0.15,"selection":"goal_effect","search_k":48,"consequence_weight":0.75},
    "C1_progress_consequence":{"view":"full","k":5,"replan":1,"monotone":True,"advance":2,"retarget":0.0,"aggregate":"weighted","smooth":0.10,"selection":"progress","search_k":36,"consequence_weight":0.80},
    "C2_response_alignment":{"view":"full","k":5,"replan":1,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"medoid","smooth":0.0,"selection":"response","search_k":48,"consequence_weight":0.65},
    "C3_short_persistent":{"view":"full","k":1,"replan":2,"monotone":True,"advance":2,"retarget":0.0,"aggregate":"nearest","smooth":0.0,"selection":"distance"},
    "C4_smooth_low":{"view":"full","k":5,"replan":1,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"weighted","smooth":0.25,"selection":"distance"},
    "C5_conservative_medoid":{"view":"full","k":9,"replan":1,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"medoid","smooth":0.0,"selection":"distance"},
    "C6_goal_retarget":{"view":"full","k":5,"replan":1,"monotone":True,"advance":1,"retarget":0.20,"aggregate":"weighted","smooth":0.10,"selection":"goal_effect","search_k":48,"consequence_weight":0.70},
    "C7_physical_default":{"view":"physical","k":1,"replan":10,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"nearest","smooth":0.0,"selection":"distance"},
}
EXP22_ROUTES=[
    {"route":"D_physical_chunk","view":"physical","k":1,"replan":10,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"nearest","smooth":0.0,"max_steps":140},
    {"route":"E0_fixed_goal_consequence",**EXP22_MODES["C0_goal_consequence"],"max_steps":220},
    {"route":"E1_fixed_progress_consequence",**EXP22_MODES["C1_progress_consequence"],"max_steps":220},
    {"route":"E2_fixed_response_alignment",**EXP22_MODES["C2_response_alignment"],"max_steps":220},
    {"route":"F0_stall_goal_progress_smooth","modes":["C0_goal_consequence","C1_progress_consequence","C4_smooth_low"],"stall_window":24,"minimum_progress_gain":0.025,"minimum_dwell":30,"max_steps":240},
    {"route":"F1_stall_smooth_response_progress","modes":["C4_smooth_low","C2_response_alignment","C1_progress_consequence"],"stall_window":20,"minimum_progress_gain":0.020,"minimum_dwell":25,"max_steps":260},
    {"route":"F2_stall_progress_persistent_goal","modes":["C1_progress_consequence","C3_short_persistent","C0_goal_consequence"],"stall_window":28,"minimum_progress_gain":0.030,"minimum_dwell":35,"max_steps":260},
    {"route":"F3_diverse_periodic_cycle","modes":["C4_smooth_low","C0_goal_consequence","C2_response_alignment","C5_conservative_medoid"],"maximum_dwell":55,"minimum_dwell":30,"max_steps":280},
    {"route":"F4_task_specialized_fsm","modes":["C0_goal_consequence","C1_progress_consequence","C4_smooth_low"],"task_modes":{"open_the_middle_drawer_of_the_cabinet":["C1_progress_consequence","C3_short_persistent","C0_goal_consequence"],"put_the_bowl_on_the_plate":["C4_smooth_low","C0_goal_consequence"],"turn_on_the_stove":["C4_smooth_low","C2_response_alignment","C0_goal_consequence","C1_progress_consequence"]},"stall_window":22,"minimum_progress_gain":0.020,"minimum_dwell":25,"maximum_dwell":70,"max_steps":280},
]
EXP23_ROUTES=[
    {"route":"D_physical_chunk","view":"physical","k":1,"replan":10,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"nearest","smooth":0.0,"max_steps":140},
    {"route":"P0_guarded_diverse","modes":["C4_smooth_low","C0_goal_consequence","C2_response_alignment","C5_conservative_medoid"],"maximum_dwell":55,"minimum_dwell":25,"max_steps":300,"force_guard":"retract","guard_fraction":0.70,"guard_gain":1.0},
    {"route":"P1_guarded_goal","modes":["C0_goal_consequence","C4_smooth_low","C2_response_alignment"],"stall_window":22,"minimum_progress_gain":0.02,"minimum_dwell":25,"maximum_dwell":85,"max_steps":280,"force_guard":"retract","guard_fraction":0.70,"guard_gain":1.0},
    {"route":"P2_guarded_smooth_response","modes":["C4_smooth_low","C2_response_alignment","C1_progress_consequence","C0_goal_consequence"],"stall_window":20,"minimum_progress_gain":0.02,"minimum_dwell":22,"maximum_dwell":70,"max_steps":300,"force_guard":"retract","guard_fraction":0.75,"guard_gain":0.75},
    {"route":"P3_phase_risk_arbitration","modes":["C4_smooth_low","C0_goal_consequence","C2_response_alignment","C5_conservative_medoid"],"progress_bands":{"turn_on_the_stove":[{"minimum":0.65,"modes":["C4_smooth_low","C2_response_alignment","C1_progress_consequence"]},{"minimum":0.45,"modes":["C0_goal_consequence","C4_smooth_low","C2_response_alignment"]}],"put_the_bowl_on_the_plate":[{"minimum":0.0,"modes":["C4_smooth_low","C0_goal_consequence"]}]},"stall_window":22,"minimum_progress_gain":0.02,"minimum_dwell":25,"maximum_dwell":75,"max_steps":300,"force_guard":"retract","guard_fraction":0.70,"guard_gain":1.0},
    {"route":"P4_guarded_task_portfolio","modes":["C0_goal_consequence","C1_progress_consequence","C4_smooth_low"],"task_modes":{"open_the_middle_drawer_of_the_cabinet":["C4_smooth_low","C0_goal_consequence","C2_response_alignment","C5_conservative_medoid"],"put_the_bowl_on_the_plate":["C4_smooth_low","C0_goal_consequence"],"turn_on_the_stove":["C0_goal_consequence","C4_smooth_low","C2_response_alignment","C1_progress_consequence"]},"stall_window":24,"minimum_progress_gain":0.02,"minimum_dwell":25,"maximum_dwell":65,"max_steps":300,"force_guard":"retract","guard_fraction":0.75,"guard_gain":1.0},
    {"route":"P5_soft_force_scaling","modes":["C4_smooth_low","C0_goal_consequence","C2_response_alignment","C5_conservative_medoid"],"maximum_dwell":55,"minimum_dwell":25,"max_steps":300,"force_guard":"scale","guard_fraction":0.55,"guard_scale":0.25},
    {"route":"P6_progress_stall_guard","modes":["C0_goal_consequence","C1_progress_consequence","C4_smooth_low","C2_response_alignment"],"stall_window":18,"minimum_progress_gain":0.015,"minimum_dwell":20,"maximum_dwell":70,"max_steps":300,"force_guard":"retract","guard_fraction":0.65,"guard_gain":0.5},
    {"route":"P7_unguarded_phase_control","modes":["C4_smooth_low","C0_goal_consequence","C2_response_alignment","C5_conservative_medoid"],"progress_bands":{"turn_on_the_stove":[{"minimum":0.65,"modes":["C4_smooth_low","C2_response_alignment","C1_progress_consequence"]},{"minimum":0.45,"modes":["C0_goal_consequence","C4_smooth_low","C2_response_alignment"]}]},"stall_window":22,"minimum_progress_gain":0.02,"minimum_dwell":25,"maximum_dwell":75,"max_steps":300},
]
DRAWER_GOAL={"modes":["C0_goal_consequence","C4_smooth_low","C2_response_alignment"],"stall_window":22,"minimum_progress_gain":0.02,"minimum_dwell":25,"maximum_dwell":85,"force_guard":"retract","guard_fraction":0.70,"guard_gain":1.0}
STOVE_SOFT={"modes":["C4_smooth_low","C0_goal_consequence","C2_response_alignment","C5_conservative_medoid"],"maximum_dwell":55,"minimum_dwell":25,"force_guard":"scale","guard_fraction":0.55,"guard_scale":0.25}
BOWL_STABLE={"modes":["C4_smooth_low","C0_goal_consequence"],"stall_window":24,"minimum_progress_gain":0.02,"minimum_dwell":30,"maximum_dwell":90}
EXP24_ROUTES=[
    {"route":"D_physical_chunk","view":"physical","k":1,"replan":10,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"nearest","smooth":0.0,"max_steps":140},
    {"route":"T0_task_modular_primary","modes":["C4_smooth_low"],"task_controls":{"open_the_middle_drawer_of_the_cabinet":DRAWER_GOAL,"put_the_bowl_on_the_plate":BOWL_STABLE,"turn_on_the_stove":STOVE_SOFT},"max_steps":300},
    {"route":"T1_drawer_diverse_unguarded","modes":["C4_smooth_low"],"task_controls":{"open_the_middle_drawer_of_the_cabinet":{"modes":["C4_smooth_low","C0_goal_consequence","C2_response_alignment","C5_conservative_medoid"],"stall_window":22,"minimum_progress_gain":0.02,"minimum_dwell":25,"maximum_dwell":75},"put_the_bowl_on_the_plate":BOWL_STABLE,"turn_on_the_stove":STOVE_SOFT},"max_steps":300},
    {"route":"T2_drawer_diverse_retract","modes":["C4_smooth_low"],"task_controls":{"open_the_middle_drawer_of_the_cabinet":{"modes":["C4_smooth_low","C0_goal_consequence","C2_response_alignment","C5_conservative_medoid"],"maximum_dwell":55,"minimum_dwell":25,"force_guard":"retract","guard_fraction":0.70,"guard_gain":1.0},"put_the_bowl_on_the_plate":BOWL_STABLE,"turn_on_the_stove":STOVE_SOFT},"max_steps":300},
    {"route":"T3_drawer_goal_unguarded","modes":["C4_smooth_low"],"task_controls":{"open_the_middle_drawer_of_the_cabinet":{"modes":["C0_goal_consequence","C4_smooth_low","C2_response_alignment"],"stall_window":22,"minimum_progress_gain":0.02,"minimum_dwell":25,"maximum_dwell":85},"put_the_bowl_on_the_plate":BOWL_STABLE,"turn_on_the_stove":STOVE_SOFT},"max_steps":300},
    {"route":"T4_drawer_response_first","modes":["C4_smooth_low"],"task_controls":{"open_the_middle_drawer_of_the_cabinet":{"modes":["C2_response_alignment","C0_goal_consequence","C4_smooth_low"],"stall_window":20,"minimum_progress_gain":0.015,"minimum_dwell":22,"maximum_dwell":75,"force_guard":"retract","guard_fraction":0.70,"guard_gain":1.0},"put_the_bowl_on_the_plate":BOWL_STABLE,"turn_on_the_stove":STOVE_SOFT},"max_steps":300},
    {"route":"T5_drawer_goal_response_smooth","modes":["C4_smooth_low"],"task_controls":{"open_the_middle_drawer_of_the_cabinet":{"modes":["C0_goal_consequence","C2_response_alignment","C4_smooth_low"],"stall_window":18,"minimum_progress_gain":0.015,"minimum_dwell":20,"maximum_dwell":70,"force_guard":"retract","guard_fraction":0.70,"guard_gain":0.75},"put_the_bowl_on_the_plate":BOWL_STABLE,"turn_on_the_stove":STOVE_SOFT},"max_steps":320},
    {"route":"T6_drawer_goal_retarget","modes":["C4_smooth_low"],"task_controls":{"open_the_middle_drawer_of_the_cabinet":{"modes":["C6_goal_retarget","C0_goal_consequence","C4_smooth_low"],"stall_window":20,"minimum_progress_gain":0.02,"minimum_dwell":25,"maximum_dwell":75,"force_guard":"retract","guard_fraction":0.70,"guard_gain":1.0},"put_the_bowl_on_the_plate":BOWL_STABLE,"turn_on_the_stove":STOVE_SOFT},"max_steps":320},
    {"route":"T7_drawer_medoid_goal","modes":["C4_smooth_low"],"task_controls":{"open_the_middle_drawer_of_the_cabinet":{"modes":["C5_conservative_medoid","C0_goal_consequence","C2_response_alignment"],"stall_window":24,"minimum_progress_gain":0.02,"minimum_dwell":25,"maximum_dwell":80,"force_guard":"retract","guard_fraction":0.70,"guard_gain":1.0},"put_the_bowl_on_the_plate":BOWL_STABLE,"turn_on_the_stove":STOVE_SOFT},"max_steps":320},
]


def drawer_control(modes,window=20,dwell=25,maximum=75,retreat=True):
    value={"modes":modes,"physical_stall_window":window,"minimum_physical_gain":0.003,"minimum_dwell":dwell,"maximum_dwell":maximum}
    if retreat:value.update({"force_guard":"retract","guard_fraction":0.70,"guard_gain":1.0})
    return value


EXP25_ROUTES=[
    {"route":"D_physical_chunk","view":"physical","k":1,"replan":10,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"nearest","smooth":0.0,"max_steps":140},
    {"route":"U0_physical_progress_primary","modes":["C4_smooth_low"],"task_controls":{"open_the_middle_drawer_of_the_cabinet":drawer_control(["C0_goal_consequence","C4_smooth_low","C2_response_alignment","C5_conservative_medoid"]),"put_the_bowl_on_the_plate":BOWL_STABLE,"turn_on_the_stove":STOVE_SOFT},"max_steps":320},
    {"route":"U1_physical_smooth_first","modes":["C4_smooth_low"],"task_controls":{"open_the_middle_drawer_of_the_cabinet":drawer_control(["C4_smooth_low","C0_goal_consequence","C2_response_alignment","C5_conservative_medoid"]),"put_the_bowl_on_the_plate":BOWL_STABLE,"turn_on_the_stove":STOVE_SOFT},"max_steps":320},
    {"route":"U2_physical_response_first","modes":["C4_smooth_low"],"task_controls":{"open_the_middle_drawer_of_the_cabinet":drawer_control(["C2_response_alignment","C0_goal_consequence","C4_smooth_low","C5_conservative_medoid"]),"put_the_bowl_on_the_plate":BOWL_STABLE,"turn_on_the_stove":STOVE_SOFT},"max_steps":320},
    {"route":"U3_physical_medoid_first","modes":["C4_smooth_low"],"task_controls":{"open_the_middle_drawer_of_the_cabinet":drawer_control(["C5_conservative_medoid","C0_goal_consequence","C4_smooth_low","C2_response_alignment"]),"put_the_bowl_on_the_plate":BOWL_STABLE,"turn_on_the_stove":STOVE_SOFT},"max_steps":320},
    {"route":"U4_physical_retarget","modes":["C4_smooth_low"],"task_controls":{"open_the_middle_drawer_of_the_cabinet":drawer_control(["C6_goal_retarget","C0_goal_consequence","C4_smooth_low","C2_response_alignment"]),"put_the_bowl_on_the_plate":BOWL_STABLE,"turn_on_the_stove":STOVE_SOFT},"max_steps":320},
    {"route":"U5_physical_fast_switch","modes":["C4_smooth_low"],"task_controls":{"open_the_middle_drawer_of_the_cabinet":drawer_control(["C0_goal_consequence","C4_smooth_low","C2_response_alignment","C5_conservative_medoid"],12,18,55),"put_the_bowl_on_the_plate":BOWL_STABLE,"turn_on_the_stove":STOVE_SOFT},"max_steps":320},
    {"route":"U6_physical_slow_switch","modes":["C4_smooth_low"],"task_controls":{"open_the_middle_drawer_of_the_cabinet":drawer_control(["C0_goal_consequence","C4_smooth_low","C2_response_alignment","C5_conservative_medoid"],30,35,95),"put_the_bowl_on_the_plate":BOWL_STABLE,"turn_on_the_stove":STOVE_SOFT},"max_steps":320},
    {"route":"U7_retrieval_progress_control","modes":["C4_smooth_low"],"task_controls":{"open_the_middle_drawer_of_the_cabinet":DRAWER_GOAL,"put_the_bowl_on_the_plate":BOWL_STABLE,"turn_on_the_stove":STOVE_SOFT},"max_steps":320},
]
def task_modular_stove(route,stove,max_steps=320):
    return {"route":route,"modes":["C4_smooth_low"],"task_controls":{"open_the_middle_drawer_of_the_cabinet":DRAWER_GOAL,"put_the_bowl_on_the_plate":BOWL_STABLE,"turn_on_the_stove":stove},"max_steps":max_steps}


EXP27_ROUTES=[
    {"route":"D_physical_chunk","view":"physical","k":1,"replan":10,"monotone":False,"advance":0,"retarget":0.0,"aggregate":"nearest","smooth":0.0,"max_steps":140},
    task_modular_stove("V0_default70_soft_goal",{"modes":["C7_physical_default","C4_smooth_low","C0_goal_consequence","C2_response_alignment"],"maximum_dwell":70,"minimum_dwell":35,"force_guard":"scale","guard_fraction":0.55,"guard_scale":0.25}),
    task_modular_stove("V1_default110_soft_goal",{"modes":["C7_physical_default","C4_smooth_low","C0_goal_consequence"],"maximum_dwell":110,"minimum_dwell":55,"force_guard":"scale","guard_fraction":0.55,"guard_scale":0.25}),
    task_modular_stove("V2_soft_diverse_control",STOVE_SOFT),
    task_modular_stove("V3_goal_guarded_stove",{"modes":["C0_goal_consequence","C4_smooth_low","C2_response_alignment"],"stall_window":22,"minimum_progress_gain":0.02,"minimum_dwell":25,"maximum_dwell":85,"force_guard":"retract","guard_fraction":0.70,"guard_gain":1.0}),
    task_modular_stove("V4_response_soft_goal",{"modes":["C2_response_alignment","C4_smooth_low","C0_goal_consequence","C1_progress_consequence"],"stall_window":20,"minimum_progress_gain":0.02,"minimum_dwell":22,"maximum_dwell":70,"force_guard":"retract","guard_fraction":0.75,"guard_gain":0.75}),
    task_modular_stove("V5_phase_risk_stove",{"modes":["C4_smooth_low","C0_goal_consequence","C2_response_alignment"],"progress_bands":{"turn_on_the_stove":[{"minimum":0.65,"modes":["C4_smooth_low","C2_response_alignment","C1_progress_consequence"]},{"minimum":0.45,"modes":["C0_goal_consequence","C4_smooth_low","C2_response_alignment"]}]},"stall_window":22,"minimum_progress_gain":0.02,"minimum_dwell":25,"maximum_dwell":75,"force_guard":"retract","guard_fraction":0.70,"guard_gain":1.0}),
    task_modular_stove("V6_soft_default_goal",{"modes":["C4_smooth_low","C7_physical_default","C0_goal_consequence","C2_response_alignment"],"maximum_dwell":65,"minimum_dwell":30,"force_guard":"scale","guard_fraction":0.55,"guard_scale":0.25}),
    task_modular_stove("V7_medoid_goal_soft",{"modes":["C5_conservative_medoid","C0_goal_consequence","C4_smooth_low","C2_response_alignment"],"stall_window":24,"minimum_progress_gain":0.02,"minimum_dwell":25,"maximum_dwell":80,"force_guard":"retract","guard_fraction":0.70,"guard_gain":1.0}),
]
VIEW={"physical":np.r_[0:3,9:15],"object":np.r_[3:15],"full":np.r_[0:26]}


def dump(path,value):path.write_text(json.dumps(value,indent=2),encoding="utf-8")
def parquet(path,rows):pq.write_table(pa.Table.from_pylist(rows),path,compression="zstd")
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def padded(pos,quat):
    p=np.zeros((2,3));q=np.zeros((2,4));p[:len(pos)]=pos;q[:len(quat)]=quat
    return p,q


def contact_flags(value) -> np.ndarray:
    text=str(value).lower()
    return np.asarray([float("gripper" in text or "finger" in text),float("plate" in text or "cabinet" in text or "stove" in text),float(text not in ("","[]","()"))])


def feature(eef,pos,quat,contact):
    p,q=padded(np.asarray(pos,float),np.asarray(quat,float));relative=p-eef
    return np.r_[eef,p.reshape(-1),relative.reshape(-1),q.reshape(-1),contact_flags(contact)]


def boundary_objects(boundary,task):
    from scripts.exp12.prepare_ranking import TASK_BODIES
    names=boundary["body_names"];return np.asarray([boundary["body_positions"][names.index(x)] for x in TASK_BODIES[task]],float),np.asarray([boundary["body_quaternions"][names.index(x)] for x in TASK_BODIES[task]],float)


def build_library(training_run:Path,k=10):
    records=json.loads((training_run/"artifacts/reference_snapshots_manifest.json").read_text())["episodes"];libraries={}
    for task in TASKS:
        rows=[]
        for record in records:
            if record["task"]!=task:continue
            directory=training_run/record["relative_directory"];boundaries=json.loads((directory/"boundaries.json").read_text())
            with np.load(directory/"trajectory_states.npz",allow_pickle=False) as z:actions=np.asarray(z["actions"],float)
            episode_features=[]
            for boundary in boundaries[:len(actions)]:
                pos,quat=boundary_objects(boundary,task);contact=" ".join(f"{x['geom1_name']}|{x['geom2_name']}" for x in boundary["contact_pairs"])
                episode_features.append(feature(np.asarray(boundary["eef_position"]),pos,quat,contact))
            for i in range(len(actions)):
                pos,_=boundary_objects(boundaries[i],task);chunk=actions[np.clip(np.arange(i,i+k),0,len(actions)-1)];successor=min(i+k,len(actions)-1)
                rows.append({"episode":record["episode"],"index":i,"progress":i/max(1,len(actions)-1),"feature":episode_features[i],"successor_feature":episode_features[successor],"goal_feature":episode_features[-1],"chunk":chunk,"eef":np.asarray(boundaries[i]["eef_position"],float),"anchor":pos[0]})
        matrix=np.asarray([x["feature"] for x in rows]);libraries[task]={"rows":rows,"matrix":matrix,"scale":matrix.std(0),"episodes":np.asarray([x["episode"] for x in rows]),"indexes":np.asarray([x["index"] for x in rows])}
    return libraries


def choose_chunk(spec,state,library,memory,exclude_episode=None):
    cols=VIEW[spec["view"]];matrix=library["matrix"][:,cols];query=state["feature"][cols];pool=np.flatnonzero(library["episodes"]!=exclude_episode) if exclude_episode else np.arange(len(matrix));scale=matrix[pool].std(0);distance=standardized_distance(query,matrix,scale)
    if spec["monotone"] and memory.get("episode") is not None:
        constrained=monotone_window(library["episodes"],library["indexes"],memory["episode"],memory["index"],30)
        if len(constrained):pool=constrained
    ordered=pool[np.argsort(distance[pool],kind="stable")]
    selection=spec.get("selection","distance")
    if selection!="distance":
        shortlist=ordered[:min(len(ordered),int(spec.get("search_k",48)))];local_distance=distance[shortlist];distance_scale=max(float(np.median(local_distance)),1e-6);base_score=local_distance/distance_scale
        weight=float(spec.get("consequence_weight",0.75));rows=[library["rows"][int(i)] for i in shortlist]
        if selection=="progress":score=base_score-weight*np.asarray([x["progress"] for x in rows])
        elif selection=="goal_effect":
            goal_distance=np.asarray([np.linalg.norm((x["successor_feature"]-x["goal_feature"])/(library["scale"]+1e-6)) for x in rows]);goal_distance/=max(float(np.median(goal_distance)),1e-6);score=base_score+weight*goal_distance
        elif selection=="response":
            align=[]
            for x in rows:
                effect=(x["successor_feature"]-x["feature"])/(library["scale"]+1e-6);goal=(x["goal_feature"]-state["feature"])/(library["scale"]+1e-6);align.append(float(np.dot(effect,goal)/(np.linalg.norm(effect)*np.linalg.norm(goal)+1e-6)))
            score=base_score-weight*np.asarray(align)
        else:raise ValueError(f"unknown consequence selection: {selection}")
        ordered=shortlist[np.argsort(score,kind="stable")]
    selected=ordered[:spec["k"]]
    first=library["rows"][int(selected[0])];memory["episode"]=first["episode"];memory["index"]=int(first["index"])+int(spec.get("advance",0))
    chunks=np.asarray([library["rows"][int(i)]["chunk"] for i in selected])
    if spec["aggregate"]=="weighted":chunk=weighted_chunk(chunks,distance[selected])
    elif spec["aggregate"]=="median":
        chunk=np.median(chunks,axis=0);chunk[:,6]=np.where(np.median(np.sign(chunks[:,:,6]),axis=0)>=0,1.,-1.)
    elif spec["aggregate"]=="medoid":
        center=np.median(chunks,axis=0);chunk=chunks[int(np.argmin(np.linalg.norm((chunks-center).reshape(len(chunks),-1),axis=1)))].copy()
    else:chunk=chunks[0].copy()
    if spec["retarget"]:
        source_relative=first["eef"]-first["anchor"];current_relative=state["eef"]-state["pos"][0];correction=(source_relative-current_relative)/.05
        chunk[:,:3]+=spec["retarget"]*correction[None,:]
    smooth=float(spec.get("smooth",0.0))
    if smooth and memory.get("previous_chunk") is not None:
        chunk[:,:6]=(1-smooth)*chunk[:,:6]+smooth*memory["previous_chunk"][:,:6]
    memory["previous_chunk"]=chunk.copy()
    requested=chunk.copy();executed=chunk.copy();executed[:,:6]=np.clip(executed[:,:6],-1,1);executed[:,6]=np.sign(executed[:,6]);progress=float(np.median([library["rows"][int(i)]["progress"] for i in selected]));return requested,executed,selected.tolist(),progress


def runtime_state(obs):
    return {"eef":np.asarray(obs["eef_position"]),"pos":np.asarray(obs["object_positions"]),"quat":np.asarray(obs["object_quaternions"]),"feature":feature(np.asarray(obs["eef_position"]),obs["object_positions"],obs["object_quaternions"],obs["contact_mode_json"])}


def estimate_progress(state,library,exclude_episode=None,k=9):
    matrix=library["matrix"][:,VIEW["full"]];query=state["feature"][VIEW["full"]];pool=np.flatnonzero(library["episodes"]!=exclude_episode) if exclude_episode else np.arange(len(matrix));distance=standardized_distance(query,matrix,matrix[pool].std(0));selected=pool[np.argsort(distance[pool],kind="stable")[:k]]
    return float(np.median([library["rows"][int(i)]["progress"] for i in selected]))


def task_physical_progress(env,task,obs):
    if task=="open_the_middle_drawer_of_the_cabinet":
        address=env.sim.model.get_joint_qpos_addr("wooden_cabinet_1_middle_level");return -float(env.sim.data.qpos[address])
    if task=="turn_on_the_stove":
        address=env.sim.model.get_joint_qpos_addr("flat_stove_1_button");return abs(float(env.sim.data.qpos[address]))
    pos=np.asarray(obs["object_positions"],float);return -float(np.linalg.norm(pos[0]-pos[1]))


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-id",required=True);p.add_argument("--stage",choices=("calibration","formal"),required=True);p.add_argument("--reference-run",type=Path,required=True);p.add_argument("--branch-manifest",type=Path,required=True);p.add_argument("--training-run",type=Path,default=Path("runs/exp8_s2_independent_refs_20260814"));p.add_argument("--authorization",type=Path);p.add_argument("--route-set",choices=("exp15","exp16","exp17","exp21","exp22","exp23","exp24","exp25","exp27"),default="exp15");p.add_argument("--safety-envelope",type=Path);p.add_argument("--maximum-steps",type=int,default=80);p.add_argument("--exclude-target-demo",action="store_true")
    args=p.parse_args();out=ROOT/"runs"/args.run_id
    if out.exists():raise FileExistsError(f"immutable run exists: {out}")
    artifacts,manifests=out/"artifacts",out/"manifests";artifacts.mkdir(parents=True);manifests.mkdir();started=datetime.now(timezone.utc).isoformat();stdout=io.StringIO();stderr=io.StringIO();env=None
    try:
        training=(ROOT/args.training_run).resolve();library=build_library(training);branches=json.loads((ROOT/args.branch_manifest).read_text());routes={"exp15":ROUTES,"exp16":EXP16_ROUTES,"exp17":EXP17_ROUTES,"exp21":EXP21_ROUTES,"exp22":EXP22_ROUTES,"exp23":EXP23_ROUTES,"exp24":EXP24_ROUTES,"exp25":EXP25_ROUTES,"exp27":EXP27_ROUTES}[args.route_set];safety_envelope=json.loads((ROOT/args.safety_envelope).read_text()) if args.safety_envelope else None
        if args.authorization:
            allowed=set(json.loads((ROOT/args.authorization).read_text())["authorized_routes"]);routes=[x for x in routes if x["route"] in allowed or x["route"]=="D_physical_chunk"]
        protocol={"stage":args.stage,"route_set":args.route_set,"routes":routes,"default_route":"D_physical_chunk","training_run":training.name,"training_hash":sha(training/"artifacts/reference_snapshots_manifest.json"),"target_future_candidate_access":False,"expert_path_isolated":True,"exclude_target_demo_from_neighbors_and_scale":args.exclude_target_demo,"maximum_rollout_steps":args.maximum_steps,"safety_envelope":str(args.safety_envelope) if args.safety_envelope else None,"frozen_before_outcomes":True}
        dump(manifests/"recovery_protocol.json",protocol);dump(manifests/"branch_manifest.json",branches);dump(manifests/"preoutcome_hashes.json",{"protocol":sha(manifests/"recovery_protocol.json"),"branches":sha(manifests/"branch_manifest.json")})
        selection,task_manifest=load_selection(ROOT/"experiments/exp1_decision_sparsity/manifests/selected_tasks_pilot.json",ROOT/"experiments/exp1_decision_sparsity/manifests/tasks.json");selected={x["name"]:x for x in selection["tasks"]};wrapper,robosuite_root,assets_root=bootstrap_runtime(ROOT/"third_party/LIBERO",ROOT/"data",artifacts/"libero_config")
        channel_schema=json.loads((ROOT/"experiments/exp3_time_indexed_q_criticality/manifests/effect_channel_schema.json").read_text());contact_schema=load_schema(ROOT/"experiments/exp7_contact_mode_conditioned/manifests/contact_mode_schema.json")
        summaries=[];steps=[];experts=[]
        grouped=defaultdict(list)
        for b in branches:grouped[(b["task"],b["episode"])].append(b)
        with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
            for (task,episode),demo_branches in grouped.items():
                task_def=selected[task];source=task_source_record(task_manifest,task_def["suite"],task_def["task_id"]);env=wrapper.ControlEnv(**environment_kwargs(Path(source["bddl_file_path"])))
                load_episode(env,dataset_path=ROOT/"data"/task_def["demonstration_relative_path"],episode_index=int(episode.split("_")[-1]),robosuite_package_root=robosuite_root,libero_assets_root=assets_root)
                ref_root=ROOT/demo_branches[0].get("reference_run",str(args.reference_run));ref_dir=ref_root/demo_branches[0]["reference_directory"]
                with np.load(ref_dir/"trajectory_states.npz",allow_pickle=False) as z:target_actions=np.asarray(z["actions"],float);integrations=np.asarray(z["integration"],float)
                body_ids=[int(env.sim.model.body_name2id(name)) for name in channel_schema["task_object_audit"][task]["bodies"]]
                for branch in demo_branches:
                    t=int(branch["branch_time"]);controller=ref_dir/f"controller_{t:04d}.npz"
                    # Isolated evaluation-only expert path. It is never passed to choose_chunk.
                    restore_d(env,integrations[t],controller);expert_rows=engine.rollout(env,target_actions,t,None,body_ids,contact_schema,task);experts.append({"branch_id":branch["branch_id"],"task":task,"success":bool(expert_rows[-1]["predicate"]),"steps":len(expert_rows)})
                    for spec in routes:
                        control={**spec,**spec.get("task_controls",{}).get(task,{})}
                        restore_d(env,integrations[t],controller);obs=engine.observation(env,body_ids,contact_schema,task);memory={};pending=None;requested=None;retrieved=[];clip_count=0;action_count=0;safety=False;success=bool(obs["predicate"]);route_steps=[];exceedance_count=0;absolute_200=False;active_stage=-1;mode_index=0;mode_switches=0;mode_started=0;progress_history=[];physical_history=[];retrieval_progress=0.0;guard_events=0;previous_action=None;estimated_progress=estimate_progress(runtime_state(obs),library[task],episode if args.exclude_target_demo else None)
                        threshold=float(safety_envelope["tasks"][task]["primary_threshold_n"]) if safety_envelope else 200.;required=int(safety_envelope["tasks"][task]["consecutive_exceedances_to_stop"]) if safety_envelope else 1
                        route_limit=int(control.get("max_steps",args.maximum_steps))
                        for offset in range(route_limit):
                            if success:break
                            stage=sum(offset>=x for x in control.get("switch_steps",[]));stage_start=0 if stage==0 else control["switch_steps"][stage-1]
                            if "stages" in control:active=EXP17_BASE[control["stages"][stage]]
                            elif "modes" in control:
                                mode_names=control.get("task_modes",{}).get(task,control["modes"])
                                for band in control.get("progress_bands",{}).get(task,[]):
                                    if estimated_progress>=float(band["minimum"]):mode_names=band["modes"];break
                                active=EXP22_MODES[mode_names[min(mode_index,len(mode_names)-1)]];stage=mode_index;stage_start=mode_started
                            else:active=control
                            if stage!=active_stage:memory={};pending=None;active_stage=stage
                            if pending is None or (offset-stage_start)%active["replan"]==0:
                                requested,pending,retrieved,retrieval_progress=choose_chunk(active,runtime_state(obs),library[task],memory,episode if args.exclude_target_demo else None);progress_history.append((offset,retrieval_progress))
                            local=(offset-stage_start)%active["replan"];action=pending[min(local,len(pending)-1)].copy();req=requested[min(local,len(requested)-1)];clip=bool(np.any(np.abs(req[:6]-action[:6])>1e-12));guarded=False;pre_force=float(np.linalg.norm(obs["ee_force"])) if obs["force_valid"] else float("nan")
                            if control.get("force_guard") and np.isfinite(pre_force) and pre_force>float(control.get("guard_fraction",.7))*threshold:
                                guarded=True;guard_events+=1
                                if control["force_guard"]=="retract" and previous_action is not None:action[:6]=-float(control.get("guard_gain",1.0))*previous_action[:6]
                                elif control["force_guard"]=="scale":action[:6]*=float(control.get("guard_scale",.25))
                                else:action[:6]=0.;pending=None
                            action[:6]=np.clip(action[:6],-1,1);clip_count+=clip;action_count+=1;previous_action=action.copy()
                            env.step(action);obs=engine.observation(env,body_ids,contact_schema,task);force=float(np.linalg.norm(obs["ee_force"])) if obs["force_valid"] else float("nan");absolute_200=absolute_200 or bool(np.isfinite(force) and force>200);exceedance_count=exceedance_count+1 if np.isfinite(force) and force>threshold else 0;safety=bool(exceedance_count>=required or (np.isfinite(force) and force>1000));success=bool(obs["predicate"])
                            physical=task_physical_progress(env,task,obs);physical_history.append((offset,physical));route_steps.append({"branch_id":branch["branch_id"],"task":task,"episode":episode,"route":spec["route"],"offset":offset,"coordination_mode":active.get("selection",active.get("aggregate","distance")),"mode_index":mode_index,"estimated_initial_progress":estimated_progress,"retrieval_progress":retrieval_progress,"physical_progress":physical,"force_guarded":guarded,"requested_action":req.tolist(),"executed_action":action.tolist(),"clipped":clip,"retrieved_indices":retrieved,"eef_position":obs["eef_position"].tolist(),"object_positions":obs["object_positions"].tolist(),"predicate":success,"contact_mode_json":obs["contact_mode_json"],"ee_force":obs["ee_force"].tolist(),"force_valid":obs["force_valid"],"safety_stop":safety})
                            if safety:break
                            if "modes" in control and not success:
                                mode_names=control.get("task_modes",{}).get(task,control["modes"])
                                for band in control.get("progress_bands",{}).get(task,[]):
                                    if estimated_progress>=float(band["minimum"]):mode_names=band["modes"];break
                                dwell=offset-mode_started+1;switch=guarded
                                if dwell>=int(control.get("maximum_dwell",10**9)):switch=True
                                window=int(control.get("stall_window",0));recent=[v for step,v in progress_history if step>=offset-window+1]
                                if window and dwell>=int(control.get("minimum_dwell",window)) and len(recent)>=max(4,window//2):
                                    half=len(recent)//2;gain=max(recent[half:])-max(recent[:half]);switch=switch or gain<float(control.get("minimum_progress_gain",0.0))
                                physical_window=int(control.get("physical_stall_window",0));physical_recent=[v for step,v in physical_history if step>=offset-physical_window+1]
                                if physical_window and dwell>=int(control.get("minimum_dwell",physical_window)) and len(physical_recent)>=max(4,physical_window//2):
                                    half=len(physical_recent)//2;gain=max(physical_recent[half:])-max(physical_recent[:half]);switch=switch or gain<float(control.get("minimum_physical_gain",0.0))
                                switch=switch or bool(np.isfinite(force) and force>.8*threshold and dwell>=int(control.get("minimum_dwell",1)))
                                if switch and mode_index+1<len(mode_names):mode_index+=1;mode_switches+=1;mode_started=offset+1;active_stage=-1;progress_history=[]
                        steps.extend(route_steps);summaries.append({"branch_id":branch["branch_id"],"task":task,"episode":episode,"route":spec["route"],"success":success,"safety_stop":safety,"absolute_200_exceeded":absolute_200,"steps":len(route_steps),"mode_switches":mode_switches,"guard_events":guard_events,"estimated_initial_progress":estimated_progress,"final_mode_index":mode_index,"clipped_action_fraction":clip_count/max(1,action_count),"all_states_finite":all(np.all(np.isfinite(x["eef_position"])) for x in route_steps),"terminal_contact_mode_json":obs["contact_mode_json"],"terminal_object_positions":obs["object_positions"].tolist()})
                    print(json.dumps({"branch":branch["branch_id"],"policies":len(routes),"summaries":len(summaries)},sort_keys=True))
                env.close();env=None
        parquet(artifacts/"candidate_summaries.parquet",summaries);parquet(artifacts/"per_step.parquet",steps);parquet(artifacts/"expert_upper_bound.parquet",experts)
        metrics={"status":"completed","run_id":args.run_id,"stage":args.stage,"started_utc":started,"completed_utc":datetime.now(timezone.utc).isoformat(),"branch_count":len(branches),"route_count":len(routes),"candidate_rollout_count":len(summaries),"per_step_count":len(steps),"target_future_candidate_access":False,"expert_success_rate":float(np.mean([x["success"] for x in experts]))}
        dump(out/"metrics.json",metrics);(out/"stdout.log").write_text(stdout.getvalue(),encoding="utf-8");(out/"stderr.log").write_text(stderr.getvalue(),encoding="utf-8");print(json.dumps(metrics,indent=2));return 0
    except Exception as exc:
        if env is not None:env.close()
        stderr.write(traceback.format_exc());dump(out/"metrics.json",{"status":"failed","error":repr(exc)});(out/"stderr.log").write_text(stderr.getvalue(),encoding="utf-8");raise


if __name__=="__main__":raise SystemExit(main())
