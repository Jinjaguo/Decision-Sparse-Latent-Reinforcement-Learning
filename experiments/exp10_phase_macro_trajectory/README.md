# EXP10 reproducibility map

Use the `libero-exp1` conda environment and set `PYTHONPATH=src;.`.

```powershell
python scripts/exp10/prepare_stage_a.py --run-id exp10_a0_phase_macro_dataset_r2_20260814
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python scripts/exp10/run_stage_a_routes.py --run-id exp10_a1_a6_multiroute_models_r1_20260814 --dataset-run runs/exp10_a0_phase_macro_dataset_r2_20260814
python scripts/exp10/evaluate_endpoint_route.py --run-id exp10_a7_endpoint_route_20260814
python scripts/exp10/evaluate_seed_variance.py --run-id exp10_a7_seed_variance_20260814
python scripts/exp10/materialize_protocol_artifacts.py --run-id exp10_a9_protocol_artifacts_20260814
python scripts/exp10/audit_outputs.py --run-id exp10_a10_final_output_audit_20260814
```

All run directories are immutable. The two incomplete development runs and the
superseded dataset builds are preserved. Stage B must not be run because the final
audit records no qualifying route.
