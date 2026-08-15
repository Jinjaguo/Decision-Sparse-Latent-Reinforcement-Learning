# EXP9 reproducibility map

Stage A:

```powershell
$env:PYTHONPATH = (Join-Path ([IO.Path]::GetFullPath('.')) 'src')
python scripts/exp9/build_hybrid_targets.py --output-run runs/exp9_a0_hybrid_targets_r1_20260814
python scripts/exp9/retrospective_feasibility.py --target-run runs/exp9_a0_hybrid_targets_r1_20260814 --output-run runs/exp9_a1_a6_retrospective_models_20260814
```

Stage B must not start unless `architecture_selection.json` explicitly records
`new_cohort_authorized: true`.
