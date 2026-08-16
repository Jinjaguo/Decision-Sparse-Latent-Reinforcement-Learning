# EXP_R9 report: safety-ranking diagnostic

R9 compares held-out unsafe ranking from the R5 factorized unsafe head with
the R6 safety-only control. Only seven of 60 branches contain unsafe
candidates, all in the stove task; other task metrics are correctly undefined.

| head | unsafe AP | mean unsafe rank | top-1 unsafe hit |
|---|---:|---:|---:|
| factorized unsafe | 0.755 | 2.17 | 0.571 |
| safety-only | 0.714 | 2.38 | 0.571 |

The factorized head ranks unsafe candidates better than the safety-only head,
but the small support and task concentration prevent a general safety claim.
R8 remains the fixed-threshold negative control: threshold changes did not
improve unsafe recall.

After EXP_R9 completion, C: was checked once: 1,180.10 GB free of 1,862.02
GB. E: was not inspected.
