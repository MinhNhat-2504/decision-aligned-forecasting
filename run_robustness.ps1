# Reproducible robustness suite for the paper.  Each output has a distinct tag,
# so the original benchmark artefacts remain unchanged.  Use -u for live logs.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# 1. Decision outcomes across independent initialisations.
python -u -B src\decision_robustness.py --tag decision_seed
$env:DATASET = 'green'
python -u -B src\decision_robustness.py --tag decision_seed
Remove-Item Env:DATASET

# 2. Same task, capacity matched within 0.4% parameter count.
python -u -B src\capacity_matched.py --seeds 3
$env:DATASET = 'green'
python -u -B src\capacity_matched.py --seeds 3
Remove-Item Env:DATASET

# 3. Earlier rolling origin: train through January, tune through mid-February,
# and test on the subsequent two weeks.  This never reads a later observation.
$env:T_END = '2016-02-29 23:00:00'
$env:TRAIN_END = '2016-01-31 23:00:00'
$env:VAL_END = '2016-02-15 23:00:00'
python -u -B src\decision_robustness.py --seeds 3 --tag early_origin_seed
$env:DATASET = 'green'
python -u -B src\decision_robustness.py --seeds 3 --tag early_origin_seed
Remove-Item Env:DATASET
Remove-Item Env:T_END
Remove-Item Env:TRAIN_END
Remove-Item Env:VAL_END

# 4. Stronger routing comparator, evaluated on the locked primary test block.
python -u -B src\routing_robustness.py
$env:DATASET = 'green'
python -u -B src\routing_robustness.py
