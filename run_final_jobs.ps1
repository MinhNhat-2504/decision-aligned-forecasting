# Long-running robustness suite for the final submission draft.
# Every command writes a separate CSV/NPZ tag; no primary benchmark is overwritten.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Clear-DatasetEnv {
    Remove-Item Env:DATASET -ErrorAction SilentlyContinue
}

# 1. Complete the missing green multi-seed Graph WaveNet-Q decision analysis.
$env:DATASET = 'green'
python -u -B src\decision_robustness.py --models GraphWaveNet-Q --seeds 3 --tag decision_seed_green_gwnq
Clear-DatasetEnv

# 2. Parameter-budget robustness: 61k-vs-62k and 147k-vs-148k model pairs.
python -u -B src\capacity_matched.py --seeds 3 --tag capacity_matched_yellow
$env:DATASET = 'green'
python -u -B src\capacity_matched.py --seeds 3 --tag capacity_matched_green
Clear-DatasetEnv

# 3. Earlier temporal origin: Jan train, first half of Feb validation, second
# half of Feb test.  It removes all March observations from the data object.
$env:T_END = '2016-02-29 23:00:00'
$env:TRAIN_END = '2016-01-31 23:00:00'
$env:VAL_END = '2016-02-15 23:00:00'
python -u -B src\decision_robustness.py --seeds 3 --tag early_origin_yellow
$env:DATASET = 'green'
python -u -B src\decision_robustness.py --seeds 3 --tag early_origin_green
Clear-DatasetEnv
Remove-Item Env:T_END -ErrorAction SilentlyContinue
Remove-Item Env:TRAIN_END -ErrorAction SilentlyContinue
Remove-Item Env:VAL_END -ErrorAction SilentlyContinue

# 4. Insert only completed results, regenerate LaTeX, and compile the final PDF.
python -u -B src\update_robustness_manuscript.py
python -u -B src\markdown_to_latex.py
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
