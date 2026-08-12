set -o pipefail
cd "$(dirname "$0")"
LOG=outputs/gpu_run.log
: > $LOG

echo "########## 1. YELLOW: them DCRNN + GraphWaveNet (5 seed, append) ##########" | tee -a $LOG
python -u src/train_eval.py --append --models DCRNN,GraphWaveNet 2>>outputs/stderr.log | tee -a $LOG
echo "########## 2. GREEN: benchmark day du (3 seed) ##########" | tee -a $LOG
DATASET=green python -u src/train_eval.py --seeds 3 2>>outputs/stderr.log | tee -a $LOG
echo "########## 3. GREEN: ablation (3 seed) ##########" | tee -a $LOG
DATASET=green python -u src/train_eval.py --ablation --seeds 3 2>>outputs/stderr.log | tee -a $LOG
echo "########## 4. PHAN TICH HA NGUON (CPU) ##########" | tee -a $LOG
for DS in yellow green; do
  echo "--- $DS ---" | tee -a $LOG
  DATASET=$DS python -u src/dm_test.py    2>>outputs/stderr.log | tee -a $LOG
  DATASET=$DS python -u src/operations.py 2>>outputs/stderr.log | tee -a $LOG
  DATASET=$DS python -u src/ksweep.py     2>>outputs/stderr.log | tee -a $LOG
  DATASET=$DS python -u src/figures.py    2>>outputs/stderr.log | tee -a $LOG
done
echo "########## DONE ##########" | tee -a $LOG
