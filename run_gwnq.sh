set -o pipefail
cd "$(dirname "$0")"
LOG=outputs/gwnq_run.log
: > $LOG
echo "########## YELLOW: GraphWaveNet-Q (5 seed) ##########" | tee -a $LOG
python -u src/train_eval.py --models GraphWaveNet-Q 2>>outputs/stderr.log | tee -a $LOG
echo "########## GREEN: GraphWaveNet-Q (3 seed) ##########" | tee -a $LOG
DATASET=green python -u src/train_eval.py --models GraphWaveNet-Q --seeds 3 2>>outputs/stderr.log | tee -a $LOG
echo "########## MERGE + PHAN TICH LAI ##########" | tee -a $LOG
for DS in yellow green; do
  echo "--- $DS ---" | tee -a $LOG
  DATASET=$DS python -u src/merge_results.py 2>>outputs/stderr.log | tee -a $LOG
  DATASET=$DS python -u src/operations.py    2>>outputs/stderr.log | tee -a $LOG
  DATASET=$DS python -u src/dm_test.py       2>>outputs/stderr.log | tee -a $LOG
  DATASET=$DS python -u src/figures.py       2>>outputs/stderr.log | tee -a $LOG
done
echo "########## DONE ##########" | tee -a $LOG
