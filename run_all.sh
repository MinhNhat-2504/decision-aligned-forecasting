set -e
cd "$(dirname "$0")"
LOG=outputs/pipeline.log
: > $LOG
echo "########## 1. BENCHMARK (5 seeds) ##########" | tee -a $LOG
python -u src/train_eval.py 2>>outputs/stderr.log | tee -a $LOG
echo "########## 2. ABLATION (3 seeds) ##########" | tee -a $LOG
python -u src/train_eval.py --ablation --seeds 3 2>>outputs/stderr.log | tee -a $LOG
echo "########## 3. OPERATIONS ##########" | tee -a $LOG
python -u src/operations.py 2>>outputs/stderr.log | tee -a $LOG
echo "########## 4. FIGURES ##########" | tee -a $LOG
python -u src/figures.py 2>>outputs/stderr.log | tee -a $LOG
echo "########## 5. RELATION ATTENTION ##########" | tee -a $LOG
python -u src/analysis.py 2>>outputs/stderr.log | tee -a $LOG
echo "########## DONE ##########" | tee -a $LOG
