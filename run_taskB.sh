set -o pipefail
cd "$(dirname "$0")"
LOG=outputs/taskB.log
: > $LOG
echo "########## YELLOW ##########" | tee -a $LOG
python -u src/decision_selection.py --seeds 3 2>>outputs/stderr.log | tee -a $LOG
echo "########## GREEN ##########" | tee -a $LOG
DATASET=green python -u src/decision_selection.py --seeds 3 2>>outputs/stderr.log | tee -a $LOG
echo "########## DONE ##########" | tee -a $LOG
