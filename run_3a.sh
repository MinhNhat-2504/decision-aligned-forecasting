set -o pipefail
cd "$(dirname "$0")"
LOG=outputs/mondrian.log
: > $LOG
echo "########## YELLOW (3 nhom) ##########" | tee -a $LOG
python -u src/mondrian_cqr.py --seeds 3 --bins 3 2>>outputs/stderr.log | tee -a $LOG
echo "########## GREEN (3 nhom) ##########" | tee -a $LOG
DATASET=green python -u src/mondrian_cqr.py --seeds 3 --bins 3 2>>outputs/stderr.log | tee -a $LOG
echo "########## YELLOW (5 nhom, do nhay) ##########" | tee -a $LOG
python -u src/mondrian_cqr.py --seeds 3 --bins 5 2>>outputs/stderr.log | tee -a $LOG
echo "########## DONE ##########" | tee -a $LOG
