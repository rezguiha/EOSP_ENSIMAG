# =============================================================================
# Created By  : Hamdi REZGUI
# Created Date: March 30 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Script to evaluate the baseline models without training on WikIRS
# collection.The result will be in stdout directory if you have no error (check
# directory stderr before)
# =============================================================================
source activate python3.7

COLLPATH=/home/mrim/rezguiha/work/repro_chap7_res/wikIRS_78/
INDEXPATH=/home/mrim/rezguiha/work/repro_chap7_res/enwikIRS_full_documents_indexed

RESULT_PLOT_PATH=/home/mrim/rezguiha/work/repro_chap7_res/test_evaluation_wikIRS/

python3 evaluating_baseline_wikIR_without_training.py -c ${COLLPATH} -i $INDEXPATH -p ${RESULT_PLOT_PATH}plots -r ${RESULT_PLOT_PATH}results  -n "Evaluating_baseline_WikIRS" > ${RESULT_PLOT_PATH}stdout/out_eval_wikIRS.log 2> ${RESULT_PLOT_PATH}stderr/err_eval_wikIRS.log &

mail -s "Evaluating wikIRS_78 collection without training" hamdi.rezgui1993@gmail.com <<< "finished"