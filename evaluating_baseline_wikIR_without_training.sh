# =============================================================================
# Created By  : Hamdi REZGUI
# Created Date: March 30 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Script to evaluate the baseline models without training on WikIR
# collection.The result will be in stdout directory if you have no error (check
# directory stderr before)
# =============================================================================
source activate python3.7

COLLPATH=/home/mrim/rezguiha/work/repro_chap7_res/wikIR_78/
INDEXPATH=/home/mrim/rezguiha/work/repro_chap7_res/enwikIR_full_documents_indexed

RESULT_PLOT_PATH=/home/mrim/rezguiha/work/repro_chap7_res/test_evaluation_wikIR/

# COLLPATH=/home/mrim/rezguiha/work/repro_chap7_res/mini_wikIR_78/
# INDEXPATH=/home/mrim/rezguiha/work/repro_chap7_res/mini_indexed_wikIR

# RESULT_PLOT_PATH=/home/mrim/rezguiha/work/repro_chap7_res/test_evaluation_wikIR/


python3 evaluating_baseline_wikIR_without_training.py -c ${COLLPATH} -i $INDEXPATH -p ${RESULT_PLOT_PATH}plots -r ${RESULT_PLOT_PATH}results  -n "Evaluating_baseline_WikIR" > ${RESULT_PLOT_PATH}stdout/out_eval_wikIR.log 2> ${RESULT_PLOT_PATH}stderr/err_eval_wikIR.log &

mail -s "Monitoring Evaluating wikIR_78 collection without training" hamdi.rezgui1993@gmail.com <<< "finished"