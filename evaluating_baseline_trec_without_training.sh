# =============================================================================
# Created By  : Hamdi REZGUI
# Created Date: March 29 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Script to evaluate the baseline models without training on TREC
# collection. Change the name of the collection to the TREC collection you wish
# to evaluate. The result will be in stdout directory if you have no error (check
# directory stderr before)
# =============================================================================
source activate python3.7

FOLDS=5
FASTTEXT_PATH=/home/mrim/rezguiha/work/repro_chap7_res/fastText/cc.en.300.bin
RESULT_PLOT_PATH=/home/mrim/rezguiha/work/repro_chap7_res/test_evaluation_trec/
for COLL in AP88-89 FT91-94 LA
do
    COLLPATH=TREC/${COLL}
    INDEXPATH=TREC/${COLL}

    python3 evaluating_baseline_trec_without_training.py -c ${COLLPATH} -i $INDEXPATH -p ${COLLPATH}/plots -r ${COLLPATH}/results  -n "Evaluating_baseline_TREC_${COLL}" -f $FOLDS > ${COLLPATH}/stdout/eval.log 2> ${COLLPATH}/stderr/eval.log &
done
mail -s "Evaluating LA collection without training" hamdi.rezgui1993@gmail.com <<< "finished"