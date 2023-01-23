# =============================================================================
# Created By  : Hamdi REZGUI
# Created Date: March 21 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Script to train the classical IR models on the 3 TREC collections
# =============================================================================
source ~/anaconda3/etc/profile.d/conda.sh
conda activate hamdi

NB_EPOC=300

DROPOUT=0.0
FOLDS=5
LR=1e-4
FASTTEXT_PATH=/home/mrim/rezguiha/work/repro_chap7_res/fastText/cc.en.300.bin

for COLL in LA
do
    COLLPATH=TREC/${COLL}/
    INDEXPATH=TREC/${COLL}/

    for MODEL in DIR
    do
        for L1_WEIGHT in 1e-5
            do

        python3 training_on_trec_collection3.py -c ${COLLPATH} -i $INDEXPATH -f ${FASTTEXT_PATH} -p ${COLLPATH}plots -r ${COLLPATH}results -w ${COLLPATH}weights -e $NB_EPOC -l ${L1_WEIGHT} -n ${COLL}_${MODEL}_${L1_WEIGHT}_${DROPOUT}_GPU_weights_LR_${LR} --lr $LR -d ${DROPOUT}  --IR_model ${MODEL} > ${COLLPATH}stdout/${MODEL}_${L1_WEIGHT}_${DROPOUT}_GPU_weights_LR_${LR} 2> ${COLLPATH}stderr/${MODEL}_${L1_WEIGHT}_${DROPOUT}_GPU_weights_LR_${LR} &
            done
    done
done
mail -s "training_on_trec_collection" hamdi.rezgui1993@gmail.com <<< "finished"
