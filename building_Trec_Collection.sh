# =============================================================================
# Created By  : Hamdi REZGUI
# Created Date: March 23 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Script to index TREC collections
# =============================================================================
source activate python3.7
LANGUAGE=en
FASTTEXT_PATH=/home/mrim/data/embeddings/cc.${LANGUAGE}.300.bin
#In shell script a boolean variable is just a string. So for put whatever you want in BUILD_FOLDs if you want it to be true and erase the -b $BUILD_FOLDS if you want it to be false because it is False by default. Add -b BUILD_FOLDS if you want otherwise
BUILD_FOLDS=false

for COLL in AP88-89 LA FT91-94
do
    COLLPATH=TREC/${COLL}/
    INDEXPATH=TREC/${COLL}/

    python3 build_Trec_Collection.py -c $COLLPATH -i $INDEXPATH -f $FASTTEXT_PATH  -n ${COLL} > ${COLLPATH}stdout/build 2> ${COLLPATH}stderr/build &

done
mail -s "Building_Trec_Collecitons" hamdi.rezgui1993@gmail.com <<< "finished"