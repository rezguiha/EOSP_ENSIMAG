# =============================================================================
# Created By  : Hamdi REZGUI
# Created Date: March 19 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Script to index WikIRS collection
# =============================================================================

source activate /home/mrim/rezguiha/anaconda3/envs/python3.7


COLL_PATH=/home/mrim/rezguiha/work/repro_chap7_res/wikIRS_78
FASTTEXT_PATH=/home/mrim/rezguiha/work/repro_chap7_res/fastText/cc.en.300.bin
#INDEX_PATH=/home/mrim/rezguiha/work/repro_chap7_res/enwikIRS_indexed
INDEX_PATH=/home/mrim/rezguiha/work/repro_chap7_res/enwikIRS_full_documents_indexed

python3 building_wikir_collection.py -c $COLL_PATH -i $INDEX_PATH -f $FASTTEXT_PATH > out_python_build_wikirS.log 2>&1 &
mail -s "building_wikir_collection" hamdi.rezgui1993@gmail.com <<< "finished"