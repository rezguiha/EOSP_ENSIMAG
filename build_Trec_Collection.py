# =============================================================================
# Created By  : Jibril FREJJ
# Created Date: April 13 2020
# Modified By  : Hamdi REZGUI
# Modified Date: March 23 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Parsing arguments and launching steps to build a TREC collection
# =============================================================================
import argparse
import os
import time
from Trec_Collection_opt import TrecCollection
import Trec_Collection_opt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--original_coll_path', type=str,help='if present, then produces documents.xml from original source')
    parser.add_argument('-m', '--max_files', type=int,help='if present, then only this number of files are treated from original source')
    parser.add_argument('-c', '--coll_path', nargs="?", type=str)
    parser.add_argument('-i', '--index_path', nargs="?", type=str)
    parser.add_argument('-f', '--fasttext_path', nargs="?", type=str)
    parser.add_argument('-l', '--language', nargs="?", type=str, default='english')
    parser.add_argument('-b', '--build_folds_preprocess', nargs="?", type=bool, default=False) #HR
    parser.add_argument('-n', '--name_of_collection', nargs="?", type=str)

    args = parser.parse_args()

    print("-----------------",args.name_of_collection,"---------------------",flush=True)
    print('-----------------start--------------',flush=True)
    
    #HR

    # For total time execution
    startTime = time.time()

    if args.max_files is not None:
        max_files = args.max_files
        print("Max files treated:",max_files)
    else:
        max_files = -1;

    # if we add the argument --original_coll_path then the original files
    # are uncompressed ans assembled into the XML file documents.xml
    # Do nothing if documents.xml aready exists
    if args.original_coll_path is not None:
        if not os.path.isfile(os.path.join(args.coll_path,'documents.xml')):
            print("Build documents.xml from "+args.original_coll_path)
            Trec_Collection_opt.installFromOrigin(args.original_coll_path,args.coll_path,max_files)
        else:
            print('documents.xml => OK')
    # Set to True only if the collection haven't been processed by creating folds containing 
    # , queries and qrels for each fold and creating csv files from Xml format

    if args.build_folds_preprocess:
        print("Creating folds for the K-fold cross validation and preprocessing",flush=True)
        Trec_Collection_opt.read_collection(args.coll_path)

    print("=> Loading collection documents",flush=True)
    tic=time.time()
    Collection = TrecCollection(k=5)
    nbDocs = Collection.load_documents(args.coll_path)
    toc=time.time()
    print("Number of documents:",nbDocs)
    print("Loading documents time :",round(toc-tic),"s")

    print("=> Build and save inverted structure ",flush=True)
    tic = time.time()
    Collection.build_inverted_index_and_vocabulary(args.index_path)
    toc = time.time()
    print("<= Time to build and save the inverted and direct structure",round(toc-tic),"s")

    totalTime = time.time()-startTime
    print("Total time: ",round(totalTime),"s")
    print("Number of doc processed per s:",round(nbDocs/totalTime))
    print("-----------------Finished----------------", flush=True)


if __name__ == "__main__":
    main()
