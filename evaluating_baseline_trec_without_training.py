# =============================================================================
# Created By  : Hamdi REZGUI
# Created Date: March 29 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Code to Evaluate baseline IR models before training for TDV
# for a TREC Collection 
# =============================================================================
import os
import pickle
import argparse
import time
import pytrec_eval

from Trec_Collection_opt import TrecCollection
import Inverted_structure
import utils



def main():
    # parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--coll_path', nargs="?", type=str)
    parser.add_argument('-i', '--indexed_path', nargs="?", type=str)
    parser.add_argument('-p', '--plot_path', nargs="?", type=str)
    parser.add_argument('-r', '--results_path', nargs="?", type=str)
    parser.add_argument('-f', '--folds', nargs="?", type=int, default=5)
    parser.add_argument('-n', '--experiment_name', nargs="?", type=str)


    args = parser.parse_args()

    print(args, flush=True)
    print('---------------------start-------------------', flush=True)
    start0=time.time()
    #Loading indexed_structure#HR
    start=time.time()
    inverted_structure=Inverted_structure.Inverted_structure()
    inverted_structure.load(args.indexed_path)
    end=time.time()
    print("Time for loading indexed TREC collection ",end-start,flush=True)
    #Computing idf and collection frequencies
    start=time.time()
    inverted_structure.compute_idf()
    inverted_structure.compute_collection_frequencies()
    end=time.time()
    print("Time for computing idf and collection frequencies ", end-start,flush=True)
    #Loading queries and processing queries
    start=time.time()
    Collection = TrecCollection()
    Collection.load_folds_queries(args.coll_path)
    folds_processed_queries=Collection.process_queries(vocabulary=inverted_structure.vocabulary)
    end=time.time()
    print("Time to load queries and process them ",end-start,flush=True) 
    
    # Loading relevance judgements from collection
    with open(args.coll_path + '/qrels', 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)
 
    # Creating for each fold and for a certain experiment a directory for results and plots data
    plot_values_folds_list=[]
    for fold in range(args.folds):

        plot_values = dict()

        for model_name in ['tf',
                           'tf_idf',
                           'DIR',
                           'BM25', 'JM']:
            plot_values[model_name] = [[], []]

        if not os.path.exists(args.results_path + '/fold' + str(fold) + '/' + args.experiment_name):
            os.makedirs(args.results_path + '/fold' + str(fold) + '/' + args.experiment_name)


        if not os.path.exists(args.plot_path + '/fold' + str(fold) + '/'):
            os.makedirs(args.plot_path + '/fold' + str(fold) + '/')
        # Computing metrics for baseline models for a certain fold and updating plot_values dictionnary
#         utils.eval_baseline_index_trec(inverted_structure,
#                                        folds_processed_queries[fold],
#                                        fold,
#                                        qrel,
#                                        plot_values,
#                                        args.results_path,
#                                        args.experiment_name,
#                                        0)
        utils.eval_baseline_diff_index_trec(inverted_structure,
                                       folds_processed_queries[fold],
                                       fold,
                                       qrel,
                                       plot_values,
                                       args.results_path,
                                       args.experiment_name,
                                       0)
        # appending plot values to the list
        plot_values_folds_list.append(plot_values)
    
    print("Total time= ",round(time.time()-start0)," s", flush=True)
    #printing the evaluation baseline models without training.
        
    ndcg20 =dict()
    P20=dict()
    for model_name in ['tf_idf',
                       'DIR',
                       'BM25']:
        ndcg20[model_name] = []
        P20[model_name]=[]
    for fold in range(args.folds):
        for model_name in ['tf_idf',
                           'DIR',
                           'BM25' ]:
            ndcg20[model_name].append(plot_values_folds_list[fold][model_name][1][0]['ndcg_cut_20'])
            P20[model_name].append(plot_values_folds_list[fold][model_name][1][0]['P_20'])
    for model_name in [ 'tf_idf',
                       'DIR',
                       'BM25']:
        average=sum(ndcg20[model_name])/args.folds
        maximum=max(ndcg20[model_name])
        average_P20=sum(P20[model_name])/args.folds
        maximum_P20=max(P20[model_name])
        print("ndcg20 ",model_name," average of folds", average," of collection ",os.path.basename(args.coll_path) ,flush=True)
        print("ndcg20 ", model_name, " max of folds", maximum," of collection ",os.path.basename(args.coll_path), flush=True)
        print("P20 ",model_name," average of folds", average_P20," of collection ",os.path.basename(args.coll_path) ,flush=True)
        print("P20 ", model_name, " max of folds", maximum_P20," of collection ",os.path.basename(args.coll_path), flush=True)


#     ndcg20 =dict()
#     for model_name in ['tf_idf',
#                        'DIR',
#                        'BM25']:
#         ndcg20[model_name] = []
#     for fold in range(args.folds):
#         for model_name in ['tf_idf',
#                        'DIR',
#                        'BM25']:
#             ndcg20[model_name].append(plot_values_folds_list[fold][model_name][1][0]['ndcg_cut_20'])
#     for model_name in ['tf_idf',
#                        'DIR',
#                        'BM25']:
#         average=sum(ndcg20[model_name])/args.folds
#         maximum=max(ndcg20[model_name])
#         print("ndcg20 ",model_name," average of folds", average," of collection ",os.path.basename(args.coll_path) ,flush=True)
#         print("ndcg20 ", model_name, " max of folds", maximum," of collection ",os.path.basename(args.coll_path), flush=True)

    print("-----------------Finished-------------------", flush=True)


if __name__ == "__main__":
    main()
