# =============================================================================
# Created By  : Jibril FREJJ
# Created Date: February 16 2020
# Modified By  : Hamdi REZGUI
# Modified Date: March 21 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Definition of the classes of differentiable IR models with their
# network architecture
# =============================================================================
#packages
import os
import json
import pickle
import random
import collections
import numpy as np
import pytrec_eval
import tensorflow as tf
from collections import Counter
import time
import resource
#files
import baseline_models_and_tdv_implementation

####Useful functions for reading qrel files when you are loading the collection #HR
def read_qrels(path):
    """Reads qrel files for both Wikir and Trec collection""" #HR
    qrels = []
    with open(path,'r') as f:
        for line in f:
            rel = line.split()
            qrels.append([rel[0],rel[2]])
    return qrels

def read_trec_train_qrels(path):
    """read qrel files for trec collection for training and build dictionnary containing qrels for positive documents and negative documents""" #HR
    pos_qrels = []
    neg_qrels = dict()
    with open(path,'r') as f:
        for line in f:
            rel = line.split()
            if rel[3] == '1':
                pos_qrels.append([rel[0],rel[2]])
            else :
                if rel[0] not in neg_qrels:
                    neg_qrels[rel[0]] = [rel[2]]
                else:
                    neg_qrels[rel[0]].append(rel[2])
    return {'pos':pos_qrels,'neg':neg_qrels}







####Useful functions for after computation of TDV weights by training models to update inverted index, compute some properties like idf,evaluate performance
### of baseline models and their associated TDV implementation  #HR

def build_inverted_index(Collection, weights):
    """Function that updates the inverted index of a collection by erasing the tokens that have 0 as TDV and multiply the rest by their TDV weight""" #HR
    inverted_index = dict()
    for key, value in Collection.inverted_index.items():
        if weights[key] == 0:
            continue
        inverted_index[key] = Counter()
        for doc_id in value:
            inverted_index[key][doc_id] += weights[key] * Collection.inverted_index[key][doc_id]
    return inverted_index


def compute_idf(Collection, inverted_index, weights=None):
    """Functions that compute the idf with or without introduction of TDV""" #HR
    nb_docs = len(Collection.doc_index)
    if weights is None:
        return {token: np.log((nb_docs + 1) / (1 + len(inverted_index[token]))) for token in inverted_index}
    else:
        sums = {key: sum(inverted_index[key].values()) for key in inverted_index}
        maxdf = max(sums.values())
        return {token: np.log((maxdf + 1) / (1 + sums[token])) for token in inverted_index}


    # Here in the following function we give the weights when we want the docslengths to be the number of occurence
    # the weights are here for regularization purposes
def compute_docs_length(inverted_index, weights=None):
    """Function that computes document length with TDV or without it""" #HR
    docs_length = Counter()

    if weights is None:
        for term, posting in inverted_index.items():
            for doc_id, nb_occurence in posting.items():
                docs_length[doc_id] += nb_occurence

    else:
        for term, posting in inverted_index.items():
            for doc_id, nb_occurence in posting.items():
                docs_length[doc_id] += nb_occurence / weights[term]

    return docs_length


def compute_collection_frequencies(docs_length, inverted_index):
    """Function that computes frequency of tokens in a  collection""" #HR
    coll_length = sum([value for key, value in docs_length.items()])
    return {token: sum([freq for _, freq in inverted_index[token].items()]) / coll_length for token in inverted_index}


def evaluate_inverted_index(inverted_index):
    """Function that takes an inverted index and calculate its vocabulary size and total number of elements""" #HR
    vocab_size = len(inverted_index)
    tot_nb_elem = 0
    for key, value in inverted_index.items():
        tot_nb_elem += len(value)
    return vocab_size, tot_nb_elem


def compute_metrics(queries_ID,documents_ID, qrel, baseline_model,score_file_path,top_k=1000, save_res=False):
    """Function that saves the results of retrieval: the top_k documents according to their score in a format suitable for the pytrec_eval library . Then, it computes different metrics for IR using the pytrec_eval package""" #HR
    # queries_ID is the array of queries IDs from a Queries instance
    #documents_ID is the array of document IDs from the Inverted_structure instance
    #qrel is the loaded query document relavance file
    #Score_file_path is the full path to the directory including the name of the file where to store the top_k results in the format of pytrec_eval
    #baseline_model is the instance of the class corresponding to the baseline model to evaluate. baseline_model.runQueries() is a result generator 

    #Writing the top k results in the format for pytrec_eval
    with open(score_file_path, 'w') as f:
            for internal_query_ID, counter_doc_relavance_score in enumerate(baseline_model.runQueries()):
                for i, scores in enumerate(counter_doc_relavance_score.most_common(top_k)):
                    internal_document_ID=int(scores[0])
                    relavance_score=scores[1]
                    f.write(str(queries_ID[internal_query_ID]) + ' Q0 ' + str(documents_ID[internal_document_ID]) + ' ' + str(i) + ' ' + str(relavance_score) + ' 0\n')

    #Loading result score file using pytrec_eval
    with open(score_file_path, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)
    if not save_res:
        os.remove(score_file_path)

    #Evaluating metrics for all queries that have a query document relavance
    measures = {"map", "ndcg_cut", "recall", "P"}

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, measures)

    all_metrics = evaluator.evaluate(run)
    #Aggregating metrics and computing the average
    metrics = {'P_5': 0,
               'P_10': 0,
               'P_20': 0,
               'ndcg_cut_5': 0,
               'ndcg_cut_10': 0,
               'ndcg_cut_20': 0,
               'ndcg_cut_1000': 0,
               'map': 0,
               'recall_1000': 0}

    nb_queries = len(all_metrics)
    for key, values in all_metrics.items():
        for metric in metrics:
            metrics[metric] += values[metric] / nb_queries

    return metrics

def utils_compute_info_retrieval(Collection, weights, weighted=True):
    """Computes inverted index, idf, document length and c_frequency for a collection with TDV weights""" #HR
    inverted_index = build_inverted_index(Collection, weights)
    if weighted:
        idf = compute_idf(Collection, inverted_index, weights)
        docs_length = compute_docs_length(inverted_index)
        c_freq = compute_collection_frequencies(docs_length, inverted_index)
    else:
        idf = compute_idf(Collection, inverted_index)
        docs_length = compute_docs_length(inverted_index, weights)
        c_freq = compute_collection_frequencies(docs_length, inverted_index)
    return inverted_index, idf, docs_length, c_freq


# HR added this function to evaluate baseline models on TREC. It is a modified version of eval_baseline_index in the original file. The calls for the function in other files were different from its definition. I added the JM model too. #HR
def eval_baseline_index_trec(inverted_structure,
                        queries_of_fold_struct,
                        fold,
                        qrel,
                        plot_values,
                        results_path,
                        experiment_name,
                        epoch):
    """This function computes the metrics for the baseline models for term matching methods and
    updates the plot values dictionary for a certain fold and a certain epoch.This function is to be used on Trec collection """ #HR
    print('tf')

    baseline_model = baseline_models_and_tdv_implementation.simple_tf(queries_of_fold_struct,inverted_structure)

    if not os.path.exists(results_path + '/fold' + str(fold) + '/' + experiment_name + '/tf/'):
        os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/tf/')

    metrics = compute_metrics(queries_of_fold_struct.queries_IDs,
                              inverted_structure.document_IDs,
                              qrel,
                              baseline_model,
                              results_path + '/fold' + str(fold) + '/' +  experiment_name + '/tf/' + str(epoch))

    plot_values['tf'][0].append(1.0)
    plot_values['tf'][1].append(metrics)

    print('tf_idf')

    baseline_model = baseline_models_and_tdv_implementation.tf_idf(queries_of_fold_struct,inverted_structure)

    if not os.path.exists(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/tf_idf/'):
        os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/tf_idf/')

    metrics = compute_metrics(queries_of_fold_struct.queries_IDs,
                              inverted_structure.document_IDs,
                              qrel,
                              baseline_model,
                              results_path + '/fold' + str(fold) + '/' +  experiment_name + '/tf_idf/' + str(epoch))

    plot_values['tf_idf'][0].append(1.0)
    plot_values['tf_idf'][1].append(metrics)

    print('DIR')

    baseline_model = baseline_models_and_tdv_implementation.dir_language_model(queries_of_fold_struct,inverted_structure)

    if not os.path.exists(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/DIR/'):
        os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/DIR/')

    metrics = compute_metrics(queries_of_fold_struct.queries_IDs,
                              inverted_structure.document_IDs,
                              qrel,
                              baseline_model,
                              results_path + '/fold' + str(fold) + '/' + experiment_name + '/DIR/' + str(epoch))

    plot_values['DIR'][0].append(1.0)
    plot_values['DIR'][1].append(metrics)

    print('BM25')

    baseline_model = baseline_models_and_tdv_implementation.Okapi_BM25(queries_of_fold_struct,inverted_structure)

    if not os.path.exists(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/BM25/'):
        os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/BM25/')

    metrics = compute_metrics(queries_of_fold_struct.queries_IDs,
                              inverted_structure.document_IDs,
                              qrel,
                              baseline_model,
                              results_path + '/fold' + str(fold) + '/' +  experiment_name + '/BM25/' + str(epoch))

    plot_values['BM25'][0].append(1.0)
    plot_values['BM25'][1].append(metrics)

    print('JM')

    baseline_model = baseline_models_and_tdv_implementation.JM_language_model(queries_of_fold_struct,inverted_structure)

    if not os.path.exists(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/JM/'):
        os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/JM/')

    metrics = compute_metrics(queries_of_fold_struct.queries_IDs,
                              inverted_structure.document_IDs,
                              qrel,
                              baseline_model,
                              results_path + '/fold' + str(fold) + '/' +  experiment_name + '/JM/' + str(epoch))

    plot_values['JM'][0].append(1.0)
    plot_values['JM'][1].append(metrics)



    # HR added this function to evaluate baseline models on wikIR collections. It is a modified version of eval_baseline_index in the original file. The calls for the function in other files were different from its definition. I added the JM model too
def eval_baseline_index_wikir(inverted_structure,
                              validation_queries_struct,
                              test_queries_struct,
                              validation_qrel,
                              test_qrel,
                              validation_plot_values,
                              test_plot_values,
                              results_path,
                              experiment_name,
                              epoch):
    """This function computes the metrics for the baseline models for term matching methods and
    updates the plot values dictionary for a certain fold and a certain epoch.This function is to be used on Trec collection """ #HR
    start0=time.time()
    
    number_val_queries=validation_queries_struct.get_number_of_queries()
    number_test_queries=test_queries_struct.get_number_of_queries()
    print("Number of validation queries = ",number_val_queries,flush=True)
    print("Number of test queries = ",number_test_queries,flush=True)
    print("Memory usage start eval_baseline", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
    print('--------------------tf---------------------------',flush=True)
#     ###############validation
    
#     start=time.time()

#     baseline_model = baseline_models_and_tdv_implementation.simple_tf(validation_queries_struct,inverted_structure)


#     if not os.path.exists(results_path + '/validation/' + experiment_name + '/tf/'):
#         os.makedirs(results_path + '/validation/' + experiment_name + '/tf/')


#     metrics = compute_metrics(validation_queries_struct.queries_IDs,
#                               inverted_structure.document_IDs,
#                               validation_qrel,
#                               baseline_model,
#                               results_path + '/validation/' + experiment_name + '/tf/' + str(epoch))
    
#     end=time.time()
#     print("Metrics TF ", metrics,flush=True)
#     print("Time for computing results and metrics TF validation ",((end-start)/number_val_queries)*1000, "ms",flush=True)
#     print("Memory usage at the end of tf validation", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
#     validation_plot_values['tf'][0].append(1.0)
#     validation_plot_values['tf'][1].append(metrics)

    ################Test

    start=time.time()

    baseline_model = baseline_models_and_tdv_implementation.simple_tf(test_queries_struct,inverted_structure)


    if not os.path.exists(results_path + '/test/' + experiment_name + '/tf/'):
        os.makedirs(results_path + '/test/' + experiment_name + '/tf/')


    metrics = compute_metrics(test_queries_struct.queries_IDs,
                              inverted_structure.document_IDs,
                              test_qrel,
                              baseline_model,
                              results_path + '/test/' + experiment_name + '/tf/' + str(epoch))

    end=time.time()
    print("Time for computing results and metrics TF test ",((end-start)/number_test_queries)*1000, "ms",flush=True)
    print("Memory usage at the end of tf test", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
    
    print("Metrics TF ", metrics,flush=True)
    test_plot_values['tf'][0].append(1.0)
    test_plot_values['tf'][1].append(metrics)

    print('-----------------------tf_idf------------------------------',flush=True)
#     #########validation

#     start=time.time()

#     baseline_model = baseline_models_and_tdv_implementation.tf_idf(validation_queries_struct,inverted_structure)



#     if not os.path.exists(results_path + '/validation/' +  experiment_name + '/tf_idf/'):
#         os.makedirs(results_path + '/validation/' +  experiment_name + '/tf_idf/')


#     metrics = compute_metrics(validation_queries_struct.queries_IDs,
#                               inverted_structure.document_IDs,
#                               validation_qrel,
#                               baseline_model,
#                               results_path + '/validation/' +  experiment_name + '/tf_idf/' + str(epoch))

#     end=time.time()
#     print("Time for computing results and metrics TF-IDF validation ",((end-start)/number_val_queries)*1000, "ms",flush=True)
#     print("Memory usage at the end of tf_idf validation", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)

#     validation_plot_values['tf_idf'][0].append(1.0)
#     validation_plot_values['tf_idf'][1].append(metrics)
    ###########test

    start=time.time()

    baseline_model = baseline_models_and_tdv_implementation.tf_idf(test_queries_struct,inverted_structure)


    if not os.path.exists(results_path + '/test/' +  experiment_name + '/tf_idf/'):
        os.makedirs(results_path + '/test/' +  experiment_name + '/tf_idf/')


    metrics = compute_metrics(test_queries_struct.queries_IDs,
                              inverted_structure.document_IDs,
                              test_qrel,
                              baseline_model,
                              results_path + '/test/' +  experiment_name + '/tf_idf/' + str(epoch))

    end=time.time()
    print("Time for computing results and metrics TF-IDF test ",((end-start)/number_test_queries)*1000, "ms",flush=True)
    print("Memory usage at the end of tf_idf test", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)

    test_plot_values['tf_idf'][0].append(1.0)
    test_plot_values['tf_idf'][1].append(metrics)

    print('--------------------------DIR----------------------------',flush=True)
#     ############validation
#     start=time.time()

#     baseline_model = baseline_models_and_tdv_implementation.dir_language_model(validation_queries_struct,inverted_structure)

  
#     if not os.path.exists(results_path + '/validation/' +  experiment_name + '/DIR/'):
#         os.makedirs(results_path + '/validation/' +  experiment_name + '/DIR/')

#     metrics = compute_metrics(validation_queries_struct.queries_IDs,
#                               inverted_structure.document_IDs,
#                               validation_qrel,
#                               baseline_model,
#                               results_path + '/validation/' + experiment_name + '/DIR/' + str(epoch))
#     end=time.time()
#     print("Time for computing results and metrics DIR validation ",((end-start)/number_val_queries)*1000, "ms",flush=True)
#     print("Memory usage at the end of DIR validation", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)

#     validation_plot_values['DIR'][0].append(1.0)
#     validation_plot_values['DIR'][1].append(metrics)

    ##############test
    start=time.time()

    baseline_model = baseline_models_and_tdv_implementation.dir_language_model(test_queries_struct,inverted_structure)


    if not os.path.exists(results_path + '/test/' +  experiment_name + '/DIR/'):
        os.makedirs(results_path + '/test/' +  experiment_name + '/DIR/')


    metrics = compute_metrics(test_queries_struct.queries_IDs,
                              inverted_structure.document_IDs,
                              test_qrel,
                              baseline_model,
                              results_path + '/test/' + experiment_name + '/DIR/' + str(epoch))

    end=time.time()
    print("Time for computing resutls and metrics DIR test ",((end-start)/number_test_queries)*1000, "ms",flush=True)
    print("Memory usage at the end of DIR test", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)

    test_plot_values['DIR'][0].append(1.0)
    test_plot_values['DIR'][1].append(metrics)

    print('---------------------------BM25----------------------',flush=True)
#     ############validation
#     start=time.time()

#     baseline_model = baseline_models_and_tdv_implementation.Okapi_BM25(validation_queries_struct,inverted_structure)


#     if not os.path.exists(results_path + '/validation/' +  experiment_name + '/BM25/'):
#         os.makedirs(results_path + '/validation/' +  experiment_name + '/BM25/')


#     metrics = compute_metrics(validation_queries_struct.queries_IDs,
#                               inverted_structure.document_IDs,
#                               validation_qrel,
#                               baseline_model,
#                               results_path + '/validation/' +  experiment_name + '/BM25/' + str(epoch))

#     end=time.time()
#     print("Time for computing results and metrics BM25 validation ",((end-start)/number_val_queries)*1000, "ms",flush=True)
#     print("Memory usage at the end of BM25 validation", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)

#     validation_plot_values['BM25'][0].append(1.0)
#     validation_plot_values['BM25'][1].append(metrics)

    ############test
    start=time.time()

    baseline_model = baseline_models_and_tdv_implementation.Okapi_BM25(test_queries_struct,inverted_structure)

  
    if not os.path.exists(results_path + '/test/' +  experiment_name + '/BM25/'):
        os.makedirs(results_path + '/test/' +  experiment_name + '/BM25/')


    metrics = compute_metrics(test_queries_struct.queries_IDs,
                              inverted_structure.document_IDs,
                              test_qrel,
                              baseline_model,
                              results_path + '/test/' +  experiment_name + '/BM25/' + str(epoch))
    end=time.time()
    print("Time for computing results and metrics BM25 test ",((end-start)/number_test_queries)*1000, "ms",flush=True)
    print("Memory usage at the end of BM25 test", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)

    test_plot_values['BM25'][0].append(1.0)
    test_plot_values['BM25'][1].append(metrics)

    print('--------------------------JM------------------------------',flush=True)
#     #########validation
#     start=time.time()

#     baseline_model = baseline_models_and_tdv_implementation.JM_language_model(validation_queries_struct,inverted_structure)

 
#     if not os.path.exists(results_path + '/validation/' +  experiment_name + '/JM/'):
#         os.makedirs(results_path + '/validation/' +  experiment_name + '/JM/')

 
#     metrics = compute_metrics(validation_queries_struct.queries_IDs,
#                               inverted_structure.document_IDs,
#                               validation_qrel,
#                               baseline_model,
#                               results_path + '/validation/' +  experiment_name + '/JM/' + str(epoch))
#     end=time.time()
#     print("Time for computing results and metrics JM validation ",((end-start)/number_val_queries)*1000, "ms",flush=True)
#     print("Memory usage at the end of JM validation", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)


#     validation_plot_values['JM'][0].append(1.0)
#     validation_plot_values['JM'][1].append(metrics)

    ##########test
    start=time.time()

    baseline_model = baseline_models_and_tdv_implementation.JM_language_model(test_queries_struct,inverted_structure)


    if not os.path.exists(results_path + '/test/' +  experiment_name + '/JM/'):
        os.makedirs(results_path + '/test/' +  experiment_name + '/JM/')

 
    metrics = compute_metrics(test_queries_struct.queries_IDs,
                              inverted_structure.document_IDs,
                              test_qrel,
                              baseline_model,
                              results_path + '/test/' +  experiment_name + '/JM/' + str(epoch))
    end=time.time()
    print("Time for computing results and metrics JM test ",((end-start)/number_test_queries)*1000, "ms",flush=True)
    print("Memory usage at the end of JM validation", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)

    test_plot_values['JM'][0].append(1.0)
    test_plot_values['JM'][1].append(metrics)
    print("Total time to evaluate baselines models and compute metrics =" , time.time()-start0,flush=True)
    

    # HR added this function to evaluate baseline models on TREC after training to get the TDV weights. It is a modified version of eval_learned_index in the original file. The calls for the function in other files were different from its definition. I added the JM model too. #HR
def eval_learned_index_trec(new_inverted_structure,
                            queries_of_fold_struct,
                            fold,
                            qrel,
                            plot_values,
                            results_path,
                            experiment_name,
                            epoch,
                            IR_model,
                            plot_path,
                            prop_elem_index,
                            model):
    """Evaluate the performance of baseline models and their corresponding weighted (TDV) versions and saves the results in a pickle object file
    This is is to be used after training the neural model (calculating the TDV weights of terms) """ #HR



    if IR_model == 'tf_idf':

        print('------------tf_idf----------',flush=True)

        baseline_model = baseline_models_and_tdv_implementation.tf_idf(queries_of_fold_struct,new_inverted_structure)

        if not os.path.exists(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/tf_idf/'):
            os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/tf_idf/')

        metrics = compute_metrics(queries_of_fold_struct.queries_IDs,
                              new_inverted_structure.document_IDs,
                              qrel,
                              baseline_model,
                              results_path + '/fold' + str(fold) + '/' +  experiment_name + '/tf_idf/' + str(epoch))
        
        plot_values['tf_idf'][0].append(prop_elem_index)
        plot_values['tf_idf'][1].append(metrics)


    if IR_model == 'DIR':

        mu = model.mu.numpy()

        print('DIR')

        baseline_model = baseline_models_and_tdv_implementation.dir_language_model(queries_of_fold_struct,new_inverted_structure,mu=mu)

        if not os.path.exists(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/DIR/'):
            os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/DIR/')

        metrics = compute_metrics(queries_of_fold_struct.queries_IDs,
                              new_inverted_structure.document_IDs,
                              qrel,
                              baseline_model,
                              results_path + '/fold' + str(fold) + '/' + experiment_name + '/DIR/' + str(epoch))

        plot_values['DIR'][0].append(prop_elem_index)
        plot_values['DIR'][1].append(metrics)


    if IR_model == 'BM25':

        k1 = model.k1.numpy()
        b = model.b.numpy()

        print('BM25')

        baseline_model = baseline_models_and_tdv_implementation.Okapi_BM25(queries_of_fold_struct,new_inverted_structure,k1=k1,b=b)

        if not os.path.exists(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/BM25/'):
            os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/BM25/')

        metrics = compute_metrics(queries_of_fold_struct.queries_IDs,
                              new_inverted_structure.document_IDs,
                              qrel,
                              baseline_model,
                              results_path + '/fold' + str(fold) + '/' +  experiment_name + '/BM25/' + str(epoch))


        plot_values['BM25'][0].append(prop_elem_index)
        plot_values['BM25'][1].append(metrics)


    if IR_model == 'JM':
        lamb=model.lamb.numpy()

        print('JM')

        baseline_model = baseline_models_and_tdv_implementation.JM_language_model(queries_of_fold_struct,new_inverted_structure,Lambda=lamb)

        if not os.path.exists(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/JM/'):
            os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/JM/')

        metrics = compute_metrics(queries_of_fold_struct.queries_IDs,
                              new_inverted_structure.document_IDs,
                              qrel,
                              baseline_model,
                              results_path + '/fold' + str(fold) + '/' +  experiment_name + '/JM/' + str(epoch))

        plot_values['JM'][0].append(prop_elem_index)
        plot_values['JM'][1].append(metrics)



    pickle.dump(plot_values, open(plot_path + '/fold' + str(fold) + '/' +  experiment_name +'_epoch_'+str(epoch), 'wb'))

    # HR added this function to evaluate baseline models on TREC after training to get the TDV weights. It is a modified version of eval_learned_index in the original file. The calls for the function in other files were different from its definition. I added the JM model too. #HR

def eval_learned_index_wikir(coll_path,
                       Collection,
                       IR_model,
                       model,
                       validation_qrel,
                       test_qrel,
                       validation_plot_values,
                       test_plot_values,
                       plot_path,
                       inverted_index,
                       redefined_idf,
                       redefined_docs_length,
                       redefined_c_freq,
                       #                        idf,
                       #                        docs_length,
                       #                        c_freq,
                       prop_elem_index,
                       results_path,
                       experiment_name,
                       epoch):
    """Evaluate the performance of baseline models and their corresponding weighted (TDV) versions and saves the results in a pickle object file
    This is is to be used after training the neural model (calculating the TDV weights of terms) """ #HR

    if IR_model == 'tf':

        print('tf')
        #validation
        results = baseline_models_and_tdv_implementation.simple_tf(Collection.indexed_validation_queries,
                            inverted_index)

        if not os.path.exists(results_path + '/validation/' +  experiment_name + '/tf/'):
            os.makedirs(results_path + '/validation/' +  experiment_name + '/tf/')


        metrics = compute_metrics(coll_path,
                                  Collection,
                                  Collection.validation_queries_index,
                                  validation_qrel,
                                  results,
                                  results_path + '/validation/' +  experiment_name + '/tf/' + str(epoch))

        validaton_plot_values['tf'][0].append(prop_elem_index)
        validation_plot_values['tf'][1].append(metrics)

        #test
        results = baseline_models_and_tdv_implementation.simple_tf(Collection.indexed_test_queries,
                            inverted_index)

        if not os.path.exists(results_path + '/test/' +  experiment_name + '/tf/'):
            os.makedirs(results_path + '/test/' +  experiment_name + '/tf/')


        metrics = compute_metrics(coll_path,
                                  Collection,
                                  Collection.test_queries_index,
                                  test_qrel,
                                  results,
                                  results_path + '/test/' +  experiment_name + '/tf/' + str(epoch))

        test_plot_values['tf'][0].append(prop_elem_index)
        test_plot_values['tf'][1].append(metrics)



    if IR_model == 'tf_idf':

        print('tf_idf')
        #validation
        results = baseline_models_and_tdv_implementation.tf_idf(Collection.indexed_validation_queries,
                         inverted_index,
                         redefined_idf)

        if not os.path.exists(results_path + '/validation/' +  experiment_name + '/tf_idf/'):
            os.makedirs(results_path + '/validation/' +  experiment_name + '/tf_idf/')

        metrics = compute_metrics(coll_path,
                                  Collection,
                                  Collection.validation_queries_index,
                                  validation_qrel,
                                  results,
                                  results_path + '/validation/' +  experiment_name + '/tf_idf/' + str(epoch))

        validation_plot_values['tf_idf'][0].append(prop_elem_index)
        validation_plot_values['tf_idf'][1].append(metrics)

        #test
        results = baseline_models_and_tdv_implementation.tf_idf(Collection.indexed_test_queries,
                         inverted_index,
                         redefined_idf)

        if not os.path.exists(results_path + '/test/' +  experiment_name + '/tf_idf/'):
            os.makedirs(results_path + '/test/' +  experiment_name + '/tf_idf/')

        metrics = compute_metrics(coll_path,
                                  Collection,
                                  Collection.test_queries_index,
                                  test_qrel,
                                  results,
                                  results_path + '/test/' +  experiment_name + '/tf_idf/' + str(epoch))

        test_plot_values['tf_idf'][0].append(prop_elem_index)
        test_plot_values['tf_idf'][1].append(metrics)


    if IR_model == 'DIR':

        mu = model.mu.numpy()

        print('DIR')
        #validation
        results = baseline_models_and_tdv_implementation.dir_language_model(Collection.indexed_validation_queries,
                                     inverted_index,
                                     redefined_docs_length,
                                     redefined_c_freq,
                                     mu=mu)

        if not os.path.exists(results_path + '/validation/' +  experiment_name + '/DIR/'):
            os.makedirs(results_path + '/validation/' +  experiment_name + '/DIR/')

        metrics = compute_metrics(coll_path,
                                  Collection,
                                  Collection.validation_queries_index,
                                  validation_qrel,
                                  results,
                                  results_path + '/validation/' +  experiment_name + '/DIR/' + str(epoch))

        validation_plot_values['DIR'][0].append(prop_elem_index)
        validation_plot_values['DIR'][1].append(metrics)

        #test
        results = baseline_models_and_tdv_implementation.dir_language_model(Collection.indexed_test_queries,
                                     inverted_index,
                                     redefined_docs_length,
                                     redefined_c_freq,
                                     mu=mu)

        if not os.path.exists(results_path + '/test/' +  experiment_name + '/DIR/'):
            os.makedirs(results_path + '/test/' +  experiment_name + '/DIR/')

        metrics = compute_metrics(coll_path,
                                  Collection,
                                  Collection.test_queries_index,
                                  test_qrel,
                                  results,
                                  results_path + '/test/' +  experiment_name + '/DIR/' + str(epoch))

        test_plot_values['DIR'][0].append(prop_elem_index)
        test_plot_values['DIR'][1].append(metrics)



    if IR_model == 'BM25':

        k1 = model.k1.numpy()
        b = model.b.numpy()

        print('BM25')
        #validation
        results = baseline_models_and_tdv_implementation.Okapi_BM25(Collection.indexed_validation_queries,
                             inverted_index,
                             redefined_docs_length,
                             redefined_idf,
                             k1=k1,
                             b=b)

        if not os.path.exists(results_path + '/validation/' +  experiment_name + '/BM25/'):
            os.makedirs(results_path + '/validation/'  +  experiment_name + '/BM25/')

        metrics = compute_metrics(coll_path,
                                  Collection,
                                  Collection.validation_queries_index,
                                  validation_qrel,
                                  results,
                                  results_path + '/validation/'  +  experiment_name + '/BM25/' + str(epoch))

        validation_plot_values['BM25'][0].append(prop_elem_index)
        validation_plot_values['BM25'][1].append(metrics)

        #test
        results = baseline_models_and_tdv_implementation.Okapi_BM25(Collection.indexed_test_queries,
                             inverted_index,
                             redefined_docs_length,
                             redefined_idf,
                             k1=k1,
                             b=b)

        if not os.path.exists(results_path + '/test/' +  experiment_name + '/BM25/'):
            os.makedirs(results_path + '/test/'  +  experiment_name + '/BM25/')

        metrics = compute_metrics(coll_path,
                                  Collection,
                                  Collection.test_queries_index,
                                  test_qrel,
                                  results,
                                  results_path + '/test/'  +  experiment_name + '/BM25/' + str(epoch))

        test_plot_values['BM25'][0].append(prop_elem_index)
        test_plot_values['BM25'][1].append(metrics)


    if IR_model == 'JM':
        lamb=model.lamb.numpy()

        print('JM')
        #validation
        results = baseline_models_and_tdv_implementation.JM_language_model(Collection.indexed_validation_queries,
                             inverted_index,
                             redefined_docs_length,
                             redefined_c_freq,
                             lamb)

        if not os.path.exists(results_path + '/validation/' +  experiment_name + '/JM/'):
            os.makedirs(results_path + '/validation/' +  experiment_name + '/JM/')

        metrics = compute_metrics(coll_path,
                                  Collection,
                                  Collection.validation_queries_index,
                                  validation_qrel,
                                  results,
                                  results_path + '/validation/' +  experiment_name + '/JM/' + str(epoch))

        validation_plot_values['JM'][0].append(prop_elem_index)
        validation_plot_values['JM'][1].append(metrics)

        #test
        results = baseline_models_and_tdv_implementation.JM_language_model(Collection.indexed_test_queries,
                             inverted_index,
                             redefined_docs_length,
                             redefined_c_freq,
                             lamb)

        if not os.path.exists(results_path + '/test/' +  experiment_name + '/JM/'):
            os.makedirs(results_path + '/test/' +  experiment_name + '/JM/')

        metrics = compute_metrics(coll_path,
                                  Collection,
                                  Collection.test_queries_index,
                                  test_qrel,
                                  results,
                                  results_path + '/test/' +  experiment_name + '/JM/' + str(epoch))

        test_plot_values['JM'][0].append(prop_elem_index)
        test_plot_values['JM'][1].append(metrics)


    pickle.dump(validation_plot_values, open(plot_path + '/validation/' +  experiment_name, 'wb'))
    pickle.dump(test_plot_values, open(plot_path + '/test/' +  experiment_name, 'wb'))

    
#The function below is for testing purposes. Please delete it if it was not done
def eval_baseline_diff_index_trec(inverted_structure,
                        queries_of_fold_struct,
                        fold,
                        qrel,
                        plot_values,
                        results_path,
                        experiment_name,
                        epoch):
    """This function computes the metrics for the baseline models for term matching methods and
    updates the plot values dictionary for a certain fold and a certain epoch.This function is to be used on Trec collection """ #HR


    print('tf_idf')

    baseline_model = baseline_models_and_tdv_implementation.diff_tf_idf(queries_of_fold_struct,inverted_structure)

    if not os.path.exists(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/tf_idf/'):
        os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/tf_idf/')

    metrics = compute_metrics(queries_of_fold_struct.queries_IDs,
                              inverted_structure.document_IDs,
                              qrel,
                              baseline_model,
                              results_path + '/fold' + str(fold) + '/' +  experiment_name + '/tf_idf/' + str(epoch))

    plot_values['tf_idf'][0].append(1.0)
    plot_values['tf_idf'][1].append(metrics)

    print('DIR')

    baseline_model = baseline_models_and_tdv_implementation.diff_dir_language_model(queries_of_fold_struct,inverted_structure)

    if not os.path.exists(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/DIR/'):
        os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/DIR/')

    metrics = compute_metrics(queries_of_fold_struct.queries_IDs,
                              inverted_structure.document_IDs,
                              qrel,
                              baseline_model,
                              results_path + '/fold' + str(fold) + '/' + experiment_name + '/DIR/' + str(epoch))

    plot_values['DIR'][0].append(1.0)
    plot_values['DIR'][1].append(metrics)

    print('BM25')

    baseline_model = baseline_models_and_tdv_implementation.diff_Okapi_BM25(queries_of_fold_struct,inverted_structure)

    if not os.path.exists(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/BM25/'):
        os.makedirs(results_path + '/fold' + str(fold) + '/' +  experiment_name + '/BM25/')

    metrics = compute_metrics(queries_of_fold_struct.queries_IDs,
                              inverted_structure.document_IDs,
                              qrel,
                              baseline_model,
                              results_path + '/fold' + str(fold) + '/' +  experiment_name + '/BM25/' + str(epoch))

    plot_values['BM25'][0].append(1.0)
    plot_values['BM25'][1].append(metrics)


