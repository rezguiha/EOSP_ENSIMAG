# =============================================================================
# Created By  : Hamdi REZGUI
# Created Date: March 29 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Code to Evaluate baseline IR models before training for TDV
# for wikIR or wikIRS collections
# =============================================================================
#Packages
import os
import pickle
import argparse
import pytrec_eval
import resource
import time
#Code files
# import utils
import wikIR_Collection
import baseline_models_and_tdv_implementation


def compute_metrics(coll_path, Collection, queries_index, qrel, results, model_name, save_res=False):
    """Function that saves the results of retrieval: the top_k documents according to their score for
    a certain model identified by model_name. Then, it computes different metrics for IR using the pytrec_eval
    package""" #HR
    Collection.save_results(queries_index, results, model_name, top_k=1000)

    with open(model_name, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)
    if not save_res:
        os.remove(model_name)

    measures = {"map", "ndcg_cut", "recall", "P"}

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, measures)

    all_metrics = evaluator.evaluate(run)

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

def eval_baseline_index_wikir(coll_path,
                        Collection,
                        validation_qrel,
                        test_qrel,
                        validation_plot_values,
                        test_plot_values,
                        results_path,
                        experiment_name,
                        epoch):
    """This function computes the metrics for the baseline models for term matching methods and
    updates the plot values dictionary for a certain fold and a certain epoch.This function is to be used on Trec collection """ #HR
    print('----------------------tf----------------------',flush=True)
    start=time.time()
    #validation
    start2=time.time()
    results = baseline_models_and_tdv_implementation.simple_tf(Collection.indexed_validation_queries,
                                  Collection.inverted_index)
    end2=time.time()
    print("memory used TF validation WikIR after results computing",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
    print("time elapsed TF validation WikIR after results computing ",end2-start2,flush=True)
    
    if not os.path.exists(results_path + '/validation/' + experiment_name + '/tf/'):
        os.makedirs(results_path + '/validation/' + experiment_name + '/tf/')

    metrics = compute_metrics(coll_path,
                              Collection,
                              Collection.validation_queries_index,
                              validation_qrel,
                              results,
                              results_path + '/validation/' + experiment_name + '/tf/' + str(epoch))
    print("memory used TF validation WikIR after metrics computing",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
    print("time elapsed TF validation WikIR after metrics computing ",time.time()-end2,flush=True)
    
    validation_plot_values['tf'][0].append(1.0)
    validation_plot_values['tf'][1].append(metrics)
    print("End evaluating TF validation WikIR collection", flush=True)
    end=time.time()
    print("time elapsed TF validation WikIR collection ", end-start,flush=True)
    print("memory used TF validation WikIR collection ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
    #Test
    start=time.time()
    results = baseline_models_and_tdv_implementation.simple_tf(Collection.indexed_test_queries,
                                  Collection.inverted_index)

    if not os.path.exists(results_path + '/test/' + experiment_name + '/tf/'):
        os.makedirs(results_path + '/test/' + experiment_name + '/tf/')

    metrics = compute_metrics(coll_path,
                              Collection,
                              Collection.test_queries_index,
                              test_qrel,
                              results,
                              results_path + '/test/' + experiment_name + '/tf/' + str(epoch))

    test_plot_values['tf'][0].append(1.0)
    test_plot_values['tf'][1].append(metrics)
    
    print("End evaluating TF test WikIR collection", flush=True)
    end=time.time()
    print("time elapsed TF test WikIR collection ", end-start,flush=True)
    print("memory used TF test WikIR collection ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
    
    print('------------------------tf_idf--------------------',flush=True)
    #validation
    start=time.time()
    results = baseline_models_and_tdv_implementation.tf_idf(Collection.indexed_validation_queries,
                     Collection.inverted_index,
                     Collection.idf)

    if not os.path.exists(results_path + '/validation/' +  experiment_name + '/tf_idf/'):
        os.makedirs(results_path + '/validation/' +  experiment_name + '/tf_idf/')

    metrics = compute_metrics(coll_path,
                              Collection,
                              Collection.validation_queries_index,
                              validation_qrel,
                              results,
                              results_path + '/validation/' +  experiment_name + '/tf_idf/' + str(epoch))

    validation_plot_values['tf_idf'][0].append(1.0)
    validation_plot_values['tf_idf'][1].append(metrics)
    
    print("End evaluating TF-IDF validation WikIR collection", flush=True)
    end=time.time()
    print("time elapsed TF-IDF validation WikIR collection ", end-start,flush=True)    
    print("memory used TF-IDF validation WikIR collection ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
    
    #test
    start=time.time()
    results = baseline_models_and_tdv_implementation.tf_idf(Collection.indexed_test_queries,
                     Collection.inverted_index,
                     Collection.idf)

    if not os.path.exists(results_path + '/test/' +  experiment_name + '/tf_idf/'):
        os.makedirs(results_path + '/test/' +  experiment_name + '/tf_idf/')

    metrics = compute_metrics(coll_path,
                              Collection,
                              Collection.test_queries_index,
                              test_qrel,
                              results,
                              results_path + '/test/' +  experiment_name + '/tf_idf/' + str(epoch))

    test_plot_values['tf_idf'][0].append(1.0)
    test_plot_values['tf_idf'][1].append(metrics)

    print("End evaluating TF-IDF test WikIR collection", flush=True)
    end=time.time()
    print("time elapsed TF-IDF test WikIR collection ", end-start,flush=True)    
    print("memory used TF-IDF test WikIR collection ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
  
    print('-----------------------------DIR-----------------------------',flush=True)
    #validation
    start=time.time()
    
    results = baseline_models_and_tdv_implementation.dir_language_model(Collection.indexed_validation_queries,
                                 Collection.inverted_index,
                                 Collection.docs_length,
                                 Collection.c_freq)

    if not os.path.exists(results_path + '/validation/' +  experiment_name + '/DIR/'):
        os.makedirs(results_path + '/validation/' +  experiment_name + '/DIR/')

    metrics = compute_metrics(coll_path,
                              Collection,
                              Collection.validation_queries_index,
                              validation_qrel,
                              results,
                              results_path + '/validation/' + experiment_name + '/DIR/' + str(epoch))

    validation_plot_values['DIR'][0].append(1.0)
    validation_plot_values['DIR'][1].append(metrics)

    
    
    print("End evaluating DIR validation WikIR collection", flush=True)
    end=time.time()
    print("time elapsed DIR validation WikIR collection ", end-start,flush=True)    
    print("memory used DIR validation WikIR collection ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
    #test
    start=time.time()
    results = baseline_models_and_tdv_implementation.dir_language_model(Collection.indexed_test_queries,
                                 Collection.inverted_index,
                                 Collection.docs_length,
                                 Collection.c_freq)

    if not os.path.exists(results_path + '/test/' +  experiment_name + '/DIR/'):
        os.makedirs(results_path + '/test/' +  experiment_name + '/DIR/')

    metrics = compute_metrics(coll_path,
                              Collection,
                              Collection.test_queries_index,
                              test_qrel,
                              results,
                              results_path + '/test/' + experiment_name + '/DIR/' + str(epoch))

    test_plot_values['DIR'][0].append(1.0)
    test_plot_values['DIR'][1].append(metrics)
    
    print("End evaluating DIR test WikIR collection", flush=True)
    end=time.time()
    print("time elapsed DIR test WikIR collection ", end-start,flush=True)    
    print("memory used DIR test WikIR collection ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
    
    
    print('--------------------------BM25------------------------',flush=True)
    #validation
    start=time.time()
    results = baseline_models_and_tdv_implementation.Okapi_BM25(Collection.indexed_validation_queries,
                         Collection.inverted_index,
                         Collection.docs_length,
                         Collection.idf)

    if not os.path.exists(results_path + '/validation/' +  experiment_name + '/BM25/'):
        os.makedirs(results_path + '/validation/' +  experiment_name + '/BM25/')

    metrics = compute_metrics(coll_path,
                              Collection,
                              Collection.validation_queries_index,
                              validation_qrel,
                              results,
                              results_path + '/validation/' +  experiment_name + '/BM25/' + str(epoch))

    validation_plot_values['BM25'][0].append(1.0)
    validation_plot_values['BM25'][1].append(metrics)

    print("End evaluating BM25 validation WikIR collection", flush=True)
    end=time.time()
    print("time elapsed BM25 validation WikIR collection ", end-start,flush=True)    
    print("memory used BM25 validation WikIR collection ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
    #test
    start=time.time()
    results = baseline_models_and_tdv_implementation.Okapi_BM25(Collection.indexed_test_queries,
                         Collection.inverted_index,
                         Collection.docs_length,
                         Collection.idf)

    if not os.path.exists(results_path + '/test/' +  experiment_name + '/BM25/'):
        os.makedirs(results_path + '/test/' +  experiment_name + '/BM25/')

    metrics = compute_metrics(coll_path,
                              Collection,
                              Collection.test_queries_index,
                              test_qrel,
                              results,
                              results_path + '/test/' +  experiment_name + '/BM25/' + str(epoch))

    test_plot_values['BM25'][0].append(1.0)
    test_plot_values['BM25'][1].append(metrics)
    
    print("End evaluating BM25 test WikIR collection", flush=True)
    end=time.time()
    print("time elapsed BM25 test WikIR collection ", end-start,flush=True)    
    print("memory used BM25 test WikIR collection ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
    
    
    print('----------------------------JM---------------------------',flush=True)
    #validation
    start=time.time()
    results = baseline_models_and_tdv_implementation.JM_language_model(Collection.indexed_validation_queries,
                         Collection.inverted_index,
                         Collection.docs_length,
                         Collection.c_freq)

    if not os.path.exists(results_path + '/validation/' +  experiment_name + '/JM/'):
        os.makedirs(results_path + '/validation/' +  experiment_name + '/JM/')

    metrics = compute_metrics(coll_path,
                              Collection,
                              Collection.validation_queries_index,
                              validation_qrel,
                              results,
                              results_path + '/validation/' +  experiment_name + '/JM/' + str(epoch))

    validation_plot_values['JM'][0].append(1.0)
    validation_plot_values['JM'][1].append(metrics)

    
    print("End evaluating JM validation WikIR collection", flush=True)
    end=time.time()
    print("time elapsed JM validation WikIR collection ", end-start,flush=True)    
    print("memory used JM validation  WikIR collection ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
    #test
    start=time.time()
    results = baseline_models_and_tdv_implementation.JM_language_model(Collection.indexed_test_queries,
                         Collection.inverted_index,
                         Collection.docs_length,
                         Collection.c_freq)

    if not os.path.exists(results_path + '/test/' +  experiment_name + '/JM/'):
        os.makedirs(results_path + '/test/' +  experiment_name + '/JM/')

    metrics = compute_metrics(coll_path,
                              Collection,
                              Collection.test_queries_index,
                              test_qrel,
                              results,
                              results_path + '/test/' +  experiment_name + '/JM/' + str(epoch))

    test_plot_values['JM'][0].append(1.0)
    test_plot_values['JM'][1].append(metrics)

    print("End evaluating JM test WikIR collection", flush=True)
    end=time.time()
    print("time elapsed JM test WikIR collection ", end-start,flush=True)    
    print("memory used JM test WikIR collection ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--coll_path', nargs="?", type=str)
    parser.add_argument('-i', '--indexed_path', nargs="?", type=str)
    parser.add_argument('-p', '--plot_path', nargs="?", type=str)
    parser.add_argument('-r', '--results_path', nargs="?", type=str)
    parser.add_argument('-n', '--experiment_name', nargs="?", type=str)

    args = parser.parse_args()

    print(args, flush=True)

    if not os.path.exists(args.results_path + '/validation/' + args.experiment_name):
        os.makedirs(args.results_path + '/validation/' + args.experiment_name)

    if not os.path.exists(args.results_path + '/test/' + args.experiment_name):
        os.makedirs(args.results_path + '/test/' + args.experiment_name)


    if not os.path.exists(args.plot_path + '/validation/'):
        os.makedirs(args.plot_path + '/validation/')

    if not os.path.exists(args.plot_path + '/test/'):
        os.makedirs(args.plot_path + '/test/')

    #Initializing the results plot values for the different models #HR
    validation_plot_values = dict()
    test_plot_values = dict()

    for model_name in ['tf',
                       'tf_idf',
                       'DIR',
                       'BM25',
                        'JM']:
        validation_plot_values[model_name] = [[], []]
        test_plot_values[model_name] = [[], []]

    #Loading indexed collection #HR
    start=time.time()
    print("Start loading indexed WikIR collection", flush=True)
    print("memory used at start loading indexed WikIR collection ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
    Collection = wikIR_Collection.Collection()
    with open(args.indexed_path, 'rb') as f:
        Collection = pickle.load(f)
    
    end=time.time()
    print("End loading indexed WikIR collection", flush=True)
    print("Time elapse loading indexed WikIR collection ",end-start,flush=True)
    print("Memory usage loading indexed WikIR collection ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
    
    Collection.doc_index[-1] = "-1"
    Collection.doc_index["-1"] = -1

    #Loading validation and test query relavance values #HR
    start=time.time()
    print("Start loading validation and test qrels", flush=True)
    print("memory used at start loading validation and test qrels ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
    
    with open(args.coll_path + 'validation/qrels', 'r') as f_qrel:
        validation_qrel = pytrec_eval.parse_qrel(f_qrel)

    with open(args.coll_path + 'test/qrels', 'r') as f_qrel:
        test_qrel = pytrec_eval.parse_qrel(f_qrel)

    end=time.time()
    print("End loading validation and test qrels WikIR collection", flush=True)
    print("Time elapse loading validation and test qrels WikIR collection ",end-start,flush=True)
    print("Memory usage loading validation and test qrels WikIR collection ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
    
    print('------------------------------start--------------------------',flush=True)

    #Evaluating the baseline models without TDV weights and saving the results of validation and test partitions #HR
    start=time.time()
    print("Start evaluating baseline models WikIR collection", flush=True)
    
    eval_baseline_index_wikir(args.coll_path,
                              Collection,
                              validation_qrel,
                              test_qrel,
                              validation_plot_values,
                              test_plot_values,
                              args.results_path,
                              args.experiment_name,
                              0)
    print("End evaluating baseline models WikIR collection", flush=True)
    end=time.time()
    print("Time elapsed evaluating baseline models WikIR collection ", end-start,flush=True)
    print("Memory used at end of evaluating baseline models WikIR collection ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
    #Printing ndcg5 values for validation and test partitions #HR
    start=time.time()
    ndcg5_val = dict()
    ndcg5_test = dict()
    for model_name in ['tf',
                       'tf_idf',
                       'DIR',
                       'BM25', 'JM']:
        ndcg5_val[model_name] = validation_plot_values[model_name][1][0]['ndcg_cut_5']
        print("ndcg5 validation ",model_name," of collection ",os.path.basename(args.coll_path)," ",ndcg5_val[model_name] ,flush=True)
        ndcg5_test[model_name] = test_plot_values[model_name][1][0]['ndcg_cut_5']
        print("ndcg5 test ",model_name," of collection ",os.path.basename(args.coll_path)," ",ndcg5_test[model_name] ,flush=True)

    print("End printing baseline models performance WikIR collection", flush=True)
    end=time.time()
    print("Elapsed time printing baseline models performance WikIR collection ", end-start, flush=True)
    print("Memory used at end of printing baseline models performance WikIR collection  ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
if __name__ == "__main__":
    main()