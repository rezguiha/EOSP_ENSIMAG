# =============================================================================
# Created By  : Jibril FREJJ
# Created Date: February 16 2020
# Modified By  : Hamdi REZGUI
# Modified Date: March 21 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Code to evaluate baseline IR models without training , to do training
# to get the TDV weights for a certain differentiable model and to evaluate baseline
# IR models after taking into account the TDVs for a TREC Collection
# =============================================================================

import os
import pickle
import random
import string
import fasttext
import numpy as np
import pytrec_eval
import pandas as pd
from nltk.stem import snowball
from collections import Counter
from nltk.corpus import stopwords
from numpy.random import seed

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers


import re
import pickle
import argparse
import importlib
import pytrec_eval


import differentiable_models2
import baseline_models_and_tdv_implementation
from Trec_Collection_opt import TrecCollection
import utils
###############Training on Trec collections############### #HR



#### Training and getting results for Trec collection#### #HR
def building_batch_documents(indices_bow,frequencies_bow,batch):
    """Function that builds the indices for the sparse tensor of the batch , the sequence of documents  bow representation in a vector shape containing all the token IDs combined  and the corresponding frequency sequences 
    indices_bow: is the list of all documents and each document is a list of unique token IDs 
    frequencies_bow: is the list of all documents and each document is a list of  corresponding frequencies to the token IDs
    batch: list of document internal IDs
    Warning: indices_bow and frequencies bow must have the same length
    Warning: elements of batch which are internal document ID must be smaller than the length of indices_bow and frequencies_bow"""
    #First element processing. This is necessary so we can use 
    indices_sparse_tensor = [[token_ID,0] for token_ID in indices_bow[batch[0]]]    
    indices_bow_documents_np_array=np.array(indices_bow[batch[0]],dtype=np.int32)
    frequencies_bow_documents_np_array=np.array(frequencies_bow[batch[0]],dtype=np.float32)
    for i in range(1,len(batch)):
        indices_bow_documents_np_array=np.append(indices_bow_documents_np_array,indices_bow[batch[i]])
        frequencies_bow_documents_np_array=np.append(frequencies_bow_documents_np_array,frequencies_bow[batch[i]])
        for token_ID in indices_bow[batch[i]]:
            indices_sparse_tensor.append([token_ID,i])
    
    return indices_sparse_tensor,indices_bow_documents_np_array,frequencies_bow_documents_np_array
def building_batch_queries(indices_bow,frequencies_bow,batch):
    """Function that builds the indices for the sparse tensor of the batch , the sequence of documents or queries bow representation in a vector shape containing all frequencies of tokens combined """
    #First element processing. This is necessary so we can use 
    indices_sparse_tensor = [[token_ID,0] for token_ID in indices_bow[batch[0]]]    
    frequencies_bow_queries_np_array=np.array(frequencies_bow[batch[0]],dtype=np.float32)
    for i in range(1,len(batch)):
        frequencies_bow_queries_np_array=np.append(frequencies_bow_queries_np_array,frequencies_bow[batch[i]])
        for token_ID in indices_bow[batch[i]]:
            indices_sparse_tensor.append([token_ID,i])
    return indices_sparse_tensor,frequencies_bow_queries_np_array
def main():
    #fixing the random number generator
#     random.seed(10)
#     seed(10)
#     tf.random.set_seed(10)
    #Selecting device 1 to be used since device 0 is busy in this cas
#     os.environ["CUDA_VISIBLE_DEVICES"]="1"
    #Limiting GPU memory use to 4GB otherwise tensorflow will
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
    #parsing arguments #HR
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--coll_path', nargs="?", type=str)
    parser.add_argument('-i', '--indexed_path', nargs="?", type=str)
    parser.add_argument('-f', '--fasttext_path', nargs="?", type=str)
    parser.add_argument('-p', '--plot_path', nargs="?", type=str)
    parser.add_argument('-r', '--results_path', nargs="?", type=str)
    parser.add_argument('-w', '--weights_path', nargs="?", type=str)
    parser.add_argument('-fo', '--folds', nargs="?", type=int, default=5)
    parser.add_argument('-e', '--nb_epoch', nargs="?", type=int)
    parser.add_argument('-l', '--l1_weight', nargs="?", type=float)
    parser.add_argument('-d', '--dropout_rate', nargs="?", type=float, default=0.0)
    parser.add_argument('--lr', nargs="?", type=float)
    parser.add_argument('-n', '--experiment_name', nargs="?", type=str)
    parser.add_argument('--IR_model', nargs="?", type=str, default='tf')
    parser.add_argument('-u', '--update_embeddings', action="store_true")

    args = parser.parse_args()

    print(args, flush=True)
    start0=time.time()
    # Loading indexed collection #HR
    Collection = TrecCollection()
    Collection.load_inverted_structure(args.indexed_path)
    Collection.inverted_structure.compute_fasttext_embeddings(args.fasttext_path)
    #Loading the Bag of words structure #HR
    Collection.direct_structure.load_bow(args.indexed_path)
    #Loading and processing queries 
    Collection.load_folds_queries(args.coll_path)
    folds_processed_queries=Collection.process_queries(vocabulary=Collection.inverted_structure.vocabulary)
    #Loading qrel for folds and for training
    Collection.load_folds_and_training_qrel(args.coll_path)
    #Collection bow representation for all queries and assigning and internal query ID for each query
    Collection.get_all_bow_queries_and_internal_query_IDs()
    
    # Loading relevance judgements from collection #HR
    with open(args.coll_path + 'qrels', 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)


    print('---------------------start-------------------',flush=True)
    
    # Getting collection vocabulary size and total number of elements in collection #HR
    coll_vocab_size=Collection.inverted_structure.get_vocabulary_size()
    coll_tot_nb_elem = sum(Collection.inverted_structure.documents_length)
    # Creating for each fold and for a certain experiment a directory for results,weights and plots data #HR

    start=time.time()
    print("Time for preliminary work before training",round(start-start0),flush=True)
    for fold in range(args.folds):

        plot_values = dict()

        for model_name in ['tf_idf',
                           'DIR',
                           'BM25','JM']:
            plot_values[model_name] = [[], [],[],[],[],[],[],[]]

        if not os.path.exists(args.results_path + '/fold' + str(fold) + '/' + args.experiment_name):
            os.makedirs(args.results_path + '/fold' + str(fold) + '/' + args.experiment_name)

        if not os.path.exists(args.weights_path + '/fold' + str(fold) + '/' + args.experiment_name):
            os.makedirs(args.weights_path + '/fold' + str(fold) + '/' + args.experiment_name)

        if not os.path.exists(args.plot_path + '/fold' + str(fold) + '/'):
            os.makedirs(args.plot_path + '/fold' + str(fold) + '/')

        # Initialization of batch size, the loss function,te optimizer and the model to train #HR
        batch_gen_time = []
        batch_size = 32
        y_true = tf.ones(batch_size, )
        loss_function = tf.keras.losses.Hinge()
        optimizer = tf.keras.optimizers.Adam(args.lr)

        if args.IR_model == 'tf_idf':
            model = differentiable_models2.diff_TF_IDF(Collection.inverted_structure.embedding_matrix, dropout_rate=args.dropout_rate)

        elif args.IR_model == 'DIR':
            model = differentiable_models2.diff_DIR(Collection.inverted_structure.embedding_matrix, dropout_rate=args.dropout_rate)

        elif args.IR_model == 'BM25':
            model = differentiable_models2.diff_BM25(Collection.inverted_structure.embedding_matrix, dropout_rate=args.dropout_rate)
        #HR added JM model
        elif args.IR_model == 'JM':
            model = differentiable_models2.diff_JM(Collection.inverted_structure.embedding_matrix, dropout_rate=args.dropout_rate)

        # Training the model #HR
        print("Start training for fold ", fold, " ", args.experiment_name, flush=True)
        epoch = 0
        prop_elem_index = 1.0
        while epoch < args.nb_epoch and prop_elem_index > 0.05:
            begin=time.time()

            rank_loss = 0.0
            reg_loss = 0.0
            all_non_zero = 0.0

            number_of_mismatch_relevance=0
            number_of_match_relevance=0
            #Iterating using a generator that yields batches for query, positive documents and negative documents
            for query_batch, positive_doc_batch, negative_doc_batch in Collection.generate_training_batches(fold,batch_size):

                with tf.GradientTape() as tape:
                    """Positive document batch"""
                    pos_doc_indices_sparse_tensor,pos_doc_indices_bow_vector_shape,pos_doc_frequencies_bow_vector_shape=building_batch_documents(Collection.direct_structure.indices_bow,Collection.direct_structure.frequency_bow,positive_doc_batch)
                        
                    """Negative document batch"""
                    neg_doc_indices_sparse_tensor,neg_doc_indices_bow_vector_shape,neg_doc_frequencies_bow_vector_shape=building_batch_documents(Collection.direct_structure.indices_bow,Collection.direct_structure.frequency_bow,negative_doc_batch)
                                          
                    """Queries"""
                    query_indices_sparse_tensor,query_frequencies_bow_vector_shape=building_batch_queries(Collection.all_bow_indices_queries,Collection.all_bow_frequencies_queries,query_batch)
                    

                    pos_res, pos_d = model(query_indices_sparse_tensor,
                                           query_frequencies_bow_vector_shape,
                                           pos_doc_indices_sparse_tensor,
                                           pos_doc_indices_bow_vector_shape,
                                           pos_doc_frequencies_bow_vector_shape,
                                           batch_size)

                    neg_res, neg_d = model(query_indices_sparse_tensor,
                                           query_frequencies_bow_vector_shape,
                                           neg_doc_indices_sparse_tensor,
                                           neg_doc_indices_bow_vector_shape,
                                           neg_doc_frequencies_bow_vector_shape,
                                           batch_size)
                    
                    plot_values[args.IR_model][4].append(pos_res[0].numpy())
                    plot_values[args.IR_model][5].append(neg_res[0].numpy())
                    
                    #Testing how many times we got a mismatch in the positive and negative relavance score
                    for i in range(batch_size):
                        if neg_res[i]>pos_res[i]:
                            number_of_mismatch_relevance+=1
                        else:
                            number_of_match_relevance+=1
                    # Computing the hinge loss and the regularization loss and total loss #HR
                    ranking_loss = loss_function(y_true=y_true, y_pred=pos_res - neg_res)

                    regularization_loss = tf.norm(pos_d + neg_d, ord=1)

                    rank_loss += ranking_loss.numpy()
                    reg_loss += regularization_loss.numpy()

                    all_non_zero += tf.math.count_nonzero(pos_d + neg_d).numpy()

                    loss = (1.0 - args.l1_weight) * ranking_loss + args.l1_weight * regularization_loss
                    # Calculating gradients #HR
                    
                    gradients = tape.gradient(loss, model.trainable_variables)

                # Back propagating the gradients #HR
                plot_values[args.IR_model][7].append(gradients[:300])
                
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                plot_values[args.IR_model][6].append(model.linear.get_weights())
            print("Number of mismatch relevance for epoch ",epoch,"=",number_of_mismatch_relevance,flush=True)
            print("Number of match relevance for epoch",epoch,"=",number_of_match_relevance,flush=True)
            #Getting values of the losses after every epoch
            plot_values[args.IR_model][2].append(rank_loss)
            plot_values[args.IR_model][3].append(regularization_loss)
            ending=time.time()
            print("Time for pure training for fold ",fold, "and epoch",epoch," =", round(ending-begin),flush=True)
            # Compute the TDVs after the training and saving them #HR
            weights = model.compute_index()

            pickle.dump(weights, open(
                args.weights_path + '/fold' + str(fold) + '/' +  args.experiment_name + '/epoch_' + str(epoch), 'wb'))
            tik=time.time()
            new_inverted_structure=Collection.reduce_inverted_structure_and_apply_weights(weights)
            tok=time.time()
            print("Time to reduce inverted structure and apply weights",round(tok-tik),flush=True)
            new_inverted_structure.compute_idf()
            new_inverted_structure.compute_collection_frequencies()
            

            # Computing new vocab_size and total number of elements after introducting the TDV #HR
            new_vocab_size=new_inverted_structure.get_vocabulary_size()
            new_tot_nb_elem = sum(new_inverted_structure.documents_length)

            print(str(100 * new_vocab_size / coll_vocab_size)[0:5] + '% of the vocabulary is kept',flush=True)
            print(str(100 * new_tot_nb_elem / coll_tot_nb_elem)[0:5] + '% of the index is kept', flush=True)

            prop_elem_index = new_tot_nb_elem / coll_tot_nb_elem
            #Evaluating baseline models with their new inverted index and new idf, doc length and collection frequencies #HR
                #HR modified the eval_learned_index to a function eval_learned_index_trec
    # The previous version did not work because of a different call parameters
            tik=time.time()
            utils.eval_learned_index_trec(new_inverted_structure,
                            folds_processed_queries[fold],
                            fold,
                            qrel,
                            plot_values,
                            args.results_path,
                            args.experiment_name,
                            epoch+1,
                            args.IR_model,
                            args.plot_path,
                            prop_elem_index,
                            model)
            tok=time.time()
            print("Time to evaluate the model", round(tok-tik),flush=True)
            #Saving the trained model model for the particular fold
            model.save_weights(args.weights_path + '/fold' + str(fold) + '/' +  args.experiment_name + '/model_weights_epoch_' + str(epoch))
            epoch += 1

        print("finish training for fold ", fold, " ", args.experiment_name, flush=True) #HR

    end=time.time()
    print("-----------------Finished-------------------",flush=True) #HR
    print("Time for training =",round(end-start),flush=True)
    print("Total time =", round(end-start0),flush=True)
if __name__ == "__main__":
    main()
