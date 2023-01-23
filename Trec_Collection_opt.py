# =============================================================================
# Created By  : Jibril FREJJ
# Created Date: April 13 2020
# Modified By  : Hamdi REZGUI
# Modified Date: March 23 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Definition of some global methods and the TREC definition class
# with its internal methods
# =============================================================================
import os
import time
import pickle
import random
import string
import fasttext
import numpy as np
import pytrec_eval
import pandas as pd
import subprocess
import array as arr
import copy 

from nltk.stem import snowball
from collections import Counter
from nltk.corpus import stopwords

import std_tokenizer
import utils
from Inverted_structure import Inverted_structure
from Direct_structure import Direct_structure
from  Queries import Queries

####Useful methods for trec collection and trec collection class definition#### HR

def installFromOrigin(sourcePath: str,localPath: str, max_files: int = -1):
    """
    Install a collection from original sources
    Create a XML file documents.xml, into the localPath directory
    """
    # Number of files treated
    nbFilesTreated = 0;
    # Local file name
    localFileName = os.path.join(localPath,'documents.xml')
    fd = open(localFileName,'a')
    with os.scandir(sourcePath) as entries:
        for entry in entries:
            if (entry.name.endswith('.Z')):
                nbFilesTreated += 1
                path = os.path.join(sourcePath,entry.name)
                zcat = subprocess.Popen(['zcat', path], stdout=fd)
                if max_files > 0 and nbFilesTreated >= max_files:
                    break
    fd.close()


def build_folds(queries_ids, k=5):
    """Builds folds for the K-fold cross validation """ #HR
    nb_queries = len(queries_ids)
    nb_elem = int(nb_queries / k)
    random.shuffle(queries_ids)
    folds = []
    for i in range(k):
        folds.append(queries_ids[i * nb_elem:(i + 1) * nb_elem])
    return folds


def read_queries(queries_path):
    """ Function that reads the queries from a path of the file containing those queries. It returns  dict of
    query ids and query texts""" #HR
    queries_ids = []
    queries_text = []
    with open(queries_path, 'r') as f:
        for line in f:
            if line.startswith("<num>"):
                queries_ids.append(int(line[line.find(':') + 1:-1]))
            if line.startswith("<title>"):
                pos = max(line.find(':'), line.find('>'))
                queries_text.append("".join([char for char in line[pos + 2:-1] if char not in string.punctuation]))

    return dict(zip(queries_ids, queries_text))


def read_documents(documents_path):
    """Same function but for documents""" #HR
    doc_ids = []
    doc_text = ['']
    fill_text = False
    with open(documents_path, 'r', encoding='latin1') as f:
        # Read each line, i is the line number (not used)
        for i, line in enumerate(f):
            if "<DOCNO>" in line:
                doc_ids.append(line.strip("<DOCNO> ").strip(" </DOCNO>\n"))
            if "<TEXT>" in line:
                fill_text = True
                continue
            if "</TEXT>" in line:
                continue
            if "</DOC>" in line:
                doc_text.append('')
                fill_text = False
            elif fill_text:
                doc_text[-1] += line
    del doc_text[-1]

    for i in range(len(doc_text)):
        doc_text[i] = " ".join(doc_text[i].replace('\n', ' ').split())
        doc_text[i] = "".join(char for char in doc_text[i] if char not in string.punctuation)

    return dict(zip(doc_ids, doc_text))


def save_qrel(path, qrels, subset):
    """Function that saves query document relavence scores into a file.""" #HR
    with open(path, 'w') as f:
        for query in subset:
            for doc, rel in qrels[str(query)].items():
                f.write(str(query) + '\t0\t' + doc + '\t' + str(rel) + '\n')


def save_queries_csv(coll_path, queries, folds):
    """Function that saves folds of queries into csv files for each fold""" #HR
    for i, elem in enumerate(folds):
        index = pd.Index([key for key in elem], name='id_left')
        d = {"text_left": [queries[key] for key in elem]}
        pd.DataFrame(data=d, index=index).to_csv(coll_path + '/fold' + str(i) + '/queries.csv')


def save_documents_csv(coll_path, documents):
    """Function that saves documents into a csv file""" #HR
    index = pd.Index([key for key in documents], name='id_right')
    d = {"text_right": [documents[key] for key in documents]}
    pd.DataFrame(data=d, index=index).to_csv(coll_path + '/documents.csv')

#This function has been modified by HR. It used to process the three TREC collections at once. Now it is defined
# to handle one collection at a time
def read_collection(collection_path, k=5):
    """Function that for every TREC collection reads queries , create folds for the Kfold cross validation
    ,reads the collection qrels ,save qrels and queries for each fold,reads documents
    on xml format and saves them into csv format""" #HR

    queries = read_queries(collection_path + '/queries')

    folds = build_folds(list(queries.keys()), k=k)

    with open(collection_path + '/qrels', 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)

    for i, fold in enumerate(folds):
        if not os.path.exists(collection_path + '/fold' + str(i)):
            os.makedirs(collection_path + '/fold' + str(i))
        save_qrel(collection_path + '/fold' + str(i) + '/qrels', qrel, fold)

    save_queries_csv(collection_path, queries, folds)

    documents = read_documents(collection_path + '/documents.xml')

    save_documents_csv(collection_path, documents)


class TrecCollection:
    def __init__(self, k=5, language='english'):
        """
        documents: pandas.DataFrame, [docid] => text string
        k: number of folders for the queries
        """
        self.documents = None
        self.k = k
        self.language = language
        self.stemmer = snowball.EnglishStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.direct_structure=Direct_structure()
        self.inverted_structure=Inverted_structure()

    def load_documents(self,collection_path) -> int:
        """Function that loads the documents in csv format and returns the number of the documents"""
        self.documents = pd.read_csv(collection_path + '/documents.csv', index_col='id_right', na_filter=False)
        return self.documents.size
    
    def load_folds_and_training_qrel(self,collection_path):
        """
        It loads  the folds containing the queries per fold
        in Csv format,qrels per fold and the training qrels per fold . It is run after the function read_collection
        Return the number of documents found in the cvs file
        """ #HR
        self.folds_qrels = []
        self.folds_training_qrels = []
        for i in range(self.k):
            self.folds_qrels.append(utils.read_qrels(collection_path + '/fold' + str(i) + '/qrels'))
            self.folds_training_qrels.append(utils.read_trec_train_qrels(collection_path + '/fold' + str(i) + '/qrels'))
                          
    def load_folds_queries(self, collection_path):
        """
        Loads the collection : it loads  the folds containing the queries per fold
        in Csv format,qrels per fold and the training qrels per fold . It is run after the function read_collection
        Return the number of documents found in the cvs file
        """ #HR
        
        self.folds_queries = []
        for i in range(self.k):
            self.folds_queries.append(pd.read_csv(collection_path + '/fold' + str(i) + '/queries.csv',
                                                  index_col='id_left',
                                                  na_filter=False))

        
    def build_inverted_index_and_vocabulary(self, file_path,minimum_occurence=5,proportion_of_frequent_words=0.2):
        """Function that builds the inverted index and  the vocabulary"""
        inverted_structure = Inverted_structure()
        start=time.time()
        # Iterating over the documents
        for document_ID, document_text in self.documents.iterrows():
            inverted_structure.inverse_document(str(document_ID), document_text[0])
        end=time.time()
        number_of_documents=inverted_structure.get_number_of_documents()
        print("Average time to inverse documents TREC",round(((end-start)/number_of_documents)*1000), " ms",flush=True)
        #Filtering vocabulary ,posting lists,direct structure ,update documents' length and save direct structure
        if os.path.exists(file_path):
            start=time.time()
            inverted_structure.filter_vocabulary(file_path,minimum_occurence,proportion_of_frequent_words)  
            end=time.time()
            print("Average time to filter vocabulary,posting lists and update document lengths Trec and save direct structure",round(((end-start)/number_of_documents)*1000), " ms",flush=True)
        else:
            raise IOError('Path does not exist: %s' % file_path)
        
        #Saving inverted structure    

        start=time.time()
        inverted_structure.save(file_path)
        end=time.time()
        print("Saving time inverted structure TREC",round(end-start), " s",flush=True)
        return inverted_structure
    
    def load_inverted_structure(self,file_path):
        self.inverted_structure=Inverted_structure()
        self.inverted_structure.load(file_path)
    def load_direct_structure(self,file_path):
        self.direct_structure=Directed_structure()
        self.direct_structure.load(file_path)
    def process_queries(self,vocabulary, file_path=None, save=False):
        # Processing training queries
        self.folds_processed_queries=[]
        for i in range(self.k):
            processed_queries = Queries(vocabulary)
            for query_ID, query_text in self.folds_queries[i].iterrows():
                processed_queries.process_query_and_get_ID(query_ID, query_text[0])
            self.folds_processed_queries.append(processed_queries)
            if save and file_path != None:
                if os.path.exists(file_path):
                    processed_queries.save(file_path+"fold"+str(i))
                else:
                    raise IOError('Path does not exist: %s' % file_path)
        return self.folds_processed_queries
    def get_all_bow_queries_and_internal_query_IDs(self):
        """Function that collects all direct queries , the access table for general internal query ID in all the folds
        """
        self.all_bow_indices_queries=[]
        self.all_bow_frequencies_queries=[]
        #Dictionary for each external query ID is associated an internal token ID defined for all the queries in the database
        self.all_internal_query_IDs=dict()
        index=0
        for queries_struct in self.folds_processed_queries:
            #Collecting all direct queries of the K folds
            self.all_bow_indices_queries+=queries_struct.bow_indices_queries
            self.all_bow_frequencies_queries+=queries_struct.bow_frequencies_queries
            #Collecting all external query IDs and their updated internal query ID for all queries in the database
            for query_external_ID in queries_struct.queries_internal_IDs:
                self.all_internal_query_IDs[query_external_ID]=index
                index+=1
    #To modify to fit the new inverted structure
    def generate_training_batches(self, fold, batch_size=32):
        """Function that builds batches of queries and their corresponding negative and positive documents
        for training for a particular fold. These batches are picked from outside the fold we want to
        test or validate on""" #HR
        positive_pairs = []
        negative_pairs = {}
        for i in range(self.k):
            if i != fold:
                for elem in self.folds_training_qrels[i]['pos']:
                    #Some documents in TREC collection are missing but they appear in the qrel. That's why we added this condition. elem is a list composed of query external ID and document external ID
                    if elem[1] in self.inverted_structure.document_internal_IDs:
                        positive_pairs.append(elem)
                    #Same reason we added this condition here
                for query in self.folds_training_qrels[i]['neg']:
                    #temporary list containing all the negative documents to the query that exist in the collection of documents
                    tmp_list=[]
                    for elem in self.folds_training_qrels[i]['neg'][query]:
                        if elem in self.inverted_structure.document_internal_IDs:
                            tmp_list.append(elem)
                    negative_pairs[query]=tmp_list


        random.shuffle(positive_pairs)
        nb_docs = len(self.direct_structure.documents)
        nb_train_pairs = len(positive_pairs)
        print("Number of documents=",nb_docs,flush=True)
        print("Number of training pairs=",nb_train_pairs,flush=True)
        
        pos = 0
        while (pos + batch_size < nb_train_pairs):
            query_batch = arr.array('I')
            positive_doc_batch = arr.array('I')
            negative_doc_batch = arr.array('I')
            for q,d in positive_pairs[pos:pos+batch_size]:
                query_batch.append(self.all_internal_query_IDs[q])
                positive_doc_batch.append(self.inverted_structure.document_internal_IDs[d])
                negative_doc_batch.append(self.inverted_structure.document_internal_IDs[random.choice(negative_pairs[q])])
            yield query_batch, positive_doc_batch, negative_doc_batch
            pos += batch_size
    

            
    def reduce_inverted_structure_and_apply_weights(self,weights):
        """Creates a new inverted structure where the tokens with weights equal to 0 are erased 
        weights: are the TDV weights we get from training"""
        new_token_ID=1
        #new inverted structure to build
        new_inverted_structure=Inverted_structure()
        #the deep copy is necessary so that changing the documents_length in the new inverted structure does not occur in the original documents_length
        new_inverted_structure.documents_length=copy.deepcopy(self.inverted_structure.documents_length)
        new_inverted_structure.document_IDs=self.inverted_structure.document_IDs
        for token in self.inverted_structure.token():
            #internal token id in the vocabulary of the original inverted structure
            old_token_ID=self.inverted_structure.vocabulary[token][1]
            #length of posting list of the token in the original inverted structure
            old_token_length_of_posting_list=self.inverted_structure.vocabulary[token][0]
            if weights[old_token_ID-1]==0:
                #Updating documents length by substracting the frequency of the word from each document it appears in
                for i in range(old_token_length_of_posting_list):
                    doc_ID=self.inverted_structure.posting_lists[old_token_ID-1][2*i]
                    frequency=self.inverted_structure.posting_lists[old_token_ID-1][2*i+1]
                    new_inverted_structure.documents_length[doc_ID]-=frequency
            else:
                #The posting list length of token does not change. We update the token id in the new inverted structure 
                new_inverted_structure.vocabulary[token]=[old_token_length_of_posting_list,new_token_ID]
                new_token_ID+=1
                #Building the new posting lists of the new inverted structure and apply the weights. The new signal is the 
                #product of the weight and the frequency
                new_posting_list=arr.array('f')
                for i in range(old_token_length_of_posting_list):
                    doc_ID=self.inverted_structure.posting_lists[old_token_ID-1][2*i]
                    new_posting_list.append(doc_ID)
                    frequency=self.inverted_structure.posting_lists[old_token_ID-1][2*i+1]
                    new_posting_list.append(frequency*weights[old_token_ID-1])                    
                new_inverted_structure.posting_lists.append(new_posting_list)                
        return new_inverted_structure