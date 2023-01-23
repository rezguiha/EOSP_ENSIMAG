#packages
import pickle
from nltk.stem import snowball
from nltk.corpus import stopwords
import array as arr
from collections import Counter
### Definition of class queries
class Queries:
    def __init__(self,vocabulary):
        # External queries id, the position in this list is internal doc Id
        self.queries_IDs=[]
        #Internal queries id
        self.queries_internal_IDs=dict()
        # A list containing the processed queries which are a list of tokens
        self.queries=[]
        # Stemmer and stop words
        self.stemmer = snowball.EnglishStemmer()
        self.stop_words = set(stopwords.words('english'))
        #vocabulary is the structure that we build in inverted structure
        self.vocabulary=vocabulary
        #direct structure for queries
        self.direct_queries=[]
        #bow structure
        self.bow_indices_queries=[]
        self.bow_frequencies_queries=[]
        
    
    def process_query_and_get_ID(self,query_external_ID,non_processed_query_text):
        """Function that adds the external query ID to the list and processes the query and adds it to the processed queries list"""
        #Adding the exteral query ID
        self.queries_IDs.append(str(query_external_ID))
        self.queries_internal_IDs[str(query_external_ID)]=len(self.queries_IDs)-1
        query=[]
        direct_query=[]
        #Processing the query
        tmp_counter=Counter()
        for elem in non_processed_query_text.split(" "):
            word=elem.lower()
            token=self.stemmer.stem(word)
            if token in self.vocabulary:
                query.append(token)
                #Adding the internal token id of the vocabulary to build the direct structure of the queries
                direct_query.append(self.vocabulary[token][1])
                #Computing frequency of each token in the query
                tmp_counter[token]+=1
        self.queries.append(query)
        self.direct_queries.append(direct_query)
        #building bag of words structure
        indices_bow=[]
        frequencies_bow=[]
        for token in tmp_counter:
            indices_bow.append(self.vocabulary[token][1])
            frequencies_bow.append(tmp_counter[token])
        self.bow_indices_queries.append(indices_bow)
        self.bow_frequencies_queries.append(frequencies_bow)
        
    def get_number_of_queries(self):
        return len(self.queries_IDs)
    def get_external_ID_of_query(self,internal_query_ID):
        """Function that gets external query ID from the internal query ID"""
        try:
            external_query_ID=self.queries_IDs[internal_query_ID]
        except ValueError:
            print( "token not present in the list")
        except:
            print("Unknown Error")
        return external_query_ID
    def get_internal_ID_of_query(self,external_query_ID):
        """Function that gets the internal_query_ID from the external_query_ID"""
        internal_query_ID=None
        for i in range(len(self.queries_IDs)):
             if external_query_ID==self.queries_IDs[i]:
                internal_query_ID=i
        if internal_query_ID==None:
            raise ValueError
        else:
            return internal_query_ID
    def query(self):
        """Access all queries as a generator """
        for query in self.queries:
            yield query
    def save(self,file_path,name_of_file):
        """Saving External queries ID and processed queries"""
        #File path is the path to where to save the objects .name of files contains the name of the file for example (training, validation ...)
        #Saving external queries ID
        with open(file_path+'/'+name_of_file+'_queries_IDs', 'wb') as f:
            pickle.dump(self.queries_IDs, f, protocol=pickle.HIGHEST_PROTOCOL)
        #Saving queries 
        with open(file_path+'/'+name_of_file+'_queries', 'wb') as f:
            pickle.dump(self.queries, f, protocol=pickle.HIGHEST_PROTOCOL)
        #Saving direct queries
        with open(file_path+'/'+name_of_file+'_direct_queries', 'wb') as f:
            pickle.dump(self.direct_queries, f, protocol=pickle.HIGHEST_PROTOCOL)
    def load(self,file_path,name_of_file):
        """ Loading queries IDs and queries and direct queries"""
        self.queries_IDs=[]
        self.queries=[]
        with open(file_path+'/'+name_of_file+'_queries_IDs', 'rb') as f:
            self.queries_IDs=pickle.load(f)
        with open(file_path+'/'+name_of_file+'_queries', 'rb') as f:
            self.queries=pickle.load(f)
        with open(file_path+'/'+name_of_file+'_direct_queries', 'rb') as f:
            self.direct_queries=pickle.load(f)