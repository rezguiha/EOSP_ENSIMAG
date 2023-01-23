# =============================================================================
# Created By  : Jibril FREJJ
# Created Date: February 16 2020
# Modified By  : Hamdi REZGUI
# Modified Date: March 21 2021
# E-mail: hamdi.rezgui@grenoble-inp.org
# Description: Definition of the classes of differentiable IR models with their
# network architecture
# =============================================================================
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers

####Differentiable models"
"""This file contains the definitions of differantiable models introduced in chapter 7 for tf,tf-idf,jm,dir,bm25 that can be trained on a particular
collection. It contains the architecture of the rededifined differentiable model with the one linear layer for the prediction of TDV""" #HR
class diff_simple_TF(Model):
    def __init__(self, embedding_matrix, dropout_rate=0.1):
        super(diff_simple_TF, self).__init__()
        self.vocab_size, self.embedding_dim = embedding_matrix.shape
        #The embedding matrix has an entry for the padding value 0. It has a shape of [vocab_size+1,embedding_dim]
        self.vocab_size=self.vocab_size-1
        #The input dimention of the layer is vocab_size+1 because we have a padding value 0
        self.embedding=tf.keras.layers.Embedding(input_dim=self.vocab_size+1,output_dim=self.embedding_dim,
                                       embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                       trainable=False,mask_zero=True)
        self.linear = tf.keras.layers.Dense(1,
                                            input_shape=(self.embedding_dim,),
                                            activation='relu',
                                            bias_initializer=tf.ones_initializer())
        self.dropout_rate = dropout_rate

    def make_dense_tensor(self,indices_sparse_tensor, values_sparse_tensor,batch_size):
        """Making a dense tensor of documents batch or queries batch passing by a sparse tensor out of indices in the form of (tokenID,document or query ID in the batch) and corresponding values"""
        sparse_tensor=tf.SparseTensor(indices=indices_sparse_tensor,
                                      values=values_sparse_tensor,dense_shape=[self.vocab_size+1,batch_size])
        sparse_tensor=tf.sparse.reorder(sparse_tensor)
        dense_tensor=tf.sparse.to_dense(sparse_tensor) 
        return dense_tensor

    def call(self, q_indices_sparse_tensor_batch,q_frequencies_bow_batch,d_indices_sparse_tensor_batch,
             d_indices_bow_batch,d_frequencies_bow_batch,batch_size):
        """
        This calculates the relevance between queries and documents of a batch and returns the dense tensor of the documents #HR
        q_indices_sparse_tensor_batch: list of indices for the sparse tensor. The index is (token_ID,query_ID in the batch)
        The token_ID comes from the bag of words representation of the document
        q_frequencies_bow_batch: a vector containing the frequencies corresponding to each token in the sequence of bog of 
        words of queries
        d_indices_sparse_tensor_batch: list of indices for the sparse tensor. The index is (token_ID,document_ID in the batch).
        The token_ID comes from the bag of words representation of the document
        d_indices_bow_batch: it is a vector containing all the bag of words representation indices of the batch of documents
        combined
        d_frequencies_bow_batch: The corresponding frequencies to the d_indices_bow_batch
        Warning : seems to overload call operator #JPC
        """
        #Creating a dense tensor for the batch of the queries
        q = self.make_dense_tensor(q_indices_sparse_tensor_batch, q_frequencies_bow_batch,batch_size)
        q=tf.cast(q,dtype=tf.float32)
        #Computing the TDV weights for each token in the bag of words representation of the batch
        freq_tdv = tf.nn.dropout(self.embedding(d_indices_bow_batch), rate=self.dropout_rate)
        freq_tdv = tf.nn.dropout(self.linear(freq_tdv), rate=self.dropout_rate)
        #Multiplying by the frequency by the TDV weights. This is the signal we are looking for
        freq_tdv=tf.math.multiply(tf.squeeze(freq_tdv),d_frequencies_bow_batch)
        #Creating a dense tensor for the document batch 
        d = self.make_dense_tensor(d_indices_sparse_tensor_batch,freq_tdv,batch_size)
        #Computing relavance by doing the scalar product of the dense tensors of the document batch and the query batch
        rel = tf.math.reduce_sum(tf.math.multiply(q, d), axis=0)

        return rel, d

    def compute_index(self):
        """This computes the TDV weights for the terms in the vocabulary""" #HR
        index = [_ for _ in range(1,self.vocab_size+1)]

        all_embeddings = self.embedding(np.asarray(index))

        return np.reshape(self.linear(all_embeddings).numpy(), (self.vocab_size,))


class diff_TF_IDF(Model):
    def __init__(self, embedding_matrix, dropout_rate=0.1):
        super(diff_TF_IDF, self).__init__()
        self.vocab_size, self.embedding_dim = embedding_matrix.shape
        #The embedding matrix has an entry for the padding value 0. It has a shape of [vocab_size+1,embedding_dim]
        self.vocab_size=self.vocab_size-1
        #The input dimention of the layer is vocab_size+1 because we have a padding value 0
        self.embedding=tf.keras.layers.Embedding(input_dim=self.vocab_size+1,output_dim=self.embedding_dim,
                                       embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                       trainable=False,mask_zero=True)
        self.linear = tf.keras.layers.Dense(1,
                                            input_shape=(self.embedding_dim,),
                                            activation='relu',
                                            bias_initializer=tf.ones_initializer())
        self.dropout_rate = dropout_rate

    def make_dense_tensor(self,indices_sparse_tensor, values_sparse_tensor,batch_size):
        sparse_tensor=tf.SparseTensor(indices=indices_sparse_tensor,
                                      values=values_sparse_tensor,dense_shape=[self.vocab_size+1,batch_size])
        sparse_tensor=tf.sparse.reorder(sparse_tensor)
        dense_tensor=tf.sparse.to_dense(sparse_tensor) 
        return dense_tensor

    def call(self, q_indices_sparse_tensor_batch,q_frequencies_bow_batch,d_indices_sparse_tensor_batch,
             d_indices_bow_batch,d_frequencies_bow_batch,batch_size):

        q = self.make_dense_tensor(q_indices_sparse_tensor_batch, q_frequencies_bow_batch,batch_size)
        q=tf.cast(q,dtype=tf.float32)
        
        freq_tdv = tf.nn.dropout(self.embedding(d_indices_bow_batch), rate=self.dropout_rate)
        freq_tdv = tf.nn.dropout(self.linear(freq_tdv), rate=self.dropout_rate)
        freq_tdv=tf.math.multiply(tf.squeeze(freq_tdv),d_frequencies_bow_batch)

        d = self.make_dense_tensor(d_indices_sparse_tensor_batch,freq_tdv,batch_size)

        maxdf = tf.keras.backend.max(tf.math.reduce_sum(d, axis=1))

        idf = tf.math.log((maxdf + 1) / (1 + tf.math.reduce_sum(d, axis=1)))

        idf_d = tf.multiply(d, tf.reshape(idf, (-1, 1)))

        rel = tf.math.reduce_sum(tf.math.multiply(q, idf_d), axis=0)

        return rel, d

    def compute_index(self):
        index = [_ for _ in range(1,self.vocab_size+1)]

        all_embeddings = self.embedding(np.asarray(index))

        return np.reshape(self.linear(all_embeddings).numpy(), (self.vocab_size,))


class diff_DIR(Model):
    def __init__(self, embedding_matrix, mu=2500.0, dropout_rate=0.1):
        super(diff_DIR, self).__init__()
        self.vocab_size, self.embedding_dim = embedding_matrix.shape
        #The embedding matrix has an entry for the padding value 0. It has a shape of [vocab_size+1,embedding_dim]
        self.vocab_size=self.vocab_size-1
        #The input dimention of the layer is vocab_size+1 because we have a padding value 0
        self.embedding=tf.keras.layers.Embedding(input_dim=self.vocab_size+1,output_dim=self.embedding_dim,
                                       embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                       trainable=False,mask_zero=True)
        initializer=tf.keras.initializers.he_normal()
        self.linear = tf.keras.layers.Dense(100,
                                            input_shape=(self.embedding_dim,),
                                            activation='relu',kernel_initializer=initializer)
        
        self.linear2 = tf.keras.layers.Dense(1,
                                            input_shape=(100,),
                                            activation='relu',
                                            bias_initializer=tf.ones_initializer(),
                                            kernel_initializer=initializer)
        self.mu = tf.Variable(mu)
        self.dropout_rate = dropout_rate

    def make_dense_tensor(self,indices_sparse_tensor, values_sparse_tensor,batch_size):
        sparse_tensor=tf.SparseTensor(indices=indices_sparse_tensor,
                                      values=values_sparse_tensor,dense_shape=[self.vocab_size+1,batch_size])
        sparse_tensor=tf.sparse.reorder(sparse_tensor)
        dense_tensor=tf.sparse.to_dense(sparse_tensor) 
        return dense_tensor

    def call(self, q_indices_sparse_tensor_batch,q_frequencies_bow_batch,d_indices_sparse_tensor_batch,
             d_indices_bow_batch,d_frequencies_bow_batch,batch_size):

        q = self.make_dense_tensor(q_indices_sparse_tensor_batch, q_frequencies_bow_batch,batch_size)
        q=tf.cast(q,dtype=tf.float32)
        
        freq_tdv = tf.nn.dropout(self.embedding(d_indices_bow_batch), rate=self.dropout_rate)
        freq_tdv = tf.nn.dropout(self.linear(freq_tdv), rate=self.dropout_rate)
        freq_tdv = tf.nn.dropout(self.linear2(freq_tdv), rate=self.dropout_rate)
        freq_tdv=tf.math.multiply(tf.squeeze(freq_tdv),d_frequencies_bow_batch)

        d = self.make_dense_tensor(d_indices_sparse_tensor_batch,freq_tdv,batch_size)

        cfreq = tf.math.reduce_sum(d, axis=1) / tf.math.reduce_sum(d)

        smoothing = tf.math.log(self.mu / (tf.math.reduce_sum(d, axis=0) + self.mu))

        dir_d = tf.math.log(1 + d / (1 + self.mu * tf.reshape(cfreq, (-1, 1)))) + smoothing

        rel = tf.math.reduce_sum(tf.math.multiply(q, dir_d), axis=0)

        return rel, d

    def compute_index(self):
        index = [_ for _ in range(1,self.vocab_size+1)]

        all_embeddings = self.embedding(np.asarray(index))

        return np.reshape(self.linear2(self.linear(all_embeddings)).numpy(), (self.vocab_size,))


class diff_BM25(Model):
    def __init__(self, embedding_matrix, k1=1.2, b=0.75, dropout_rate=0.1):
        super(diff_BM25, self).__init__()
        self.vocab_size, self.embedding_dim = embedding_matrix.shape
        #The embedding matrix has an entry for the padding value 0. It has a shape of [vocab_size+1,embedding_dim]
        self.vocab_size=self.vocab_size-1
        #The input dimention of the layer is vocab_size+1 because we have a padding value 0
        self.embedding=tf.keras.layers.Embedding(input_dim=self.vocab_size+1,output_dim=self.embedding_dim,
                                       embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                       trainable=False,mask_zero=True)
        
        initializer=tf.keras.initializers.he_normal()
        self.linear = tf.keras.layers.Dense(100,
                                            input_shape=(self.embedding_dim,),
                                            activation='relu',
                                           kernel_initializer=initializer)
        
        self.linear2 = tf.keras.layers.Dense(1,
                                            input_shape=(100,),
                                            activation='relu',
                                            bias_initializer=tf.ones_initializer(),
                                            kernel_initializer=initializer)
        self.k1 = tf.Variable(k1)
        self.b = tf.Variable(b)
        self.dropout_rate = dropout_rate

    def make_dense_tensor(self,indices_sparse_tensor, values_sparse_tensor,batch_size):
        sparse_tensor=tf.SparseTensor(indices=indices_sparse_tensor,
                                      values=values_sparse_tensor,dense_shape=[self.vocab_size+1,batch_size])
        sparse_tensor=tf.sparse.reorder(sparse_tensor)
        dense_tensor=tf.sparse.to_dense(sparse_tensor) 
        return dense_tensor

    def call(self, q_indices_sparse_tensor_batch,q_frequencies_bow_batch,d_indices_sparse_tensor_batch,
             d_indices_bow_batch,d_frequencies_bow_batch,batch_size):

        q = self.make_dense_tensor(q_indices_sparse_tensor_batch, q_frequencies_bow_batch,batch_size)
        q=tf.cast(q,dtype=tf.float32)
        
        freq_tdv = tf.nn.dropout(self.embedding(d_indices_bow_batch), rate=self.dropout_rate)
        freq_tdv = tf.nn.dropout(self.linear(freq_tdv), rate=self.dropout_rate)
        freq_tdv = tf.nn.dropout(self.linear2(freq_tdv), rate=self.dropout_rate)
        freq_tdv=tf.math.multiply(tf.squeeze(freq_tdv),d_frequencies_bow_batch)

        d = self.make_dense_tensor(d_indices_sparse_tensor_batch,freq_tdv,batch_size)
        
        maxdf = tf.keras.backend.max(tf.math.reduce_sum(d, axis=1))

        idf = tf.math.log((maxdf + 1) / (1 + tf.math.reduce_sum(d, axis=1)))

        d_length = tf.math.reduce_sum(d, axis=0)

        avg_d_length = tf.reduce_mean(d_length)

        bm25_d = tf.reshape(idf, (-1, 1)) * ((self.k1 + 1) * d) / (
                    d + self.k1 * ((1 - self.b) + self.b * d_length / avg_d_length))

        rel = tf.math.reduce_sum(tf.math.multiply(q, bm25_d), axis=0)

        return rel, d

    def compute_index(self):
        index = [_ for _ in range(1,self.vocab_size+1)]

        all_embeddings = self.embedding(np.asarray(index))

        return np.reshape(self.linear2(self.linear(all_embeddings)).numpy(), (self.vocab_size,))

# The class for Jelnik-Mercer model was absent in the original file. HR created it
class diff_JM(Model):
    def __init__(self, embedding_matrix, lamb=0.15, dropout_rate=0.1):
        super(diff_JM, self).__init__()
        
        self.vocab_size, self.embedding_dim = embedding_matrix.shape
        #The embedding matrix has an entry for the padding value 0. It has a shape of [vocab_size+1,embedding_dim]
        self.vocab_size=self.vocab_size-1
        #The input dimention of the layer is vocab_size+1 because we have a padding value 0
        self.embedding=tf.keras.layers.Embedding(input_dim=self.vocab_size+1,output_dim=self.embedding_dim,
                                       embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                       trainable=False,mask_zero=True)
        self.linear = tf.keras.layers.Dense(1,
                                            input_shape=(self.embedding_dim,),
                                            activation='relu',
                                            bias_initializer=tf.ones_initializer())
        self.lamb= tf.Variable(lamb)
        self.dropout_rate = dropout_rate

    def make_dense_tensor(self,indices_sparse_tensor, values_sparse_tensor,batch_size):
        sparse_tensor=tf.SparseTensor(indices=indices_sparse_tensor,
                                      values=values_sparse_tensor,dense_shape=[self.vocab_size+1,batch_size])
        sparse_tensor=tf.sparse.reorder(sparse_tensor)
        dense_tensor=tf.sparse.to_dense(sparse_tensor) 
        return dense_tensor

    def call(self, q_indices_sparse_tensor_batch,q_frequencies_bow_batch,d_indices_sparse_tensor_batch,
             d_indices_bow_batch,d_frequencies_bow_batch,batch_size):

        q = self.make_dense_tensor(q_indices_sparse_tensor_batch, q_frequencies_bow_batch,batch_size)
        q=tf.cast(q,dtype=tf.float32)
        
        freq_tdv = tf.nn.dropout(self.embedding(d_indices_bow_batch), rate=self.dropout_rate)
        freq_tdv = tf.nn.dropout(self.linear(freq_tdv), rate=self.dropout_rate)
        freq_tdv=tf.math.multiply(tf.squeeze(freq_tdv),d_frequencies_bow_batch)

        d = self.make_dense_tensor(d_indices_sparse_tensor_batch,freq_tdv,batch_size)

        cfreq = tf.math.reduce_sum(d, axis=1) / tf.math.reduce_sum(d)
        d_length = tf.math.reduce_sum(d, axis=0)

        jm_d = tf.math.log(1+((self.lamb/(1-self.lamb))*(d/d_length)/(1+tf.reshape(cfreq,(-1,1)))))

        rel = tf.math.reduce_sum(tf.math.multiply(q, jm_d), axis=0)

        return rel, d

    def compute_index(self):
        index = [_ for _ in range(1,self.vocab_size+1)]

        all_embeddings = self.embedding(np.asarray(index))

        return np.reshape(self.linear(all_embeddings).numpy(), (self.vocab_size,))
