# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 13:29:03 2016

@author: rahma.chaabouni
"""
import time
import numpy as np
import theano
import theano.tensor as T
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from AdaBoost import AdaBoost, strong_score, marge
from logistic_sgd import display_marge

def load_data(dataset):
    import cPickle, gzip

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    print('... loading data')
    return [train_set, valid_set, test_set]
    
def take_binardata(data_xy, label):
    
    x, y = data_xy
    data_x = x.copy()
    data_y = y.copy()
    cond_0 = (data_y == label)
    cond_1 = (data_y != label)

    data_y[cond_0] = 0    
    data_y[cond_1] = 1
    
    cond = [y0 or y1 for y0, y1 in zip(cond_0,cond_1)]
    data_x = data_x[np.ix_(cond)]
    data_y = np.extract(cond, data_y)
    return (data_x, data_y)
    

def shared_dataset(data_x, data_y, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        shared_x = theano.shared(value = np.asarray(data_x, dtype = theano.config.floatX),
                                borrow = borrow)
        shared_y = theano.shared(value = np.asarray(data_y, dtype = theano.config.floatX),
                                borrow = borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

def multiclass_train(data_xy, T_max = 1, reg_L1 = 0.001, reg_L2 = 0.0, 
                     learning_rate=0.01, n_epochs=10, batch_size=20, 
                     n_in = 28*28, n_hiddens = [50, 50], n_out = 2, 
                     activations = [T.tanh, T.tanh], validation_xy = None): 

    num_labels = 10
    learners_list = []
    alphas_list = []
    
    for label in range(num_labels):
        t0 = time.time()
        print(label)  
        data_x_val, data_y_val = take_binardata(data_xy, label)
        data_x, data_y = shared_dataset(data_x_val, data_y_val)
        if validation_xy==None:
            valid_x = None
            valid_y = None
        else:
            valid_x, valid_y = take_binardata(validation_xy, label)
            valid_x, valid_y = shared_dataset(valid_x, valid_y)
                

        H = AdaBoost(data_x, data_y, T_max, 
                     reg_L1=reg_L1, reg_L2=reg_L2, 
                     learning_rate=learning_rate, n_epochs=n_epochs, 
                     batch_size=batch_size, n_in=n_in, n_hiddens=n_hiddens, 
                     n_out = n_out, activations = activations, 
                     valid_set_x = valid_x , valid_set_y = valid_y)
                     
        alphas_list.append(H[0])
        learners_list.append(H[1])
        print('temps pour un entraienement one_vs_all est', time.time()-t0)
    return alphas_list, learners_list


def multiclass_predict(alphas_list, learners_list, data_x, type_donnees = 'numpy'):
    t0 = time.time()
    if(type_donnees == 'tensor'):
        data_x = data_x.get_value()
    
    
    scores = np.zeros((len(data_x), 10))
    # Compute the score of each classifier
    scores_list = [strong_score(alphas, learners,data_x) 
                for alphas, learners in zip(alphas_list,learners_list)]
        
    for i, s in enumerate(scores_list):
        scores[:,i] = [examples[0] for examples in s[0]]    
        
     
    # predictions en maximisant le score 
    predictions = np.argmax(scores, axis= 1)
    print('temps pour prediciton one_vs_all est', time.time()-t0)
    return predictions
    
   
def Evaluation_marge(alphas_list, learners_list, data_x, data_y):
    marges_list = [marge(alphas, learners,data_x, data_y) 
                for alphas, learners in zip(alphas_list,learners_list)]
    return marges_list
                    
