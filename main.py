# -*- coding: utf-8 -*-
"""
Created on Wed May 11 15:42:35 2016

@author: rahma.chaabouni
"""
import timeit
import os
from One_Vs_All import load_data, multiclass_train, multiclass_predict, take_binardata
from logistic_sgd import compute_error_confMatrix, display_marge
import theano.tensor as T
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from AdaBoost import marge
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd


os.chdir('C:\\Users\\rahma.chaabouni\\Documents\\Recherches\\Theano\\Datasets')
dataset='mnist.pkl.gz'
datasets = load_data(dataset)

#
###############################################################################
####                    Construction et entrainement                       ####
###############################################################################
start_time = timeit.default_timer()

T_max = (2,3,5,10,20)
#T_max = [5]
erreur_train_list = []
erreur_test_list = []

for t in T_max:
    print t
    alphas, learners = multiclass_train(datasets[0], T_max = t, 
                                        reg_L1 = 0.0, reg_L2 = 0.0, 
                                        learning_rate=0.01, n_epochs=10, 
                                        batch_size=20, n_in = 28*28,  
                                        n_hiddens = [10], n_out = 2, 
                                        activations = [T.tanh])
    
    ##############################################################################
    ###                               Prédictions                             ####
    ##############################################################################
    prediction_train = multiclass_predict(alphas, learners, datasets[0][0])
    prediction_test = multiclass_predict(alphas, learners, datasets[2][0])

           

    ##############################################################################
    ###                             Evaluation                                 ###
    ##############################################################################
    #sns.set()
    # en apprentissage
    y_true_train = datasets[0][1]
    CM_train = confusion_matrix(y_true_train, prediction_train)
#    plot = sns.heatmap(pd.DataFrame(CM_train), annot=True, fmt="d", linewidths=.5)
#    fig1 = plot.get_figure()
    #fig1.savefig('heatmap_train_100boost_2couches_reg2')
    
    # en test 
    y_true_test = datasets[2][1]
    CM_test = confusion_matrix(y_true_test, prediction_test)
#    plot = sns.heatmap(pd.DataFrame(CM_test), annot=True, fmt="d", linewidths=.5)
#    fig2 = plot.get_figure()
    #fig2.savefig('heatmap_test_100boost_2couches_reg2')
    
    
#    print(CM_train)
    erreur_train_list.append(compute_error_confMatrix(CM_train, len(y_true_train)))
#    print(CM_test)
    erreur_test_list.append(compute_error_confMatrix(CM_test, len(y_true_test)))


end_time = timeit.default_timer()
print(("Temps d'excécution du code est %.2fm" % ((end_time - start_time)/60.)))

#############################################################################
##                        Explication avec la marge                      ###
#############################################################################
plt.close('all')

start_time = timeit.default_timer()

f, axarr = plt.subplots(2, 5)
iterations = [1,3, 10,100]
color = cm.rainbow(np.linspace(0, 1, len(iterations)))
i = 0
for T_max in iterations:
    print('le nombre ditération est '+ str(T_max))
    alphas, learners = multiclass_train(datasets[0], T_max = T_max, 
                                    reg_L1 = 0.001, reg_L2 = 0.00, 
                                    learning_rate=0.01, n_epochs=10, 
                                    batch_size=20, n_in = 28*28,  
                                    n_hiddens = [50, 50], n_out = 2, 
                                    activations = [T.tanh, T.tanh])
    
    for label in range(10):    
        data_x_val, data_y_val = take_binardata(datasets[0], label)
        marge_vector = marge(alphas[label], learners[label],data_x_val, data_y_val)        
        display_marge(marge_vector, color[i], eval('axarr['+ 
                                              str(0 if label<5 else 1) +','+ 
                                              str(label if label<5 else (label -5))+']'))
       
        eval('axarr['+ str(0 if label<5 else 1) +','+ 
             str(label if label<5 else (label -5))+']').set_title(label)

    i = i+1

end_time = timeit.default_timer()
print(("Temps d'excécution du code est %.2fm" % ((end_time - start_time)/60.)))
