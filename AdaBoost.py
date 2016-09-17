# -*- coding: utf-8 -*-
"""
Created on Fri May 13 09:47:52 2016

@author: rahma.chaabouni
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 09 10:52:07 2016

@author: rahma.chaabouni
"""
#from __future__ import division

import time
import numpy as np
import theano
import theano.tensor as T
from SimpleNeuralNet import MLP

def AdaBoost(train_set_x, train_set_y, T_max, reg_L1 = 0.01, reg_L2 = 0.01, learning_rate=0.01,
             n_epochs=1000, batch_size=20, n_in = 28*28, 
             n_hiddens = [500, 500], n_out = 2, activations = [T.tanh, T.tanh],
             x = T.matrix('x'), y= T.ivector('y'), w = T.dvector('w'),
             valid_set_x = None, valid_set_y = None):
    # Initialisation
    m = train_set_x.get_value(borrow= True).shape[0] # nombre de données d'apprentissage #tout get value prend beaucoup de temps
    weights = theano.shared(np.ones(m)/float(batch_size)) #poids des données d'apprentissage lors de la descente de gradient
    weights_all = theano.shared(np.ones(m)/float(m)) # poids des données d'apprentissage pour le calcule des erreurs
    n = m/batch_size
    weaks = []
    alphas = []
        
    weights_part = w/w.sum() 
    normalize_function = theano.function(
            inputs=[w],
            outputs=weights_part)
        
    # Build the T weak classifiers
    for t in range(T_max):
        print('... building the model')

        # Construct the weak learner
        rng = np.random.RandomState(1234)
        h = MLP(
        rng=rng,
        input=x,
        n_in=n_in,
        n_hiddens=n_hiddens,
        n_out=n_out,
        activations = activations
    )
        #train the weak learn
        h.train(weights, train_set_x, train_set_y, x,y,w, reg_L1, reg_L2, 
                learning_rate=learning_rate, n_epochs=n_epochs, 
                batch_size=batch_size, valid_set_x = valid_set_x, valid_set_y = valid_set_y)
        
        # Un vecteur de booleens ou chaque element est egal à 1 si la prediction de cet element par h est incorrect et 0 sinon
        incorrect = h.incorrects(y)

        # Fraction d'erreur
        epsilon = h.errors(y,weights_all)
        # Poids du weak classifieur
        alpha = 0.5* T.log((1- epsilon)/(0.000000001+ epsilon))

        # mis à jour des poids avec la formule du AdaBoost
        updates = [(weights_all, weights_all *T.exp((incorrect-(-incorrect+1))* alpha))]
        
        #calculer l'erreur à chaque itération pour vérifier que nous avons bien
        #un classifieur faible plus performant qu'un choix aléatoire
        er = theano.function(
            inputs=[],
            outputs=epsilon,
            givens={y: train_set_y,
                    x: train_set_x}
        )
        print(("l'erreur en apprentissage est " + str(er()*100) + "%"))
        compute_alpha = theano.function(
            inputs=[],
            outputs=alpha,
            updates=updates,
            givens={y: train_set_y,
                    x: train_set_x}
        )
        # calculer alpha et mettre à jour les poids à chaque calcule de alpha
        alph = compute_alpha()
        # Normaliser weights_all    
        new_val = normalize_function(weights_all.get_value())
        weights_all.set_value(new_val)
        
        # Reinitialiser les weights for the minibatch gradient descent
        new_w_batch = []
        for indice in range(n):
            new_tmp = normalize_function(weights_all.get_value()[indice*batch_size: (indice+1)*batch_size])
            new_w_batch.append(new_tmp)            

        new_w_batch = [item for sublist in new_w_batch for item in sublist]
        weights.set_value(new_w_batch)
        # sauvegarder le reseau entrainer et son poids
        weaks.append(h)
        alphas.append(alph)

    return alphas, weaks

def transform0_2(vect): 
    X = theano.shared(vect)
    index = T.vector('index')
    index = theano.tensor.eq(X, 0).nonzero()[0]
    X_update = (X, T.set_subtensor(X[index], 1))
    f = theano.function([], updates=[X_update])
    return f
        
def transform_1_2(vect): 
    X = theano.shared(vect)
    index = T.vector('index')
    index = theano.tensor.eq(X, -1).nonzero()[0]
    X_update = (X, T.set_subtensor(X[index], 0))
    f = theano.function([], updates=[X_update])
    return f

def transform_1(vect): 
    vect[vect==-1] = 0
    return vect

def transform0(vect): 
    vect[vect==0] = -1
    return vect
        
# Construire stong classifier
def Error(alphas, learners, X, Y):
    # Warning: We need 1-/1 prediction for the boosting where the neural network
    # predicts 0/1 
    predictions = [h.predict(X) for h in learners]

 #   Strong_pred = theano.function([])    
    H_x = [alpha * transform0(h.predict(X)) for alpha, h in zip(alphas, learners)]
    H_x = np.sign(sum(H_x))
    # retransform to compare it to Y
    H_x = transform_1(H_x)
    error = sum([y1 != y2 for y1, y2 in zip(Y, H_x)])/float(len(Y))
    return error, predictions, H_x

def marge(alphas,learners,X,Y):
    F_x = [alpha * transform0(h.predict(X)) for alpha, h in zip(alphas, learners)]
    F_x = np.asmatrix(F_x) 
    new_Y = transform0(Y)
    l = np.sum(F_x,axis = 0)
    toto = [item for sublist in l for item in sublist]
    return (new_Y * toto)/sum(alphas)
    
    
def strong_score(alphas,learners,X):
    pr = [alpha * h.score(X) for alpha, h in zip(alphas, learners)]
    return pr/sum(alphas)