
import HHO as hho

import numpy

import Fitness_Function
import RWN_ok

from sklearn.metrics import precision_score, recall_score#, accuracy_score
import numpy as np
from sklearn.metrics import f1_score


def selector(popSize,Iter, trainInput, testInput, trainOutput, testOutput):
  

    numFeaturesData=numpy.shape(trainInput)[1] #number of features in the  dataset

    dimF=numFeaturesData
    hidden_node = 10    
    dim_all = numFeaturesData + hidden_node

    x=hho.HHO(getattr(Fitness_Function, "Fitness_fun"),-1,1,dim_all,dimF,popSize,Iter,trainInput,trainOutput)
    
    features_after_reduced=[]
    reducedhidden = 0
    for index in range(0,dimF):
        if (x.bestIndividual[index]==1):
            features_after_reduced.append(index)
    reduced_data_train_global=trainInput[:,features_after_reduced]
    reduced_data_test_global=testInput[:,features_after_reduced]

    best_neuron_indiv = x.bestIndividual[dimF:dim_all]
    
    while np.sum(best_neuron_indiv)==0:   
         best_neuron_indiv =np.random.randint(2, size=(10,))
         
    index_pow = 9
    for index in range (0,10):

        if(best_neuron_indiv[index]==1):
            reducedhidden += 1*pow(2,index_pow)
        index_pow -= 1 
    
   
    reducedhidden = int(reducedhidden)
    model = RWN_ok.RWN(reduced_data_train_global,trainOutput, reducedhidden)
    beta, train_score, x.weights, x.biases = model.fit()
    
    target_pred_train = model.predict(reduced_data_train_global)
    
    correct = 0
    total = trainOutput.shape[0]
    for i in range(total):
        if trainOutput[i] == target_pred_train[i]:
            correct += 1
    acc_train = correct/total

    x.trainAcc=acc_train
    
    target_pred_test = model.predict(reduced_data_test_global)

    correct = 0
    total = testOutput.shape[0]

    for i in range(total):
        if testOutput[i] == target_pred_test[i]:
            correct += 1
    acc_test = correct/total
    
    y_test_ = testOutput
    y_pred  = target_pred_test

    rec4 = recall_score(y_test_,y_pred, average=None,labels=np.unique(y_pred))
    pre4 = precision_score(y_test_,y_pred, average=None,labels=np.unique(y_pred))

    f_measure = f1_score(y_test_, y_pred, average='micro')

    Rec_product = np.prod(rec4)
    Prec_product = np.prod(pre4)

    x.testAcc=acc_test
    x.beta = beta
    x.F_measure = f_measure
    x.recall = Rec_product
    x.precision = Prec_product
    x.hiddenNode = reducedhidden

    return x
    
#####################################################################    
