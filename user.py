# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:16:09 2022

@author: Neveen.Hijazi
"""
import warnings
import numpy as np
import csv
import selector 
import utils


warnings.filterwarnings("ignore", category=UserWarning)
   
Flag=False

def reduction_hidden( dimF, dim_all):
    reducedhidden = 0
    best_neuron_indiv = x.bestIndividual[dimF:dim_all]
    index_pow = 9
    for index in range (0,10):
          if(best_neuron_indiv[index]==1):
              reducedhidden += 1*pow(2,index_pow)
          index_pow -= 1    

    reducedhidden = int(reducedhidden)   
    return reducedhidden

Export=True
PopulationSize = 200
Iterations= 100
NumOfRuns = 30
NumOfUsers = 5
if __name__ == "__main__":
 

    CnvgHeader1=[]
    CnvgHeader2=[]
    CnvgHeader3=[]

    
    
    numFeaturesData, X_train, y_train , X_test , y_test = utils.load_data()
    partition_id = np.random.choice(NumOfUsers)

    (X_train_, y_train_) = utils.partition(X_train, y_train, NumOfUsers)[partition_id]

    dimF = numFeaturesData    
    hidden_node = 10
    dim_all =dimF + hidden_node
    
    for l in range(0,Iterations):
    	CnvgHeader1.append("Iter_best_Fitness"+str(l+1))

    for l in range(0,Iterations):
    	CnvgHeader2.append("Iter_No_of_Feature"+str(l+1))
        
    for l in range(0,Iterations):
    	CnvgHeader3.append("Iter_No_of_Hidden_Node"+str(l+1))

    ExportToFile="Output_experiment_user.csv" 
    for k in range (0,NumOfRuns):
              
        x=selector.selector( PopulationSize, Iterations, X_train, X_test, y_train, y_test )
        
        reducedhidden = reduction_hidden( dimF, dim_all)
        
        if(Export==True):
            with open(ExportToFile, 'a',newline='\n') as out:
                writer = csv.writer(out,delimiter=',')
                if (Flag==False):
                    header= np.concatenate([["Experiment","trainAcc","testAcc" , "Recall "  , "F-score" , "Precision" ],CnvgHeader1,CnvgHeader2, CnvgHeader3])
                    writer.writerow(header)
                a=np.concatenate([[k+1, x.trainAcc,x.testAcc, x.recall, x.F_measure, x.precision],x.convergence1,x.convergence2, x.convergence3])
                writer.writerow(a)
            out.close()
        Flag=True 
