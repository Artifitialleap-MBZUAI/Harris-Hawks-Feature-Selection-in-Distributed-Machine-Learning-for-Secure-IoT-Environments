from sklearn.model_selection import train_test_split
import RWN_ok
from sklearn.metrics import f1_score
import numpy as np
def Fitness_fun(I,trainInput,trainOutput,dim, dimF):            
         data_train_internal, data_test_internal, target_train_internal, target_test_internal = train_test_split(trainInput, trainOutput, test_size=0.25, random_state=1)

         features_after_reduced=[]
         countF=0
         for index in range(0,dimF):
           if (I[index]==1):
               features_after_reduced.append(index)
               countF +=1

         reduced_data_train_internal=data_train_internal[:,features_after_reduced]
         reduced_data_test_internal=data_test_internal[:,features_after_reduced]
   
         reducedhidden = 0
       
         best_neuron_indiv = I[dimF:dim]

         while np.sum(best_neuron_indiv)==0:   
              best_neuron_indiv =np.random.randint(2, size=(10,))
    
         index_pow = 9
        
         for index in range (0,10):
             if(best_neuron_indiv[index]==1):
                 reducedhidden += 1*pow(2,index_pow)
             index_pow -= 1    
                 
         reducedhidden = int(reducedhidden)   

         clf = RWN_ok.RWN( reduced_data_train_internal, target_train_internal, reducedhidden)

         beta, train_score, weights, baises = clf.fit()
         
         target_pred_internal_RWN = clf.predict(reduced_data_test_internal)
                 
         F1score = f1_score(target_test_internal, target_pred_internal_RWN, average='micro')

         fitness=0.99*(1-F1score)+0.01*countF/(dimF) + 0.01*(reducedhidden)/1024
         
         return fitness



