# -*- coding: utf-8 -*-
"""
Created on Thirsday March 21  2019
The original code for HHO is in the Main paper mentioned below, and we made small updates to make it binary
@author: 
% _____________________________________________________
% Main paper:
% Harris hawks optimization: Algorithm and applications
% Ali Asghar Heidari, Seyedali Mirjalili, Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, Huiling Chen
% Future Generation Computer Systems, 
% DOI: https://doi.org/10.1016/j.future.2019.02.028
% _____________________________________________________

"""
import random
import numpy
import math
from solution import solution


def HHO(objf, lb, ub, dim,dimF, SearchAgents_no, Max_iter,trainInput,trainOutput):

    # initialize the location and Energy of the rabbit
    Rabbit_Location = numpy.zeros(dim)
    Rabbit_Energy = float("inf")  # change this to -inf for maximization problems

    X=numpy.random.randint(2, size=(SearchAgents_no,dim))#generating binary individuals
      
    for i in range(0, SearchAgents_no):
        for d in range (0, dim):
            if(random.random()>=0.5):
                X[i][d]=1
           
    # Initialize convergence
    convergence_curve_1 = numpy.zeros(Max_iter)
    convergence_curve_2 = numpy.zeros(Max_iter)
    convergence_curve_3 = numpy.zeros(Max_iter)

    s = solution()

    t = 0  # Loop counter

    # Main loop
    while t < Max_iter:
        for i in range(0, SearchAgents_no):
            # Check boundries
            X[i, :] = numpy.clip(X[i, :], lb, ub)
            while numpy.sum(X[i,:])==0:   
                 X[i,:]=numpy.random.randint(2, size=(1,dim))

            # fitness of locations
            fitness = objf(X[i, :],trainInput,trainOutput,dim, dimF)
            # Update the location of Rabbit
            if fitness < Rabbit_Energy:  # Change this to > for maximization problem
                Rabbit_Energy = fitness;
                Rabbit_Location = X[i, :].copy()   
                                                  
        E1 = 2 * (1 - (t / Max_iter));  # factor to show the decreaing energy of rabbit

        # Update the location of Harris' hawks
        for i in range(0, SearchAgents_no):

            E0 = 2 * random.random() - 1  # -1<E0<1
            Escaping_Energy = E1 * (E0)  # escaping energy of rabbit Eq. (3) in the paper

            # -------- Exploration phase Eq. (1) in paper -------------------

            if abs(Escaping_Energy) >= 1:
                # Harris' hawks perch randomly based on 2 strategy:
                q = random.random()
                rand_Hawk_index = math.floor(SearchAgents_no * random.random())
                X_rand = X[rand_Hawk_index, :]
                
                if q < 0.5:
                    # perch based on other family members
                        r1=random.random() # r1 is a random number in [0,1]
                        r2=random.random() # r2 is a random number in [0,1]  
                        # perch based on other family members
                        
                        X[i, :] = (X_rand - r1) * abs(X_rand - 2 * r2 * X[i, :]) 
                        
                        for d in range (0, dim):
                            S = 1 / (1 + numpy.exp(-2*X[i, d]))
                            if (random.random()<S): 
                              X[i,d]=1;
                            else:
                              X[i,d]=0;

                elif q >= 0.5:
                    r3=random.random() # r1 is a random number in [0,1]
                    r4=random.random() # r2 is a random number in [0,1] 
                    # perch on a random tall tree (random site inside group's home range)
                    X[i, :] = (Rabbit_Location - X.mean(0)) - r3 *  ((ub - lb) * r4 + lb )

                    for j in range (0,dim):
                        ss= 1 / (1 + numpy.exp(-2*X[i, j]))
                        if (random.random()<ss): 
                          X[i,j]=1;
                        else:
                          X[i,j]=0;
                          
            # -------- Exploitation phase -------------------
            elif abs(Escaping_Energy) < 1:
                # Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                # phase 1: ----- surprise pounce (seven kills) ----------
                # surprise pounce (seven kills): multiple, short rapid dives by different hawks

                r = random.random()  # probablity of each event

                if (r >= 0.5 and abs(Escaping_Energy) < 0.5):  # Hard besiege Eq. (6) in paper
                        X[i, :] = (Rabbit_Location) - Escaping_Energy * abs(Rabbit_Location - X[i, :])
                        for d in range (0,dim): 

                            ss =  1 / (1 + numpy.exp(-2*X[i, d]))   
                            if (random.random()<ss): 
                              X[i,d]=1;
                            else:
                              X[i,d]=0;
                        
                        
                if (r >= 0.5 and abs(Escaping_Energy) >= 0.5):  # Soft besiege Eq. (4) in paper
                        Jump_strength = 2 * (1 - random.random())  # random jump strength of the rabbit
                        X[i, :] = (Rabbit_Location - X[i, :]) - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X[i, :])
                        for d in range (0,dim):      
                            ss =  1 / (1 + numpy.exp(-2*X[i, d]))   
                            if (random.random()<ss): 
                              X[i,d]=1;
                            else:
                              X[i,d]=0;
                # phase 2: --------performing team rapid dives (leapfrog movements)----------


                if (r < 0.5 and abs(Escaping_Energy) >= 0.5):  # Soft besiege Eq. (10) in paper
                  X1  = numpy.zeros([1, dim], dtype='float')
                  X2  = numpy.zeros([1, dim], dtype='float')
                  
                
                    # rabbit try to escape by many zigzag deceptive motions
                  Jump_strength = 2 * (1 - random.random())

                  LF = Levy(dim)
                  for d in range (0,dim):
                      Xn = Rabbit_Location[d] - Escaping_Energy * abs(Jump_strength * Rabbit_Location[d] - X[i, d])
                      Xn = numpy.clip(Xn, lb, ub)
                      randnum = random.random()
                      ss =  1 / (1 + numpy.exp(-2*Xn)) 
                      if (randnum <ss): 

                        X1[0,d]=1;
                      else:
                        X1[0,d]=0;
                        
                      while numpy.sum(X1[0,d])==0:   
                          # print(numpy.sum(X1[0,d]), " $$$")
                           ss =  1 / (1 + numpy.exp(-2*Xn)) 
                           if (random.random() <ss): 
    
                             X1[0,d]=1;
                           else:
                             X1[0,d]=0;
                             

                      Yn = X1[0,d] + (random.random() * LF[d])
                      Yn = numpy.clip(Yn, lb, ub)
                      
                      ss =  1 / (1 + numpy.exp(-2*Yn)) 

                      if (random.random()<ss): 
                        X2[0,d]=1;
                      else:
                        X2[0,d]=0;
                        
                        
                      while numpy.sum(X2[0,d])==0:  
                          # print(numpy.sum(X2[0,d]), " ###")
                           ss =  1 / (1 + numpy.exp(-2*Yn))
                           if (random.random() <ss): 
                             X2[0,d]=1;
                           else:
                             X2[0,d]=0;                             
                  
                  A = numpy.array(X1)
                  B = numpy.array(X2)
                  
                  A_a = (A.ravel())
                  B_b = (B.ravel())
                  
                  fitX1 = objf(A_a, trainInput,trainOutput,dim, dimF) 
                  fitX2 = objf(B_b, trainInput,trainOutput,dim, dimF)
                  if fitX1 < fitness:
                    #  fitness = fitX1
                      X[i, :] = X1.copy()
                    
                  if fitX2 < fitness:
                       # fitness = fitX2
                      X[i, :] = X2.copy() 
                  

                X_mean = X.mean(0)
                if ( r < 0.5 and abs(Escaping_Energy) < 0.5):  # Hard besiege Eq. (11) in paper

                  X1  = numpy.zeros([1, dim], dtype='float')
                  X2  = numpy.zeros([1, dim], dtype='float')
                  LF = Levy(dim)
                  Jump_strength = 2 * (1 - random.random())
                  
                  for d in range (0, dim):
                      Xn = Rabbit_Location[d] - Escaping_Energy * abs(Jump_strength * Rabbit_Location[d] - X_mean[d])
                      Xn = numpy.clip(Xn, lb, ub)
                      ss =  1 / (1 + numpy.exp(-2*Xn)) 
                      
                      if (random.random()<ss): 
                        X1[0, d]=1;
                      else:
                        X1[0, d]=0;
                        
                        
                      while numpy.sum(X1[0,d])==0:   
                          # print(numpy.sum(X1[0,d]), " ***")
                           ss =  1 / (1 + numpy.exp(-2*Xn)) 
                           if (random.random() <ss): 
                             X1[0,d]=1;
                           else:
                             X1[0,d]=0;
                      
                      Yn = X1[0, d] + (random.random()* LF[d])
                      
                      Yn = numpy.clip(Yn, lb, ub)
                      ss =  1 / (1 + numpy.exp(-2*Yn)) 
                      
                      if (random.random()<ss): 
                        X2[0, d]=1;
                      else:
                        X2[0, d]=0;
                        
                    
                      while numpy.sum(X2[0,d])==0:   
                        #   print(numpy.sum(X2[0,d]), " %%%")
                           ss =  1 / (1 + numpy.exp(-2*Yn)) 
                           if (random.random() <ss): 
                             X2[0,d]=1;
                           else:
                             X2[0,d]=0;
                  
                  A = numpy.array(X1)
                  B = numpy.array(X2)
                  
                  A_a = (A.ravel())
                  B_b = (B.ravel())
                  
                  fitX1 = objf(A_a, trainInput,trainOutput,dim, dimF)  
                  fitX2 = objf(B_b, trainInput,trainOutput,dim, dimF)
                  if fitX1 < fitness:  # improved move?
                      X[i, :] = X1.copy()
                    
                  if fitX2 < fitness:
                      X[i, :] = X2.copy()
                       
        featurecount=0
        for f in range(0,dimF):
         if Rabbit_Location[f]==1:
           featurecount=featurecount+1
           
     
        reducedhidden =0        
        best_neuron_indiv = Rabbit_Location[dimF:dim]
        index_pow = 9
        for index in range (0,10):

            if(best_neuron_indiv[index]==1):
                reducedhidden += 1*pow(2,index_pow)
            index_pow -= 1
       
       
        convergence_curve_1[t]=Rabbit_Energy
        convergence_curve_2[t]=featurecount
        convergence_curve_3[t]=reducedhidden
          
        t = t + 1                    

    s.convergence1=convergence_curve_1
    s.convergence2=convergence_curve_2
    s.convergence3=convergence_curve_3

    s.optimizer = "HHO"
    s.objfname = objf.__name__
    s.best = Rabbit_Energy
    s.bestIndividual = Rabbit_Location
    s.hiddenNode = reducedhidden

    return s


def Levy(dim):
    beta = 1.5
    sigma = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = 0.01 * numpy.random.randn(dim) * sigma
    v = numpy.random.randn(dim)
    zz = numpy.power(numpy.absolute(v), (1 / beta))
    step = numpy.divide(u, zz)
    return step
