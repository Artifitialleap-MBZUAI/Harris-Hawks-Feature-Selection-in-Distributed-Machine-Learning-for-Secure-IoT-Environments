
# The Original code of ELM for Author: Li Xudong, from NSSC.CAS Beijing
# and we make small update to fits with our code



import numpy as np
from scipy.linalg import  inv 
import torch.nn as nn

class RWN(nn.Module):
    
    def __init__(self, x, y, hidden_units=1024,   C=0.1, one_hot=True):
        self.hidden_units = hidden_units
        
        self.C = C
        self.x = x
        self.y = y
        
        self.one_hot = one_hot
        self.class_num = np.unique(self.y).shape[0]  

        self.beta = np.zeros((self.hidden_units, self.class_num))  

        if  self.one_hot:
            self.one_hot_label = np.zeros((self.y.shape[0], int(max(np.unique(self.y))+1)))

            for i in range(self.y.shape[0]):
                self.one_hot_label[i, int(self.y[i])] = 1

        self.W = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, self.x.shape[1]))
        self.b = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, 1))

        
    def __input2hidden(self, x):
       
            
        self.temH = np.dot(self.W, x.T) + self.b  
        
        self.H = self.temH * (self.temH > 0) 
        return self.H

    def __hidden2output(self, H):
        self.output = np.dot(H.T, self.beta)
        return self.output

    def fit(self):
       
        self.H = self.__input2hidden(self.x)
        

        if self.one_hot:
            self.y_temp = self.one_hot_label
        else:
            self.y_temp = self.y


        self.tmp1 = inv(np.eye(self.H.shape[0])/self.C + np.dot(self.H, self.H.T))
        self.tmp2 = np.dot(self.H.T, self.tmp1)
        self.beta = np.dot(self.tmp2.T, self.y_temp)

        self.result = self.__hidden2output(self.H)        
        self.result = np.exp(self.result)/np.sum(np.exp(self.result), axis=1).reshape(-1, 1)

        self.y_ = np.where(self.result == np.max(self.result, axis=1).reshape(-1, 1))[1]
        self.correct = 0
        for i in range(self.y.shape[0]):
            if self.y_[i] == self.y[i]:
                self.correct += 1
        self.train_score = self.correct/self.y.shape[0]
        
        return self.beta, self.train_score, self.W, self.b


    def predict(self, x):
        self.H = self.__input2hidden(x)
        self.y_ = self.__hidden2output(self.H)
        b = np.max(self.y_, axis=1)
        data = []
        for i in range(len(self.y_)):
            for j in range (len(self.y_[2])):
                elem = b[i]
                # Check if items matches the given element
                if self.y_[i][j] == elem:
                    pos = j
                    data.append(pos)
                    break      
        return data


    def score(self, x, y):
        self.prediction = self.predict(x)
        
        self.correct = 0
        for i in range(y.shape[0]):
            if self.prediction[i] == y[i]:
                self.correct += 1
        self.test_score = self.correct/y.shape[0]
        
        return self.test_score

