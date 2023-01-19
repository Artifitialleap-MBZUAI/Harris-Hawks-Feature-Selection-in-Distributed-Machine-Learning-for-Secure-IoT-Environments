
from typing import Tuple, Union, List
import numpy as np
from sklearn.model_selection import train_test_split


XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
RWNParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]
reduced_data_test_global = []
ytest_global = []


def chunkIt(seq, parts_num):
    th_hold = int(seq/parts_num)
    df_sample = seq.sample(n=th_hold,replace=False)
        
    return df_sample
 
def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions)))


def load_data():
    DatasetSplitRatio=0.34
    path = 'Breastcancer.csv'
    data_set = np.loadtxt(path, delimiter = ",")
    numRowsData=np.shape(data_set)[0]    
    numFeaturesData=np.shape(data_set)[1]-1 

    dataInput=data_set[0:numRowsData,0:-1]
    dataTarget=data_set[0:numRowsData,-1]  
       
    X_train, X_test, y_train, y_test = train_test_split(dataInput, dataTarget, test_size=DatasetSplitRatio, random_state=1) 
   
    return numFeaturesData, X_train, y_train, X_test , y_test

def reduction( trainInput , testInput , features, dimF):
    numof_F= 0
    reducedfeatures=[]
    for index in range(0,dimF):
        if (features[index]==1):
            reducedfeatures.append(index)
            numof_F = numof_F+1
            
    reduced_data_train_global=trainInput[:,reducedfeatures]
    reduced_data_test_global=testInput[:,reducedfeatures]
    
    return reduced_data_train_global, reduced_data_test_global




