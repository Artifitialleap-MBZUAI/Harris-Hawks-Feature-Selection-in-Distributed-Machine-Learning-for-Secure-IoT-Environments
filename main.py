# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 11:53:20 2022

@author: Neveen.Hijazi
"""

import warnings

import os

from multiprocessing import Pool


warnings.filterwarnings("ignore", category=UserWarning)
   

def run_process(process):
    os.system('python {}'.format(process))

NumOfUsers = 3
processes = []


    

if __name__ == "__main__":
    print("Please wait it takes time...")

    for c in range (1, NumOfUsers+1):
       processes.append('user.py')   
    
    pool = Pool(processes = NumOfUsers)
    pool.map(run_process, processes)
    pool.close()
    pool.join()

    




