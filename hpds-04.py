# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 09:59:55 2020
@author: Taufik Sutanto
"""
import warnings; warnings.simplefilter('ignore')
from numba import jit, cuda 
import numpy as np, os
# to measure exec time 
from timeit import default_timer as timer    
  
# normal function to run on cpu 
def func(a):                                 
    for i in range(10000000): 
        a[i]+= 1      
  
# function optimized to run on gpu  
@jit(target ="cuda")                          
def func2(a): 
    for i in range(10000000): 
        a[i]+= 1
        
if __name__=="__main__":     
    os.environ['NUMBAPRO_NVVM']      = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\nvvm\bin\nvvm64_33_0.dll'
    os.environ['NUMBAPRO_LIBDEVICE'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\nvvm\libdevice'    
    
    n = 10000000                            
    a = np.ones(n, dtype = np.float64) 
    b = np.ones(n, dtype = np.float32) 
      
    start = timer() 
    func(a) 
    print("without GPU:", timer()-start)     
      
    start = timer() 
    func2(a) 
    print("with GPU:", timer()-start) 