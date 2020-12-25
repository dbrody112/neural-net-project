from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
import os

#class to make pseudocode easier to follow
class Node:
    def __init__(self):
        self.inputs = []
        self.weights = []
        self.val = 0
        self.inputVal = 0
    def length(self):
        return len(self.weights)

def load_dataset(filename):
    examples = []
    if(os.path.isfile(filename)):
        with open(filename,'r') as file:
            first_line = file.readline().split()
            numExamples = int(first_line[0])
            numDependentVars = int(first_line[1])
            numTargets = int(first_line[2])
            
            for i in range(numExamples):
                instance = file.readline().split()
                instance = [float(i) for i in instance]
                examples.append(instance)
            examples = np.array(examples).T
            x = examples[:numDependentVars].T
    
            assert x.shape[0]*x.shape[1] == numDependentVars*numExamples
    
            y = examples[numDependentVars:].T.astype('int')
            
            assert y.shape[0]*y.shape[1] == numTargets*numExamples
        
            return x,y
