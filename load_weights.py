from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
import os

def load_weights(filename):
    if(os.path.isfile(filename)):
        with open(filename,'r') as file:
            architecture=[]
            numNodes = file.readline().split()
            numInputs = int(numNodes[0])
            numHidden = int(numNodes[1])
            numOutputs = int(numNodes[2])
            
            #initializing layers with appropriate size and class node
            for length in [numInputs,numHidden, numOutputs]:
                layer=[]
                for i in range(length):
                    layer.append(Node())
                architecture.append(layer)
    
            #populating node weights and values for weights coming into node (inputs)
            for layer in range(1,len(architecture)):
                for i,node in enumerate(architecture[layer]):
                    node.weights = [float(j) for j in file.readline().split()]
                    for i,input_node in enumerate(architecture[layer-1]):
                        node.inputs.append(input_node)
                
            architecture = np.array(architecture)           
            print(f'current input layer shape (populated during training): {(len(architecture[0]),architecture[0][0].length())}')
            print(f'hidden layer shape: {(len(architecture[1]),architecture[1][0].length())}')
            print(f'output layer shape: {(len(architecture[2]),architecture[2][0].length())}')
            
            assert len(architecture[0]) == numInputs
            assert len(architecture[1])*architecture[1][0].length() == numHidden*(numInputs+1)
            assert len(architecture[2])*architecture[2][0].length() == numOutputs*(numHidden+1)
            
            return architecture,numInputs, numHidden, numOutputs
