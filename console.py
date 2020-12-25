from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
import os

def train_console():
    
    training_set = ""
    network_file = ""
    outputFile = ""
    epochs = -1
    learning_rate = -1

    while(os.path.isfile(network_file)==False):
        network_file = input("Enter filename for a neural net: ")
            
    while(os.path.isfile(training_set) == False):
        training_set = input("Enter filename for a training set: ")

    
    outputFile = input("Enter filename for a output file: ")
        
    while(epochs < 0 or type(epochs)!=int):
        try:
            epochs = int(input("How many epochs would you like to train? "))
            print(" ")
        except:
            
            epochs = -1

    while(learning_rate < 0 or type(learning_rate)!=float):
        try:
            learning_rate = float(input("How much do you want the learning rate to be? "))
            print(" ")
        except:
            learning_rate = -1 
            
    return network_file, training_set, outputFile, epochs, learning_rate         

def test_console():
    test_set = ""
    weight_file = ""
    outputFile = ""
    print("Starting test phase......................")
    print("\n")
    
    while(os.path.isfile(weight_file)==False):
        weight_file = input("Enter filename for a weight file: ")
        
    while(os.path.isfile(test_set)==False):
        test_set = input("Enter filename for a test file: ")
    
    
    outputFile = input("Enter filename for an output file to store results: ")
    
    return weight_file, test_set, outputFile
