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


class NeuralNetwork:
    
    def __init__(self):
        super().__init__()
        
    def start(self):
        self.network_file, self.training_set, self.outputFile, self.epochs,self.lr = train_console()
    
    def load_architecture(self,mode = 'train'):
        if(mode == 'train'):
            self.architecture,self.numInputs,self.numHidden,self.numOutputs = load_weights(self.network_file)
        else:
            self.architecture,self.numInputs,self.numHidden,self.numOutputs = load_weights(self.trained_weight_file)
    
    def load_examples(self,mode='train'):
        if(mode == 'train'):
            self.x,self.y = load_dataset(self.training_set)
        else:
            self.x,self.y = load_dataset(self.test_set)
    
    #starting from pseodocode
    
    #propogating inputs forward to compute the outputs
    
    #ai <- xi
    def load_data_to_nodes(self,x):
        for i, node in enumerate(self.architecture[0]):
            node.val = x[i]
    
    #for L = 2 to l do: .....
    def forward_prop(self):
        for layer in self.architecture[1:]:
            for node in layer:
                node.inputVal = -1 * node.weights[0]
                for i, input_node in enumerate(node.inputs):
                    node.inputVal += input_node.val * node.weights[i+1]
                node.val = sigmoid(node.inputVal)
                
    def back_prop(self,delta):
        for i in range(len(self.architecture) - 2, 0, -1):
            curr_layer = self.architecture[i]
            next_layer = self.architecture[i+1]
            weights = []
            for j in range(len(curr_layer)):
                weights.append([node.weights[j+1] for node in next_layer])
                total_sum = 0
                for k, weight in enumerate(weights[j]):
                    total_sum += weight * delta[i+1][k]
                delta[i].append(sigmoid_deriv(curr_layer[j].inputVal) * total_sum)
                
    #update every weight in network using deltas            
    def update_weights(self, delta):
        for i in range(1, len(self.architecture)):
            previous_values = [-1] + [node.val for node in self.architecture[i-1]]
            for j, node in enumerate(self.architecture[i]):
                updates = [self.lr * delta[i][j] * val for val in previous_values]
                node.weights = [node.weights[k] + updates[k] for k in range(len(node.weights))]
                
    def write_to_training_file(self):
        with open(self.outputFile,'w') as file:
            statements = []
            file.write(str(self.numInputs) +" " + str(self.numHidden) + " " + str(self.numOutputs) + "\n")
            for i in range(1,len(self.architecture)):
                for node in self.architecture[i]:
                    statement = ""
                    for i,weight in enumerate(node.weights):
                        if(i==len(node.weights)-1):
                            statement+='{:.3f}'.format(round(weight,3)) + "\n"
                        else:
                            statement+='{:.3f}'.format(round(weight,3)) + " "
                    statements.append(statement)
                
            for phrase in statements:
                file.write(phrase)
    
    def train(self):
        self.start()
        self.load_architecture()
        self.load_examples()
        for j in range(self.epochs):
            for dependent,target in zip(self.x,self.y):
                self.load_data_to_nodes(dependent)
                self.forward_prop()
                
                #propagate deltas backward from output layer to input layer
                
                delta = [[] for layer in self.architecture]
                for i, node in enumerate(self.architecture[2]):
                    delta[2].append(sigmoid_deriv(node.inputVal) * (target[i] - node.val))
                self.back_prop(delta)
                self.update_weights(delta)
            print(f"epoch : {j+1}")
        self.write_to_training_file()
    
    def testing_input(self):
        self.weight_file, self.test_set, self.outputFile = test_console()
        
    def pred(self):
        self.testing_input()
        self.load_examples(mode = "test")
        A,B,C,D = np.zeros((self.numOutputs,1)),np.zeros((self.numOutputs,1)),np.zeros((self.numOutputs,1)),np.zeros((self.numOutputs,1))
        for dependent,target in zip(self.x,self.y):
            prediction = []
            self.load_data_to_nodes(dependent)
            self.forward_prop()
            for i in range(self.numOutputs):
                if(self.architecture[2][i].val < 0.5):
                    prediction.append(0)
                else:
                    prediction.append(1)
            for j, pred in enumerate(prediction):
                if(pred==1 and target[j]==1):
                    A[j]+=1
                elif(pred==1 and target[j]==0):
                    B[j]+=1
                elif(pred==0 and target[j]==1):
                    C[j]+=1
                else:
                    D[j]+=1     
            self.A = A
            self.B = B
            self.C = C
            self.D = D

    
    def write_to_testing_file(self):
        self.pred()
        
        overall_accuracy = [(self.A[i] + self.D[i])/(self.A[i]+self.B[i]+self.C[i]+self.D[i]) for i in range(self.numOutputs)]
        precision = [self.A[i]/(self.A[i]+self.B[i]) for i in range(self.numOutputs)]
        recall = [self.A[i]/(self.A[i]+self.C[i]) for i in range(self.numOutputs)]
        f1 = ([(2*precision[i]*recall[i])/(precision[i]+recall[i]) for i in range(self.numOutputs)])
        
        A_micro, B_micro, C_micro, D_micro = sum(self.A),sum(self.B),sum(self.C),sum(self.D)
        
        micro_overall_accuracy = (A_micro + D_micro)/(A_micro+B_micro+C_micro+D_micro)
        micro_precision = A_micro/(A_micro+B_micro)
        micro_recall = A_micro/(A_micro+C_micro)
        micro_f1 = (2*micro_recall*micro_precision)/(micro_recall + micro_precision)
        
        macro_overall_accuracy = sum(overall_accuracy)/len(overall_accuracy)
        macro_precision = sum(precision)/len(precision)
        macro_recall = sum(recall)/len(recall)
        macro_f1 = (2*macro_recall*macro_precision)/(macro_precision+macro_recall)
        
        with open(self.outputFile,'w') as file:
            for i in range(self.numOutputs):
                file.write(str(int(self.A[i][0])) + " " + str(int(self.B[i][0])) + " " + str(int(self.C[i][0])) + " " + 
                           str(int(self.D[i][0])) + " " + '{:.3f}'.format(round(overall_accuracy[i][0],3)) + " " + '{:.3f}'.format(round(precision[i][0],3)) + " " +
                          '{:.3f}'.format(round(recall[i][0],3)) + " " + '{:.3f}'.format(round(f1[i][0],3)) + "\n")
                
            file.write('{:.3f}'.format(round(micro_overall_accuracy[0],3)) + " " + '{:.3f}'.format(round(micro_precision[0],3)) + " " + 
                        '{:.3f}'.format(round(micro_recall[0],3)) + " " + '{:.3f}'.format(round(micro_f1[0],3)) +"\n")
            file.write('{:.3f}'.format(round(macro_overall_accuracy[0],3)) + " " + '{:.3f}'.format(round(macro_precision[0],3)) + " " + 
                        '{:.3f}'.format(round(macro_recall[0],3)) + " " + '{:.3f}'.format(round(macro_f1[0],3)) + "\n")
                
