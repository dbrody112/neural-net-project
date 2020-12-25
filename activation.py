from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
import os

def sigmoid(x):
    return 1/(1+math.exp(-x))
def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))
