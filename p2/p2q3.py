import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Load data
x = np.load('data/p3x.npy')
y = np.load('data/p3y.npy')

rho = 0.0001  # regularization parameter
delta = 1     # Huber loss parameter
sigma = 240   # Bandwidth of the Gaussian kernel

# +-----------------+
# | Your Code Here! |
# +-----------------+