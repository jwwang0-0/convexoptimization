# Required Python packages
import numpy as np
import cvxpy as cp
import os
import matplotlib.pyplot as plt
import time as t
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

# Load data

x = np.load(os.path.join(DATA_PATH, 'p2x.npy'))
y = np.load(os.path.join(DATA_PATH, 'p2y.npy'))

rho = 1   # regularization parameter
delta = 0.1 # Huber loss parameter

# +-----------------+
# | Your Code Here! |
# +-----------------+

#Problem 2 implementation
m = x.shape[0]
n = x.shape[1]

#Define Variable
t = cp.Variable((m,1))
r_p = cp.Variable((m,1), nonneg=True)
r_n = cp.Variable((m,1), nonneg=True)
w = cp.Variable(1)
b = cp.Variable(1)
obj = cp.Variable(1)

constraints_1 = [ 
    w*x + cp.reshape(b * np.ones(m),(m,1)) - cp.reshape(y,(m,1)) - t <= r_p,
    cp.reshape(y,(m,1)) - w*x - cp.reshape(b * np.ones(m),(m,1)) + t <= r_n,
    1/2 * cp.norm2(t) + delta * np.ones(m).T*(r_p + r_n) + rho/2 * cp.norm2(w) <= obj
    ]

constraints_2 = [ 
#    w*x + cp.reshape(b * np.ones(m),(m,1)) - cp.reshape(y,(m,1)) - t <= r_p,
#    cp.reshape(y,(m,1)) - w*x - cp.reshape(b * np.ones(m),(m,1)) + t <= r_n,
    1/2 * cp.norm2(w*x+cp.reshape(b * np.ones(m),(m,1)) - cp.reshape(y,(m,1))) +  rho/2 * cp.norm2(w) <= obj
    ]

#Define Objective
objective = cp.Minimize(obj)
prob1 = cp.Problem(objective, constraints_1)
prob2 = cp.Problem(objective, constraints_2)
prob1.solve(solver=cp.MOSEK, verbose=True)
y_predict_method2 = w.value*x + (b.value * np.ones(m)).reshape((m,1))

prob2.solve(solver=cp.MOSEK, verbose=True)
y_predict_method1 = w.value*x + (b.value * np.ones(m)).reshape((m,1))

plt.figure(figsize=(8,8))
plt.scatter([xi for xi in x],[yi for yi in y], s=15, c = "#1f77b4")
plt.scatter([xi for xi in x],[yi for yi in y_predict_method2], s=15, c = "#ff7f0e")
plt.scatter([xi for xi in x],[yi for yi in y_predict_method1], s=15, c = "#2ca02c")

plt.show()

#Problem 1 implementation
