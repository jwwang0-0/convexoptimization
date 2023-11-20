import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import os
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

# Load data
x = np.load(os.path.join(DATA_PATH, 'p3x.npy'))
y = np.load(os.path.join(DATA_PATH, 'p3y.npy'))

rho = 0.0001  # regularization parameter
delta = 1     # Huber loss parameter
sigma = 240   # Bandwidth of the Gaussian kernel

# +-----------------+
# | Your Code Here! |
# +-----------------+

#################
#original problem
#################

m = x.shape[0]
#Define Variable
t = cp.Variable((m,1))
r_p = cp.Variable((m,1), nonneg=True)
r_n = cp.Variable((m,1), nonneg=True)
w = cp.Variable(1)
b = cp.Variable(1)
obj = cp.Variable(1)

constraints_0 = [ 
    w*x.reshape((m,1)) + cp.reshape(b * np.ones(m),(m,1)) - cp.reshape(y,(m,1)) - t <= r_p,
    cp.reshape(y,(m,1)) - w*x.reshape((m,1)) - cp.reshape(b * np.ones(m),(m,1)) + t <= r_n,
    1/2 * cp.norm2(t)**2 + delta * np.ones(m).T*(r_p + r_n) + rho/2 * cp.norm2(w)**2 <= obj
    ]

#Define Objective
objective = cp.Minimize(obj)

#Solver
prob = cp.Problem(objective, constraints_0)
prob.solve(solver=cp.MOSEK, verbose=True)
y_predict_method1 = w.value*x.reshape((m,1)) + (b.value * np.ones(m)).reshape((m,1))

plt.figure(figsize=(8,8))
plt.scatter([xi for xi in x],[yi for yi in y], s=15, c = "#1f77b4", label = "Data")
plt.plot([xi for xi in x],[yi for yi in y_predict_method1], label = "Least Squares", linewidth = 2,c = "#2ca02c")

#################
#feature map problem
#################
#Define Variable
beta = cp.Variable((m,1))
obj = cp.Variable(1)

nest_list = []
for i in range(m):
    list = []
    for j in range(m):
        list.append(np.exp(-np.linalg.norm((x[i]-x[j]))**2/2/sigma**2))
    nest_list.append(list)

k = np.array(nest_list)

constraints_1 = [ 
    1/2*cp.quad_form(beta, np.eye(m)) + 1/2/rho*cp.quad_form(beta,cp.psd_wrap(k))- beta.T@y.reshape((m,1)) <= obj,
    cp.sum(beta) == 0,
    beta <= delta*np.ones(m).reshape(m,1),
    beta >= -delta*np.ones(m).reshape(m,1)
    ]

#Define Objective
objective = cp.Minimize(obj)

#Solver
prob2 = cp.Problem(objective, constraints_1)
prob2.solve(solver=cp.MOSEK, verbose=True)

w_phi= []
for j in range(m):
    sum = 0
    for i in range(m):
        sum = sum + beta.value[i] * np.exp(-np.linalg.norm((x[i]-x[j]))**2/2/sigma/sigma)
    w_phi.append(sum/rho)

w_phi_matrix = np.array(w_phi)

b_sum = 0
for i in range(m):
    b_sum = b_sum + np.exp(-np.linalg.norm((x[i]-x[0]))**2/2/sigma/sigma)*beta.value[i]

b_ = -beta.value[0]+y[0]-1/rho*b_sum

y_predict_method2 = w_phi_matrix + b_ * np.ones((m,1))
plt.plot([xi for xi in x],[yi for yi in y_predict_method2], label = "Kernel Trick", linewidth = 2, c = "#ff7f0e")
plt.title("Regression vs Kernel Trick", fontsize=16, fontweight='bold')
plt.legend()
plt.show()