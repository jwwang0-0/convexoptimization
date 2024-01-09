import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import os
PATH = os.path.dirname(__file__)

# load data
A = np.load(os.path.join(PATH,'data/A.npy'))
B = np.load(os.path.join(PATH,'data/B.npy'))
d = np.load(os.path.join(PATH,'data/d.npy'))
R = np.load(os.path.join(PATH,'data/R.npy'))
D = np.load(os.path.join(PATH,'data/D-mat.npy'))
k = np.load(os.path.join(PATH,'data/k.npy'))
N = np.load(os.path.join(PATH,'data/N.npy')).item()
Loc = np.load(os.path.join(PATH,'data/Loc.npy'))
x = np.load('xval.npy')
u = np.load('uval.npy')
V = np.load('Vval.npy')

# +-------------------+
# |  Your Code Here!  |
# +-------------------+

# Method 1

#Step 1 Obtain Y
print(d.shape)
z = d[0:N] / 3
print(z.shape)

#Step 1 Calculate Y
Y = u + V @ z
print(Y.shape)

# Method 2
#Define Variable
x = cp.Variable((N+1,1))
y = cp.Variable((N,N), nonneg = True) #NONNEG TRUE?


constraints = [ 
    x[0:N] <= cp.reshape(k,(N,1)),
    ...
    # obj >= x[N],
    # A@x + B@u - Pi@cp.reshape(d, (N+1,1)) >= 0,
    # B@V >= R - Pi@D,
    # u - Lambda@cp.reshape(d, (N+1,1)) >= 0,
    # Lambda@D + V >= 0
    ]

#Define Objective
objective = cp.Minimize(obj)
prob1 = cp.Problem(objective, constraints)
prob1.solve(solver=cp.GUROBI, verbose=True)




# Retrieve optimal objective value and decisions
opt_obj = prob.value
x_aff = np.hstack([x[0:N], alpha.value])
y_d_val = y_d.value

# Visualization
y_aff = u + V @ (d[0:N] / 3)
y_aff = np.reshape(y_aff, (N, N))

plt.figure(figsize=(12,10))
plt.title('Shipments obtained from Decision Rule Evaluation', fontsize=18)
for i in range(N):
    for j in range(N):
        if y_aff[i, j] > 0.001:
            plt.plot([Loc[0, i], Loc[0, j]], [Loc[1, i], Loc[1, j]], 'k', linewidth=y_aff[i, j] / 2)
plt.scatter(Loc[0, :], Loc[1, :], s=10 * x[0:N] + 0.01, color='b')
plt.scatter(Loc[0, :], Loc[1, :], color='b')
plt.show()

y_det = np.reshape(y_d_val, (N, N))

plt.figure(figsize=(12,10))
plt.title('Shipments obtained from Reoptimization', fontsize=18)
for i in range(N):
    for j in range(N):
        if y_det[i, j] > 0.001:
            plt.plot([Loc[0, i], Loc[0, j]], [Loc[1, i], Loc[1, j]], 'k', linewidth=y_det[i, j] / 2)
plt.scatter(Loc[0, :], Loc[1, :], s=10 * x_aff[0:N] + 0.01, color='b')
plt.scatter(Loc[0, :], Loc[1, :], color='b')
plt.show()
