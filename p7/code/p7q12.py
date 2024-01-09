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

# +-------------------+
# |  Your Code Here!  |
# +-------------------+
print(A.shape)
print(B.shape)
print(d.shape)
print(R.shape)
print(D.shape)
print(k.shape)
print(N)

#Define Variable
x = cp.Variable((N+1,1))
u = cp.Variable((N*N,1))
V = cp.Variable((N*N,N))
Pi = cp.Variable((N+1,N+1), nonneg = True)
Lambda = cp.Variable((N*N,N+1), nonneg = True)
obj = cp.Variable((1))

constraints = [ 
# NEED TO ADD k to the problem
    x[0:N] <= cp.reshape(k,(N,1)),
    obj >= x[N],
    A@x + B@u - Pi@cp.reshape(d, (N+1,1)) >= 0,
    B@V >= R - Pi@D,
    u - Lambda@cp.reshape(d, (N+1,1)) >= 0,
    Lambda@D + V >= 0
    ]

#Define Objective
objective = cp.Minimize(obj)
prob1 = cp.Problem(objective, constraints)
prob1.solve(solver=cp.GUROBI, verbose=True)

x_val = x.value
u_val = u.value
V_val = V.value
print(obj.value)

# Save optimal decisions to npy-file
np.save('xval.npy', x_val)
np.save('uval.npy', u_val)
np.save('Vval.npy', V_val)
np.save('Obj.npy', obj.value )


# Visualization

# Visualize the locations of the stores and the stock allocations to each store
plt.figure(figsize=(12, 10))
plt.scatter(Loc[0,:], Loc[1,:], s=10 * x_val[0:N]+0.01, color='b')  # the +0.01 is to avoid errors in the scatter plot if some x(i)=0
plt.scatter(Loc[0,:], Loc[1,:], facecolors='none', edgecolors='b')
plt.title('Stock Allocations from Affine Policies', fontsize=18)
plt.show()

