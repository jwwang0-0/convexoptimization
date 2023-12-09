import cvxpy as cp
import numpy as np
import random
import time
import networkx as nx
import matplotlib.pyplot as plt
from integer_solver import integer_solver
import os


# Data of coordinates (in meters)
points = np.load(os.path.join(os.path.dirname(__file__),'points_small.npy'))
#points = np.load('points_large.npy')
print(points)

# Adjacency Matrix - A is matrix with components A_ij. A_ij=1 if (i,j) is connected; 0 otherwise.
A = np.array([[ 1 if np.linalg.norm(i-j,2)<=25 else 0 for j in points] for i in points])
print(A)
    
# Draw the graph            
G = nx.from_numpy_array(A)
pos = {i: p for i,p in enumerate(points)}
nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=300, node_color='lightblue', font_size=7, font_color='black', edge_color='gray', linewidths=0.5)
plt.show()
        


#
""" SDP Relaxation """
# +-----------------+
# | Your Code Here! |
# +-----------------+

#Define Variable
n = len(points)
X = cp.Variable((n,n),symmetric=True)
x = cp.Variable((n,1))
obj = cp.Variable(1)
one = np.array([1])
matrix = cp.vstack([cp.hstack([cp.reshape(one,(1,1)), x.T]), cp.hstack([x, X])])

constraints = [ 
    obj <= cp.trace(X),
    matrix >> 0
    ]
constraints.extend( [ X[i][i]==x[i] for i in range(n)] )
for i in range(n):
    for j in range(n):
        if A[i][j] == 1:
            constraints.append(X[i][j]==0)


#Define Objective
objective = cp.Maximize(obj)

    
start = time.time()
prob = cp.Problem(objective, constraints) #FILL
prob.solve(solver=cp.MOSEK, verbose=True) #FILL
end = time.time()

print("Total time SDP: ", end-start) 
print("x value: ",x.value)
print("X value:" ,X.value )


# #%% 
# """ Rounding Heuristic """

# # +-----------------+
# # | Your Code Here! |
# # +-----------------+


# """ Draw the graph with the selected locations """


# selected_locations = 

# nx.draw_networkx_nodes(G, pos, nodelist=set(range(len(A))), node_size=200, node_color='lightblue')
# nx.draw_networkx_edges(G, pos, edge_color='gray', width=0.5)
# nx.draw_networkx_labels(G, pos, font_weight='bold', font_size=6)

# # Selected locations are lightgreen in your graph
# nx.draw_networkx_nodes(G, pos, nodelist=np.array(selected_locations), node_size=200, node_color='lightgreen')

# plt.show()


# print("Optimal value of IP is between ", your_opt_value_from_rounding_heuristic, " and ", np.floor(your_opt_val_from_sdp))


# #%%
# """ Integer Solver """
# y_val, prob_ip_val = integer_solver(A)

# # +------------------------+
# # | Plot the graph from IP |
# # +------------------------+
