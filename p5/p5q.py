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
#points = np.load(os.path.join(os.path.dirname(__file__),'points_large.npy'))


# Adjacency Matrix - A is matrix with components A_ij. A_ij=1 if (i,j) is connected; 0 otherwise.
A = np.array([[ 1 if (np.linalg.norm(i-j,2)<=25) else 0 for j in points] for i in points])
for i in range(len(points)):
    A[i][i] = 0
    
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
XX = cp.Variable((n,n),symmetric=True)
x = cp.Variable((n,1))
obj = cp.Variable(1)
one = np.array([1])
matrix = cp.bmat([[cp.reshape(one,(1,1)), x.T], [x, XX]])

constraints = [ 
    obj <= cp.trace(XX),
    matrix >> 0
    ]
constraints.extend( [ XX[i][i]==x[i] for i in range(n)] )
for i in range(n):
    for j in range(n):
        if A[i][j] == 1:
            constraints.append(XX[i][j]==0)

#Define Objective
objective = cp.Maximize(obj)

    
start = time.time()
prob = cp.Problem(objective, constraints) #FILL
prob.solve(solver=cp.MOSEK, verbose=True) #FILL
end = time.time()

print("Total time SDP: ", end-start) 
#print("x value: ",x.value)
#print("XX value: " ,XX.value)
#print("Obj value: ", obj.value)


""" Rounding Heuristic """

# +-----------------+
# | Your Code Here! |
# +-----------------+
L = []
V = [i for i in range(n)]

for i in range(n):
    if V == []:
        break
    check = np.array([x.value[j] for j in V])
    ind = np.argmax(check)
    i_star = V[ind]
    count = 0
    for k in range(n):
        if A[i_star][k] == 1 and k in L:
            count += 1
    if count > 0:
        V.pop(ind)
    else:
        L.append(i_star)
        V.pop(ind)
    # print("Iteration",i,"L: ",L, "V: ", V)

    
    


""" Draw the graph with the selected locations """


selected_locations = L

nx.draw_networkx_nodes(G, pos, nodelist=set(range(len(A))), node_size=200, node_color='lightblue')
nx.draw_networkx_edges(G, pos, edge_color='gray', width=0.5)
nx.draw_networkx_labels(G, pos, font_weight='bold', font_size=6)

# Selected locations are lightgreen in your graph
nx.draw_networkx_nodes(G, pos, nodelist=np.array(selected_locations), node_size=200, node_color='lightgreen')

plt.show()


print("Optimal value of IP is between ", len(L), " and ", np.floor(obj.value))


""" Integer Solver """
y_val, prob_ip_val = integer_solver(A)

#print(y_val, prob_ip_val)
# +------------------------+
# | Plot the graph from IP |
# +------------------------+
new_locations = []
for ind, val in enumerate(y_val):
    if abs(val - 1) < 0.0001:
        new_locations.append(ind) 


nx.draw_networkx_nodes(G, pos, nodelist=set(range(len(A))), node_size=200, node_color='lightblue')
nx.draw_networkx_edges(G, pos, edge_color='gray', width=0.5)
nx.draw_networkx_labels(G, pos, font_weight='bold', font_size=6)

sdp_u = np.setdiff1d(np.array(selected_locations), np.array(new_locations))
ip_u = np.setdiff1d(np.array(new_locations), np.array(selected_locations))
common = np.intersect1d(np.array(new_locations), np.array(selected_locations))
print("old: ", selected_locations)
print("new: ", new_locations)
print("common: ", common)
print("SDP: ", sdp_u)
print("ip_u: ", ip_u)
# Selected locations are lightgreen in your graph
nx.draw_networkx_nodes(G, pos, nodelist=common, node_size=200, node_color='lightgreen', label='SDP & IP')
if len(sdp_u) != 0:
    nx.draw_networkx_nodes(G, pos, nodelist=sdp_u, node_size=200, node_color='lightyellow', label='SDP only')
if len(ip_u) != 0:
    nx.draw_networkx_nodes(G, pos, nodelist=ip_u, node_size=200, node_color='lightpink', label='IP only')
#plt.legend()
plt.show()
