import cvxpy as cp
import time
import numpy as np

def integer_solver(adjancy_matrix):
    num_sensors = adjancy_matrix.shape[0]
    y = cp.Variable(num_sensors, boolean=True)

    objective_ip = cp.Maximize(cp.sum(y))
    constraint_ip = [adjancy_matrix[i, j] * (y[i] + y[j]) <= 1 for i in range(num_sensors) for j in range(num_sensors)]
   

    start_ip = time.time()
    prob_ip = cp.Problem(objective_ip, constraint_ip)
    prob_ip_value = prob_ip.solve(solver=cp.MOSEK, verbose=True)
    
    end_ip = time.time()

    print("Total time IP: ", end_ip-start_ip) 

    y_val = y.value

    return y_val, prob_ip_value

