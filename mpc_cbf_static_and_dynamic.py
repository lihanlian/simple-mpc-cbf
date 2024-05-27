import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import math
# System parameters
dt = 0.1  # Sampling time
N = 10     # Horizon
gamma = 0.9 # CBF parameter

# System matrices (in discrete time)
A = np.array([[1, 0, 0.1, 0],
            [0, 1, 0, 0.1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
B = np.array([[0.005, 0],
              [0, 0.005],
              [dt, 0],
              [0, dt]])
n_x = A.shape[1]
n_u = B.shape[1]

# robot has radius 0.5, moving obstacle has raidus 1, static obstacle has radius 2
def h_static(X):
    x = X[0]
    y = X[1]
    return (x-2)*(x-2) + (y-4)*(y-4) - (2+0.5)**2

# return center position of moving obstacle
def moving_obstacle(i):
    y = 6
    postion_in_cycle = i % 40
    if postion_in_cycle <= 20:
        x = 7 + (2 * postion_in_cycle)/20
    else:
        x = 9 - (2 * (postion_in_cycle-20))/20
    return x, y

def h_dynamic(x,i):
    pos_x = x[0]
    pos_y = x[1]
    move_obstacle_x, move_obstacle_y = moving_obstacle(i)
    distance = (pos_x - move_obstacle_x)*(pos_x - move_obstacle_x) + (pos_y - move_obstacle_y)*(pos_y - move_obstacle_y)
    return distance - (1 + 0.5)**2

x1_hist = []; x2_hist = []; x3_hist = []; x4_hist = []
ux_hist = []; uy_hist = []
x_ref = np.array([10, 10, 0, 0])  # goal state
current_state = np.array([0,0,0,0]) # initial state

total_cost = 0 # record total cost
i = 0 # record total timesteps
while np.linalg.norm(x_ref - current_state) > 0.1:

    # Decision variables
    U = ca.MX.sym('U', n_u*N)
    X = ca.MX.sym('X', n_x*(N+1))

    # Initialize objective and constraints
    obj = 0
    g = []
    lbg = []; ubg = []
    Q = np.array([[10, 0, 0, 0], 
                  [0, 10, 0, 0], 
                  [0, 0, 2, 0],
                  [0, 0, 0, 2]])  # State weighting
    R = np.eye(n_u)  # Control input weighting
    H = 100*np.eye(4) # Terminal cost matrix

    # Initial state
    if i == 0:
        x0 = np.array([0,0,0,0])
    else:
        x0 = np.array([x1_hist[-1], x2_hist[-1], x3_hist[-1], x4_hist[-1]])
    i += 1 # increment of time step
    # Initial state contraints
    g.append(X[:n_x] - x0)
    lbg_x0 = [0,0,0,0]
    ubg_x0 = [0,0,0,0]

    # Construct the constrained optimization problem over preidiction horizon
    for k in range(N):
        xk = X[k*n_x:(k+1)*n_x]
        uk = U[k*n_u:(k+1)*n_u]
        xk_next = X[(k+1)*n_x:(k+2)*n_x]
        
        # Dynamics constraint (=0)
        g.append(xk_next - (ca.mtimes(A, xk) + ca.mtimes(B, uk)))
        # Get h for CBF constraints
        hk_static = h_static(xk)
        hk_dynamic = h_dynamic(xk,i)

        # dhdx_static = ca.MX([2*x - 4, 2*y - 8, 0, 0])
        # dhdx_dyanmic = ca.MX([2*x - 2*moving_obstacle_x, 2*y - 2*moving_obstacle_y, 0, 0])
        moving_obstacle_x, moving_obstacle_y =  moving_obstacle(i)
        Axk = ca.mtimes(A,xk); Buk = ca.mtimes(B,uk)
        # dh/dx = dh/dx * dx/dt
        dhdt_static = ca.mtimes(2*xk[0] - 4, Axk[0]) + ca.mtimes(2*xk[1] - 8, Axk[1]) + \
              ca.mtimes(2*xk[0] - 4, Buk[0]) + ca.mtimes(2*xk[1] - 8, Buk[1])
        dhdt_dynamic = ca.mtimes(2*xk[0] - 2*moving_obstacle_x, Axk[0]) + ca.mtimes(2*xk[1] - 2*moving_obstacle_y, Axk[1]) +  \
            ca.mtimes(2*xk[0] - 2*moving_obstacle_x, Buk[0]) + ca.mtimes(2*xk[1] - 2*moving_obstacle_y, Buk[1])
        # CBF constraints: dh/dt + gama*h >= 0
        g.append(dhdt_static + gamma*hk_static)
        g.append(dhdt_dynamic + gamma*hk_dynamic)
        
        # Cost function
        if k is not N-1:
            # stage cost
            cost = ca.mtimes((xk - x_ref).T, ca.mtimes(Q, (xk - x_ref))) + ca.mtimes(uk.T, ca.mtimes(R, uk))
        else:
            # terminal cost
            cost = ca.mtimes((xk - x_ref).T, ca.mtimes(H, (xk - x_ref)))
        obj += cost

    # Add control input constraints
    u_min = -2
    u_max = 2
    g.append(U)


    # For the system dynamics constraints: xk+1 = f(xk,uk) and cbf constraints
    lbg_dynamics_and_cbf = [0, 0, 0, 0, 0, 0] * N 
    ubg_dynamics_and_cbf = [0, 0, 0, 0, ca.inf, ca.inf] * N  # Same as lbg for equality constraints

    # Add constraints for control input at each time step
    lbg_controls = [u_min] * N * n_u  # Lower bound for each control input
    ubg_controls = [u_max] * N * n_u  # Upper bound for each control input

    # Combine lbg and ubg for dynamics, cbf and control inputs constraints
    lbg = lbg_x0 + lbg_dynamics_and_cbf + lbg_controls
    ubg = ubg_x0 + ubg_dynamics_and_cbf + ubg_controls

    # NLP structure
    nlp = {'f': obj, 'x': ca.vertcat(X, U), 'g': ca.vertcat(*g)}
    opts = {'ipopt.print_level':0,'print_time': 0}
    solver = ca.nlpsol('S', 'ipopt', nlp,opts)

    # Solve the NLP
    sol = solver(lbg=lbg, ubg=ubg)
    x_opt = sol['x'][:n_x*(N+1)].full().flatten()
    u_opt = sol['x'][n_x*(N+1):].full().flatten()
    total_cost += sol['f']
    current_state = np.array([x_opt[0], x_opt[1], x_opt[2], x_opt[3]])
    print(current_state)
    x1_hist.append(x_opt[4])
    x2_hist.append(x_opt[5])
    x3_hist.append(x_opt[6])
    x4_hist.append(x_opt[7])
    ux_hist.append(u_opt[0])
    uy_hist.append(u_opt[1])

print(f"total time steps to reach goal state: {i}")
print(f'total cost of operation: {total_cost}')
trajectory = {
    'x': x1_hist,
    'y': x2_hist,
    'dx': x3_hist,
    'dy': x4_hist,
    'ux': ux_hist,
    'uy': uy_hist
}
with open(f'./data/mpc_cbf_static_and_dynamic_N_10_gamma_{gamma}.pkl', 'wb') as file:
    pickle.dump(trajectory, file)
