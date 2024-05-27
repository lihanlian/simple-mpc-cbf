import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
# System parameters
dt = 0.1  # Sampling time
N = 20   # Horizon

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

# robot has radius 0.5, static obstacle has radius 2
def h_static(x):
    pos_x = x[0]
    pos_y = x[1]
    distance = (pos_x - 5)*(pos_x - 5) + (pos_y - 5)*(pos_y - 5)
    return distance - (2 + 0.5)**2

x1_hist = []; x2_hist = []; x3_hist = []; x4_hist = []
ux_hist = []; uy_hist = []
x_ref = np.array([10, 10, 0, 0])  # Reference state
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

    # Initial state equality constraint
    if i == 0:
        x0 = np.array([0,0,0,0])
    else:
        x0 = np.array([x1_hist[-1], x2_hist[-1], x3_hist[-1], x4_hist[-1]])
    i += 1 # increment of time step
    # Inital state constraint
    g.append(X[:n_x] - x0)
    lbg_x0 = [0,0,0,0]
    ubg_x0 = [0,0,0,0]

    # Construct the constrained optimization problem over preidiction horizon
    for k in range(N):
        xk = X[k*n_x:(k+1)*n_x]
        uk = U[k*n_u:(k+1)*n_u]
        xk_next = X[(k+1)*n_x:(k+2)*n_x]
        
        # System dynamics constraint (=0)
        g.append(xk_next - (ca.mtimes(A, xk) + ca.mtimes(B, uk)))
        # Euclidean distance constraint (>=0)
        g.append(h_static(xk))
        
        # Cost function
        if k is not N-1:
            # Stage cost
            cost = ca.mtimes((xk - x_ref).T, ca.mtimes(Q, (xk - x_ref))) + ca.mtimes(uk.T, ca.mtimes(R, uk))
        else:
            # Terminal cost
            cost = ca.mtimes((xk - x_ref).T, ca.mtimes(H, (xk - x_ref)))
        obj += cost
    
    # Add control input constraints
    u_min = -2
    u_max = 2
    g.append(U)

    # For the dynamics constraints: xk+1 = f(xk,uk) and euclidean distance constraints
    lbg_dynamics_and_euclidean = [0, 0, 0, 0, 0] * N 
    ubg_dynamics_and_euclidean = [0, 0, 0, 0, ca.inf] * N  # Same as lbg for equality constraints

    # Add constraints for control input at each time step
    lbg_controls = [u_min] * N * n_u  # Lower bound for each control input
    ubg_controls = [u_max] * N * n_u  # Upper bound for each control input

    # Combine lbg and ubg for both dynamics and control inputs
    lbg = lbg_x0 + lbg_dynamics_and_euclidean + lbg_controls
    ubg = ubg_x0 + ubg_dynamics_and_euclidean + ubg_controls

    # NLP structure
    nlp = {'f': obj, 'x': ca.vertcat(X, U), 'g': ca.vertcat(*g)}
    opts = {'ipopt.print_level':0,'print_time': 0}
    solver = ca.nlpsol('S', 'ipopt', nlp, opts)

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
with open(f'./data/mpc_dc_static_N_{N}.pkl', 'wb') as file:
    pickle.dump(trajectory, file)