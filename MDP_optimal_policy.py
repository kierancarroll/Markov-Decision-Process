#Q2.3
import numpy as np

#Bellman equations 12x12 transition matrix. State 0 is terminal, 1-4 stochastic, 5-11 deterministic.

def compute_values(epsilon):
    """
    Computes v* and optimal policy for noise level epsilon.
    """

    T_L = np.array([
    [1,0,0,0,0,0,0,0,0,0,0,0],
    [1-epsilon,0,epsilon,0,0,0,0,0,0,0,0,0],
    [0,1-epsilon,0,epsilon,0,0,0,0,0,0,0,0],
    [0,0,1-epsilon,0,epsilon,0,0,0,0,0,0,0],
    [0,0,0,1-epsilon,0,epsilon,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,0]
    ])

    T_R = np.array([
        [1,0,0,0,0,0,0,0,0,0,0,0],
        [epsilon,0,1-epsilon,0,0,0,0,0,0,0,0,0],
        [0,epsilon,0,1-epsilon,0,0,0,0,0,0,0,0],
        [0,0,epsilon,0,1-epsilon,0,0,0,0,0,0,0],
        [0,0,0,epsilon,0,1-epsilon,0,0,0,0,0,0],
        [0,0,0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0]
    ])

    R_L = np.array([0, 10-11*epsilon, -1, -1,-1, -1, -1, -1, -1, -1, -1, -1])
    R_R = np.array([0, 11*epsilon-1, -1, -1,-1, -1, -1, -1, -1, -1, -1, 10])

    v = np.zeros(12)

    for i in range(10000):
        v_new = np.zeros(12)
        for s in range(12):
            # Q-values for both actions at state s
            Q_L = R_L[s] + T_L[s] @ v
            Q_R = R_R[s] + T_R[s] @ v

            v_new[s] = max(Q_L, Q_R)

        if np.max(np.abs(v - v_new)) < 1e-12:
            print(f"Converged in {i} iterations.")
            break

        v = v_new

    # extract policy
    policy = []
    for s in range(12):
        Q_L = R_L[s] + T_L[s] @ v
        Q_R = R_R[s] + T_R[s] @ v
        print(f"State {s}: Q_L = {Q_L}, Q_R = {Q_R}")
        # if Q_L == Q_R:
            # print(f"State {s} has equal Q-values for both actions.")
        policy.append("L" if Q_L >= Q_R else "R") #ties broken aribitrarily


    return v, policy

v_02, pi_02 = compute_values(0.2)
print("Value function ε=0.2:", v_02)
print("Policy ε=0.2:", pi_02)
v_045, pi_045 = compute_values(0.45)
print("Value function ε=0.45:", v_045)
print("Policy ε=0.45:", pi_045)