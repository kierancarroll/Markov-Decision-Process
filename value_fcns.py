import numpy as np

def sigmoid(x):
  return 1/(1+np.exp(-x))

def compute_vtheta(theta, epsilon, tol=1e-12, max_iter=10000):
    # build transitions for this epsilon
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

    for it in range(max_iter):
        v_new = np.zeros(12)
        for s in range(12):

            pR = sigmoid(s - theta)
            pL = 1 - pR

            Q_L = R_L[s] + T_L[s] @ v
            Q_R = R_R[s] + T_R[s] @ v

            # Bellman update under fixed stochastic policy
            v_new[s] = pL * Q_L + pR * Q_R

        # convergence check
        if np.max(np.abs(v_new - v)) < tol:
            print(f"Converged in {it} iterations")
            break

        v = v_new.copy()

    return v

epsilon = 0.45

vtheta1 = compute_vtheta(2.3, epsilon)
vtheta2 = compute_vtheta(3.5, epsilon)
print(vtheta1)
print(vtheta2)
print(sum(vtheta1))
print(sum(vtheta2))