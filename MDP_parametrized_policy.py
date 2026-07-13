#Q2.4
import numpy as np
import random

#seed randomness for reproducibility
random.seed(42)
np.random.seed(42)

def sigmoid(x):
  return 1/(1+np.exp(-x))

def run_MC(num_iterations, theta, epsilon, alpha):

  for _ in range(num_iterations):
    s_0 = random.randint(0,11)
    s = s_0
    grad_J = 0

    while s != 0:
      p_R = sigmoid(s - theta)
      move = np.random.binomial(size=1, n=1, p= p_R)[0]
      if move == 1:
        grad_log_pi_wrt_theta = p_R - 1
      else:
        grad_log_pi_wrt_theta = p_R

      if s in [1,2,3,4]:
        execute = np.random.binomial(size=1, n=1, p= (1-epsilon))[0]
        if (move == 1 and execute == 1) or (move == 0 and execute == 0):
          s = (s+1) % 12
        else:
          assert (move == 1 and execute == 0) or (move == 0 and execute == 1), "INVALID MOVE"
          s = (s-1) % 12
        if s == 0:
          grad_J += 10*grad_log_pi_wrt_theta
        else:
          grad_J += -1*grad_log_pi_wrt_theta
      else:
        if move == 1:
          s = (s+1) % 12
        else:
          s = (s-1) % 12

        if s == 0:
          grad_J += 10*grad_log_pi_wrt_theta
        else:
          grad_J += -1*grad_log_pi_wrt_theta

    theta += alpha*grad_J

  return theta


thetas = []
for i in range(100000):
  theta = run_MC(1000, 0, 0.45, 0.1)
  thetas.append(theta)
print(f"theta optimal = {np.mean(thetas)}")
# theta = run_MC(1000, 0, 0.45, 0.1)
# print(f"theta optimal = {theta}")