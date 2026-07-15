![Status](https://img.shields.io/badge/status-Completed-brightgreen)
![Field](https://img.shields.io/badge/field-Computer%20Science-blue)

# Markov Decision Process (MDP) – Optimal Policies & Policy Gradient Estimation

This educational project analyzes a circular Markov Decision Process (MDP) resembling a 12-hour clock face, studying how transition noise affects the optimal policy, and how well a simple parameterized (REINFORCE) policy can approximate that optimal policy.

## Background / Context

The state space consists of **12 states arranged on a circle** (like a clock face), where state $0$ (a.k.a. state $12$) is an **absorbing goal state**. From any other state $s$, the agent picks one of two actions:

- **L** — move counter-clockwise to state $(s-1)$ mod $12$
- **R** — move clockwise to state $(s+1)$ mod $12$

**Noisy transitions:** In states $1–4$, the chosen action is executed correctly with probability $1-ε$, and the *opposite* action is executed instead with probability $ε$. States $5–11$ are always deterministic (chosen action is always the one executed), regardless of $ε$.

**Rewards:** Any transition into a non-absorbing state gives an immediate reward of $-1$. A transition into the absorbing state gives an immediate reward of $+10$. The discount factor is set to $γ = 1$ for simplicity.

This project covers two parts of the analysis:

1. **Optimal policy under noise ($ε = 0.2$ and $ε = 0.45$)** — solving the Bellman optimality equations via value iteration to find $v^*$, $q^*$, and $π^*$, and explaining how increasing noise reshapes the optimal policy.
2. **Parameterized policy fit via REINFORCE** — fitting a single-parameter stochastic policy $π_θ(a=R|s) = σ(s - θ)$ (where $σ$ is the sigmoid function) via Monte Carlo policy-gradient estimation, and comparing the resulting $θ^*$ to the true optimal policy found above.

## Project Overview

**Part 1 — Optimal policy under noise:**

Value iteration was run to convergence (summed absolute difference between successive $v^*$ vectors $< 1e-10$) for two noise levels:

```python
epsilon_values = [0.2, 0.45]
```

For each $ε$, the converged value function $v^*$, the action-value function $q^*(s, a)$, and the resulting greedy policy $π^*$ were extracted.

**Part 2 — Parameterized policy via REINFORCE:**

A single-parameter policy $π_θ(a=R|s) = σ(s - θ)$ was fit using the REINFORCE gradient estimator: $∇_θJ(θ) = E_τ[ G(τ) · ∇_θ log π_θ(τ) ]$

```python
num_paths_per_run   = 100000   # paths sampled per θ-optimization
learning_rate alpha  = 0.01
num_runs             = 1000    # independent optimizations, averaged for stability
epsilon_values       = [0.0, 0.45]
```

Each path starts from a uniformly random state and terminates upon reaching the absorbing state; $θ$ is updated via gradient ascent after each full path.

## Results & Conclusions

### Part 1 — Optimal policies under noise

**ε = 0.2**

Optimal policy $π^*$: $[L/R, L, L, L, L, R, R, R, R, R, R, R]$ (states 0–11)

| State | q(s, L) | q(s, R) |
|---|---|---|
| 0 | 0.0 | 0.0 |
| 1 | 9.337 | 7.349 |
| 2 | 7.686 | 5.733 |
| 3 | 6.082 | 4.270 |
| 4 | 4.666 | 3.416 |
| 5 | 3.666 | 4.0 |
| 6 | 3.0 | 5.0 |
| 7 | 4.0 | 6.0 |
| 8 | 5.0 | 7.0 |
| 9 | 6.0 | 8.0 |
| 10 | 7.0 | 9.0 |
| 11 | 8.0 | 10.0 |

**ε = 0.45**

Optimal policy $π^*$: $[L/R, L, L, L, R, R, R, R, R, R, R, R]$ (states 0–11)

| State | q(s, L) | q(s, R) |
|---|---|---|
| 0 | 0.0 | 0.0 |
| 1 | 6.595 | 5.838 |
| 2 | 3.433 | 2.953 |
| 3 | 1.791 | 1.648 |
| 4 | 1.785 | 2.006 |
| 5 | 1.006 | 4.0 |
| 6 | 3.0 | 5.0 |
| 7 | 4.0 | 6.0 |
| 8 | 5.0 | 7.0 |
| 9 | 6.0 | 8.0 |
| 10 | 7.0 | 9.0 |
| 11 | 8.0 | 10.0 |

**Conclusions:**
- At $ε = 0$ (deterministic case), the optimal policy simply takes the shortest path to the absorbing state: $L$ for states 1–5, $R$ for states 7–11, and either action for state 6 (equidistant both ways).
- As noise is introduced, states near the "stochastic zone" (states 1–4) start to prefer routing *away* from the noisy region and through the fully deterministic states (5–11) instead, since oscillating among noisy states repeatedly costs $-1$ per extra step.
- At $ε = 0.2$, state 5 switches from $L$ to $R$ compared to the deterministic case (state 6 remains indifferent). This can be seen as $q(5, L) < q(5, L)$ now.
- At $ε = 0.45$, the effect is stronger: state 4 *also* switches to $R$, since the expected cost of oscillating through the noisy states 1–4 now outweighs the shorter nominal path length. Again, this can be seen as $q(4, L) < q(4, L)$ now.
- In short: **the higher the noise, the more states "give up" on the short path through the noisy region and instead route through the guaranteed-deterministic states**, even if that path is nominally longer.

### Part 2 — REINFORCE fit of the parameterized policy $π_θ(a=R|s) = σ(s-θ)$

| Noise level $ε$ | Estimated $θ^*$ (averaged over 1000 runs) |
|---|---|
| 0.0 | ≈ 6.0 |
| 0.45 | ≈ 2.3 |

**Conclusions:**
- At $ε = 0$, $θ^* ≈ 6$ matches the true optimal policy well: $σ(6-6) = 0.5$ reproduces the indifference at state 6, $σ(s-6) < 0.5$ for $s ≤ 5$ (favoring L, as in $π^*$), and $σ(s-6) > 0.5$ for $s ≥ 7$ (favoring R, as in $π*$).
- At $ε = 0.45$, $θ*$ decreases to ≈ 2.3, correctly shifting the "decision boundary" downward so that states 4, 5, 6 now favor R — consistent with the noisy optimal policy found in Part 1.
- However, $θ^* ≈ 2.3$ implies $π_θ(R|s=3) = σ(0.7) > 0.5$, i.e. the parameterized policy prefers $R$ in state 3, which **disagrees** with the true optimal policy (which picks $L$ in state 3 at $ε = 0.45$).
- This mismatch arises from the policy's **shared single parameter across all states**: $θ$ cannot be tuned per-state, so the optimizer trades a *locally* suboptimal action in state 3 for a *globally* higher expected return by keeping neighboring, more consequential states (4, 5, 6, …) reliably locked into the deterministic route. This was confirmed numerically: the expected return norm $‖v_π‖$ is higher for $θ = 2.3$ (≈59.4) than for $θ = 3.5$ (≈58.1), even though $θ = 3.5$ would make state 3 individually optimal.
- **Takeaway:** the parameterized policy captures the *overall trend* of the true optimal policy well, but its limited expressiveness (one shared parameter for 12 states) forces it to sacrifice per-state optimality in favor of maximizing global expected return.

## Repository Structure
```
project-root/
├── MDP_optimal_policy.py             # Value iteration using Bellman equations
├── MDP_parametrized_policy.py        # Finds optimal parametrized policy using REINFORCE algorithm
├── value_fcns.py                     # Numerically confirms conclusions about REINFORCE tradeoff conclusion
├── requirements.txt                  # Install all required packages and libraries
├── .gitignore
└── README.md
```

## Usage

### 1. Clone the repository

```bash
git clone https://github.com/kierancarroll/Markov-Decision-Process.git
cd /Markov-Decision-Process
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the full experiments:

```bash
python  MDP_optimal_policy.py
python  MDP_parametrized_policy.py
python  value_fcns.py
```

This will:
- Compute $v^*$, $q^*$, and $π^*$ for $ε = 0.2$ and $ε = 0.45$ via value iteration
- Run REINFORCE to estimate $θ^*$ for the parameterized policy at $ε = 0.0$ and $ε = 0.45$
- Print all resulting values, q-tables, policies, and $θ^*$ estimates