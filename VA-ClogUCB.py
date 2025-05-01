import numpy as np
import matplotlib.pyplot as plt
import timeit
from scipy.special import expit as sigmoid
from scipy.optimize import minimize

# Problem-specific functions for the Combinatorial Logistic Bandit (CLogB) problem
def cal_inf(G, S_A):
    """Calculate the expected reward (influence) of super arm S_A."""
    # Computes the reward as the sum of probabilities that at least one arm in S_A covers each target node
    return np.sum(1 - np.prod(1 - G[S_A], axis=0))

def greedy(G, k_A):
    """Greedily select k_A arms to maximize the expected reward."""
    n_s = G.shape[0]  # Number of base arms
    S_A = []  # Initialize empty super arm
    for _ in range(k_A):  # Select k_A arms
        inf_out = np.zeros(n_s)  # Store influence for each candidate arm
        for i in range(n_s):  # Evaluate each arm not yet selected
            if i not in S_A:
                inf_out[i] = cal_inf(G, S_A + [i])  # Compute influence if arm i is added
        S_A.append(np.argmax(inf_out))  # Add arm with maximum influence
    return S_A, cal_inf(G, S_A)  # Return selected super arm and its influence

def compute_mle(history, lambda_t, theta_init, max_iter=100, tol=1e-6, window_size=1000):
    """Compute Maximum Likelihood Estimation (MLE) using gradient descent with vectorized history processing."""
    if not history:  # If no history, return initial theta
        return theta_init.copy()

    dim_A = theta_init.shape[0]  # Dimension of theta
    theta = theta_init.copy()  # Initialize theta

    # Limit history to the last window_size entries for computational efficiency
    history = history[-window_size:] if len(history) > window_size else history

    # Vectorized processing of history
    G_history = np.array([g for g, _ in history])  # Feature vectors: shape (len(history), dim_A)
    X = np.array([x for _, x in history])  # Observed outcomes: shape (len(history),)

    for _ in range(max_iter):  # Iterate up to max_iter times
        mu = sigmoid(np.dot(G_history, theta))  # Predicted means
        grad = np.sum(G_history * (mu - X)[:, None], axis=0) + lambda_t * theta  # Gradient of negative log-likelihood
        hess = np.dot(G_history.T * mu * (1 - mu), G_history) + lambda_t * np.eye(dim_A)  # Hessian matrix
        try:
            step = np.linalg.solve(hess, grad)  # Compute Newton step
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad  # Use pseudo-inverse if Hessian is singular
        theta -= step  # Update theta
        if np.linalg.norm(step) < tol:  # Stop if step size is below tolerance
            break
    return theta

def compute_g_t(history, theta, lambda_t):
    """Compute g_t(theta) vectorized, used for MLE projection."""
    G_history = np.array([g for g, _ in history])  # Feature vectors: shape (len(history), dim_A)
    mu = sigmoid(np.dot(G_history, theta))  # Predicted means
    g = np.sum(mu[:, None] * G_history, axis=0) + lambda_t * theta  # Compute g_t as per paper's Eq. (5)
    return g

def compute_projected_mle(theta_hat, history, H_t_inv, Q_t_bound, lambda_t):
    """Compute projected MLE onto the bonus-vanishing region Q_t."""
    def objective(theta):
        g_theta = compute_g_t(history, theta, lambda_t)  # g_t(theta)
        g_theta_hat = compute_g_t(history, theta_hat, lambda_t)  # g_t(theta_hat)
        diff = g_theta - g_theta_hat  # Difference for projection
        return diff.T @ H_t_inv @ diff  # Objective: minimize norm in Hessian inverse space

    constraints = {'type': 'ineq', 'fun': lambda theta: Q_t_bound - np.linalg.norm(theta)}  # Constraint: theta within Q_t
    result = minimize(objective, theta_hat, constraints=constraints, method='SLSQP')  # Optimize using SLSQP
    return result.x if result.success else theta_hat  # Return projected theta or original on failure

def compute_Q_t_bound(history, sigma_t):
    """Compute the boundary of the bonus-vanishing region Q_t with vectorized processing."""
    if not history:  # If no history, return default bound L
        return L
    G_history = np.array([g for g, _ in history])  # Feature vectors: shape (len(history), dim_A)
    norms = np.sqrt(np.sum(G_history ** 2, axis=1))  # L2 norms of feature vectors
    bounds = sigma_t * norms  # Scale norms by confidence radius
    return max(bounds)  # Maximum bound defines Q_t

def estimate_kappa(L):
    """Estimate the nonlinearity parameter kappa."""
    return 4 * np.exp(L)  # Upper bound on kappa as per paper's Remark 2

# Parameter settings for the VA-CLogUCB algorithm
n_s = 500  # Number of base arms
n_t = 1  # Number of target nodes
k_A = 15  # Size of super arm
T = 2000  # Total rounds
dim_A = 10  # Feature dimension
noise_level = 0.01  # Noise level for observations
delta = 1 / T  # Confidence parameter
L = 1.0  # Bound on theta's L2 norm
kappa = estimate_kappa(L)  # Nonlinearity parameter

# Initialization
np.random.seed(3)  # Set random seed for reproducibility
G = np.random.uniform(-1, 1, (n_s, n_t, dim_A))  # Feature vectors: shape (n_s, n_t, dim_A)
w = np.random.rand(n_t)  # Weights for target nodes
pi = np.random.uniform(-1, 1, dim_A)  # True parameter theta*
G_pi = sigmoid(np.dot(G, pi))  # True arm means
G_w = G_pi * w  # Weighted means for reward computation
S_A_opt, opt = greedy(G_w, k_A)  # Optimal super arm and its reward

# VA-CLogUCB Algorithm
start = timeit.default_timer()  # Start timing
theta_hat = np.zeros(dim_A)  # Initialize estimated parameter
H_t = np.zeros((dim_A, dim_A))  # Hessian matrix
Gram = np.zeros((dim_A, dim_A))  # Gram matrix for covariance
history = []  # Store (feature, outcome) pairs
inf_t = np.zeros(T)  # Store rewards per round
regret = np.zeros(T)  # Store instantaneous regret
cumulative_regret = np.zeros(T)  # Store cumulative regret
theta_history = []  # Store theta estimates over time
Q_t_history = []  # Store Q_t bounds over time
Q_t_bound = L  # Initial Q_t bound

for t in range(T):
    lambda_t = dim_A * np.log(4 * (1 + t * k_A) / delta)  # Time-varying regularization
    sigma_t = (2 * L + 1) * (2 * L + 3) * np.sqrt(dim_A * np.log(4 * (1 + t * k_A) / delta))  # Confidence radius

    # Compute MLE
    theta_hat = compute_mle(history, lambda_t, theta_hat, window_size=1000)  # Estimate theta using MLE

    # Project MLE onto bonus-vanishing region Q_t if necessary
    Q_t_bound = compute_Q_t_bound(history, sigma_t)  # Update Q_t bound
    if np.linalg.norm(theta_hat) > Q_t_bound:  # Check if theta_hat is outside Q_t
        H_t += lambda_t * np.eye(dim_A)  # Add regularization to Hessian
        H_t_inv = np.linalg.inv(H_t)  # Compute inverse Hessian
        theta_hat = compute_projected_mle(theta_hat, history, H_t_inv, Q_t_bound, lambda_t)  # Project theta_hat

    # Compute Upper Confidence Bound (UCB)
    G_theta_hat = sigmoid(np.dot(G, theta_hat))  # Estimated means
    dot_ell = G_theta_hat * (1 - G_theta_hat)  # Variance term: ell'(x) = ell(x)(1-ell(x))
    H_t += lambda_t * np.eye(dim_A)  # Update Hessian with regularization
    Gram += kappa * lambda_t * np.eye(dim_A)  # Update Gram matrix
    H_t_inv = np.linalg.inv(H_t)  # Inverse Hessian
    V_t_inv = np.linalg.inv(Gram)  # Inverse covariance
    norms_H_t = np.sqrt(np.sum(G @ H_t_inv * G, axis=2))  # Norms for Hessian-based bonus
    norms_V_t = np.sqrt(np.sum(G @ V_t_inv * G, axis=2))  # Norms for covariance-based bonus
    CR = sigma_t * dot_ell * norms_H_t + (1 / 8) * kappa * sigma_t ** 2 * norms_V_t ** 2  # Exploration bonus
    UCB = np.clip(G_theta_hat + CR, 0, 1)  # UCB values
    G_t = UCB * w  # Weighted UCB for reward computation
    S_A, _ = greedy(G_t, k_A)  # Select super arm greedily

    # Observe outcomes and update
    for i in S_A:  # Process each arm in super arm
        noise = np.random.normal(0, noise_level, n_t)  # Add noise to true means
        noisy_G_pi = G_pi[i] + noise
        noisy_G_pi = np.clip(noisy_G_pi, 0, 1)  # Ensure means stay in [0, 1]
        X_i = (np.random.rand(n_t) < noisy_G_pi).astype(int)  # Generate Bernoulli outcomes
        for j in range(n_t):  # Process each target node
            history.append((G[i, j], X_i[j]))  # Store feature and outcome
            weight = dot_ell[i, j]  # Variance weight for Hessian
            H_t += weight * np.outer(G[i, j], G[i, j])  # Update Hessian
            Gram += np.outer(G[i, j], G[i, j])  # Update Gram matrix
            if X_i[j] == 1:  # Cascade model: Stop if reward is 1
                break
        if any(X_i == 1):  # Stop processing arms if any yields reward
            break
    H_t += lambda_t * np.eye(dim_A)  # Add regularization to Hessian
    Gram += kappa * lambda_t * np.eye(dim_A)  # Add regularization to Gram matrix

    inf_t[t] = cal_inf(G_w, S_A)  # Compute reward for this round
    regret[t] = opt - inf_t[t]  # Compute instantaneous regret
    cumulative_regret[t] = cumulative_regret[t - 1] + regret[t] if t > 0 else regret[t]  # Update cumulative regret
    theta_history.append(theta_hat.copy())  # Store current theta estimate
    Q_t_history.append(Q_t_bound)  # Store current Q_t bound

stop = timeit.default_timer()  # End timing
print(f"VA-CLogUCB Time: {stop - start:.2f} seconds")  # Print execution time
print(f"Cumulative Regret: {np.sum(regret):.2f}")  # Print total regret

# Plotting results
plt.figure(figsize=(10, 6))
plt.plot(np.arange(T), cumulative_regret, label='Cumulative Regret')
plt.xlabel('Time Step')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regret over Time for VA-CLogUCB')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(np.linalg.norm(theta_history - pi, axis=1))
plt.xlabel('Time Step')
plt.ylabel('Estimation Error ||theta_hat - pi||')
plt.title('Parameter Estimation Error over Time for VA-CLogUCB')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(Q_t_history)
plt.xlabel('Time Step')
plt.ylabel('Q_t Bound')
plt.title('Evolution of Q_t Bound for VA-CLogUCB')
plt.grid(True)
plt.show()