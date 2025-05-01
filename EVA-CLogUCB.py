import numpy as np
import matplotlib.pyplot as plt
import timeit
from scipy.special import expit as sigmoid
from scipy.optimize import minimize

# Utility functions for the Combinatorial Logistic Bandit (CLogB) problem
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

def compute_mle(history, theta_init, lambda_t, Q_bound, V_T0, theta_T0, window_size=1000):
    """Compute Maximum Likelihood Estimation (MLE) within the nonlinearity-restricted region Q."""
    if not history:  # If no history, return initial theta
        return theta_init.copy()

    # Preprocess history data, limiting to the last window_size entries for efficiency
    history = history[-window_size:] if len(history) > window_size else history
    G_history = np.array([g for g, _ in history])  # Feature vectors: shape (len(history), dim_A)
    X = np.array([x for _, x in history])  # Observed outcomes: shape (len(history),)

    def objective(theta):
        # Compute negative log-likelihood (cross-entropy loss) vectorized
        mu = sigmoid(np.dot(G_history, theta))  # Predicted means: shape (len(history),)
        ll = np.sum(X * np.log(mu + 1e-10) + (1 - X) * np.log(1 - mu + 1e-10))  # Log-likelihood
        reg = (lambda_t / 2) * np.sum(theta ** 2)  # L2 regularization term
        return -ll + reg  # Minimize negative log-likelihood plus regularization

    # Define constraint for the nonlinearity-restricted region Q
    def constraint(theta):
        diff = theta - theta_T0  # Difference from burn-in stage MLE
        return Q_bound - np.sqrt(np.dot(diff, V_T0 @ diff))  # Ensure theta lies within Q

    constraints = {'type': 'ineq', 'fun': constraint}  # Inequality constraint for optimization
    result = minimize(objective, theta_init, constraints=constraints, method='SLSQP',
                      options={'maxiter': 100, 'ftol': 1e-6})  # Optimize using SLSQP
    return result.x if result.success else theta_init  # Return optimized theta or initial theta on failure

def compute_Q_bound(T_0, kappa, L, dim_A, T, delta):
    """Compute the boundary of the nonlinearity-restricted region Q after the burn-in stage."""
    lambda_0 = dim_A * np.log(4 * (2 + T_0) / delta)  # Regularization parameter for burn-in stage
    confidence_radius = (L**2 + 4*L + 19/4) * np.sqrt(kappa * dim_A * np.log(4 * (2 + T) / delta))  # Q region radius
    return confidence_radius, lambda_0

def estimate_kappa(L):
    """Estimate the nonlinearity parameter kappa based on the bound of theta."""
    return 4 * np.exp(L)  # Upper bound on kappa as per paper's Remark 2

# Parameter settings for the EVA-CLogUCB algorithm
n_s = 500  # Number of base arms
n_t = 1  # Number of target nodes
k_A = 10  # Size of super arm
T = 2000  # Total rounds
dim_A = 10  # Feature dimension
noise_level = 0.01  # Noise level for observations
delta = 1 / T  # Confidence parameter
L = 1.0  # Bound on theta's L2 norm
kappa = estimate_kappa(L)  # Nonlinearity parameter
T_0 = min(int((4 * L**2 + 16 * L + 19)**2 * kappa * dim_A**2 * np.log(4 * (2 + T) / delta)), int(0.1 * T))  # Burn-in stage length

# Initialization
np.random.seed(1)  # Set random seed for reproducibility
G = np.random.uniform(-1, 1, (n_s, n_t, dim_A))  # Feature vectors: shape (n_s, n_t, dim_A)
w = np.random.rand(n_t)  # Weights for target nodes
pi = np.random.uniform(-1, 1, dim_A)  # True parameter theta*
G_pi = sigmoid(np.dot(G, pi))  # True arm means
G_w = G_pi * w  # Weighted means for reward computation
S_A_opt, opt = greedy(G_w, k_A)  # Optimal super arm and its reward

# EVA-CLogUCB Algorithm
start = timeit.default_timer()  # Start timing
theta_hat = np.zeros(dim_A)  # Initialize estimated parameter
H_t = np.zeros((dim_A, dim_A))  # Hessian matrix
Gram = np.zeros((dim_A, dim_A))  # Gram matrix for covariance
history = []  # Store (feature, outcome) pairs
inf_t = np.zeros(T)  # Store rewards per round
regret = np.zeros(T)  # Store instantaneous regret
cumulative_regret = np.zeros(T)  # Store cumulative regret
theta_history = []  # Store theta estimates over time
Q_bound = None  # Q region boundary
V_T0 = None  # Covariance matrix after burn-in
theta_T0 = None  # MLE after burn-in
lambda_0 = dim_A * np.log(4 * (2 + T_0) / delta)  # Burn-in regularization parameter

for t in range(T):
    lambda_t = dim_A * np.log(4 * (1 + t * k_A) / delta)  # Time-varying regularization
    v_t = 3 * (L + 1.5) * np.sqrt(dim_A * np.log(4 * (1 + t) / delta))  # Confidence radius for exploration bonus

    if t < T_0:
        # Burn-in stage: Select arm with maximum uncertainty
        V_t = Gram + kappa * lambda_0 * np.eye(dim_A)  # Covariance matrix
        V_t_inv = np.linalg.inv(V_t)  # Inverse covariance
        norms = np.sqrt(np.sum(G @ V_t_inv * G, axis=2))  # Norms for uncertainty
        i_t = np.argmax(norms.max(axis=1))  # Select arm with highest uncertainty
        available_arms = list(range(n_s))  # All available arms
        S_A = [i_t]  # Initialize super arm with most uncertain arm
        available_arms.remove(i_t)  # Remove selected arm
        if len(available_arms) > 0 and k_A > 1:  # Add additional arms randomly
            num_additional_arms = min(k_A - 1, len(available_arms))
            additional_arms = np.random.choice(available_arms, size=num_additional_arms, replace=False)
            S_A.extend(additional_arms)

        for i in S_A:  # Process each arm in super arm
            noise = np.random.normal(0, noise_level, n_t)  # Add noise to true means
            noisy_G_pi = G_pi[i] + noise
            noisy_G_pi = np.clip(noisy_G_pi, 0, 1)  # Ensure means stay in [0, 1]
            X_i = (np.random.rand(n_t) < noisy_G_pi).astype(int)  # Generate Bernoulli outcomes
            for j in range(n_t):  # Process each target node
                history.append((G[i, j], X_i[j]))  # Store feature and outcome
                Gram += np.outer(G[i, j], G[i, j])  # Update Gram matrix
                H_t += np.outer(G[i, j], G[i, j]) * sigmoid(theta_hat @ G[i, j]) * (1 - sigmoid(theta_hat @ G[i, j]))  # Update Hessian
                if X_i[j] == 1:  # Cascade model: Stop if reward is 1
                    break
            if any(X_i == 1):  # Stop processing arms if any yields reward
                break
        inf_t[t] = cal_inf(G_w, S_A)  # Compute reward for this round
    else:
        if t == T_0:
            # End of burn-in: Compute Q region and burn-in MLE
            Q_bound, lambda_0 = compute_Q_bound(T_0, kappa, L, dim_A, T, delta)  # Q region parameters
            V_T0 = Gram + kappa * lambda_0 * np.eye(dim_A)  # Covariance after burn-in
            theta_T0 = compute_mle(history, theta_hat, lambda_0, Q_bound, V_T0, theta_hat, window_size=1000)  # Burn-in MLE

        # Learning stage: Compute MLE within Q
        theta_hat = compute_mle(history, theta_hat, lambda_t, Q_bound, V_T0, theta_T0, window_size=1000)

        # Compute Upper Confidence Bound (UCB)
        G_theta_hat = sigmoid(np.dot(G, theta_hat))  # Estimated means
        dot_ell = G_theta_hat * (1 - G_theta_hat)  # Variance term: ell'(x) = ell(x)(1-ell(x))
        H_t += lambda_t * np.eye(dim_A)  # Update Hessian with regularization
        Gram += kappa * lambda_t * np.eye(dim_A)  # Update Gram matrix
        H_t_inv = np.linalg.inv(H_t)  # Inverse Hessian
        V_t_inv = np.linalg.inv(Gram)  # Inverse covariance
        norms_H_t = np.sqrt(np.sum(G @ H_t_inv * G, axis=2))  # Norms for Hessian-based bonus
        norms_V_t = np.sqrt(np.sum(G @ V_t_inv * G, axis=2))  # Norms for covariance-based bonus
        CR = np.sqrt(np.e) * dot_ell * v_t * norms_H_t + (1 / 8) * kappa * v_t**2 * norms_V_t**2  # Exploration bonus
        UCB = np.clip(G_theta_hat + CR, 0, 1)  # UCB values
        G_t = UCB * w  # Weighted UCB for reward computation
        S_A, _ = greedy(G_t, k_A)  # Select super arm greedily

        # Observe outcomes and update
        for i in S_A:
            noise = np.random.normal(0, noise_level, n_t)
            noisy_G_pi = G_pi[i] + noise
            noisy_G_pi = np.clip(noisy_G_pi, 0, 1)
            X_i = (np.random.rand(n_t) < noisy_G_pi).astype(int)
            for j in range(n_t):
                history.append((G[i, j], X_i[j]))
                weight = dot_ell[i, j]  # Variance weight for Hessian
                H_t += weight * np.outer(G[i, j], G[i, j])
                Gram += np.outer(G[i, j], G[i, j])
                if X_i[j] == 1:  # Cascade model: Stop if reward is 1
                    break
            if any(X_i == 1):
                break
        inf_t[t] = cal_inf(G_w, S_A)

    regret[t] = opt - inf_t[t]  # Compute instantaneous regret
    cumulative_regret[t] = cumulative_regret[t - 1] + regret[t] if t > 0 else regret[t]  # Update cumulative regret
    theta_history.append(theta_hat.copy())  # Store current theta estimate

stop = timeit.default_timer()  # End timing
print(f"EVA-CLogUCB Time: {stop - start:.2f} seconds")  # Print execution time
print(f"Cumulative Regret: {np.sum(regret):.2f}")  # Print total regret

# Plotting results
plt.figure(figsize=(10, 6))
plt.plot(np.arange(T), cumulative_regret, label='Cumulative Regret')
plt.xlabel('Time Step')
plt.ylabel('Cumulative Regret')
plt.title('Cumulative Regret over Time for EVA-CLogUCB')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(np.linalg.norm(theta_history - pi, axis=1))
plt.xlabel('Time Step')
plt.ylabel('Estimation Error ||theta_hat - pi||')
plt.title('Parameter Estimation Error over Time for EVA-CLogUCB')
plt.grid(True)
plt.show()