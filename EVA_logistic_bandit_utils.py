import numpy as np
from scipy.special import expit as sigmoid
from scipy.optimize import minimize

def compute_mle(history, theta_init, lambda_t, Q_bound, V_T0, theta_T0, window_size=1000):
    """
    Compute Maximum Likelihood Estimation (MLE) within the nonlinearity-restricted region Q.

    Args:
        history (list): List of tuples (g, x), where g is the feature vector and x is the observed outcome.
        theta_init (np.ndarray): Initial parameter vector for optimization.
        lambda_t (float): Regularization parameter for L2 penalty.
        Q_bound (float): Radius of the nonlinearity-restricted region Q.
        V_T0 (np.ndarray): Matrix defining the Q region (typically from burn-in stage).
        theta_T0 (np.ndarray): MLE parameter from the burn-in stage.
        window_size (int, optional): Maximum number of history entries to consider. Defaults to 1000.

    Returns:
        np.ndarray: Optimized parameter vector theta, or theta_init if optimization fails.
    """
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
    """
    Compute the boundary of the nonlinearity-restricted region Q after the burn-in stage.

    Args:
        T_0 (int): Length of the burn-in stage.
        kappa (float): Nonlinearity parameter for the logistic function.
        L (float): Bound on the norm of the parameter theta.
        dim_A (int): Dimension of the feature vector.
        T (int): Total number of rounds.
        delta (float): Confidence parameter for the bound.

    Returns:
        tuple: (confidence_radius, lambda_0)
            - confidence_radius (float): Radius of the Q region.
            - lambda_0 (float): Regularization parameter for the burn-in stage.
    """
    lambda_0 = dim_A * np.log(4 * (2 + T_0) / delta)  # Regularization parameter for burn-in stage
    confidence_radius = (L**2 + 4*L + 19/4) * np.sqrt(kappa * dim_A * np.log(4 * (2 + T) / delta))  # Q region radius
    return confidence_radius, lambda_0