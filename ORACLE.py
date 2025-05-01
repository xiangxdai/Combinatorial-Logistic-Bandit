import numpy as np


def cal_inf(G, S_A):
    """
    Calculate the expected reward (influence) of super arm S_A.

    Parameters:
    -----------
    G : numpy.ndarray
        Matrix of shape (n_arms, n_targets) where G[i, j] is the probability that arm i covers target j.
    S_A : list
        List of indices representing the selected super arm.

    Returns:
    --------
    float
        Expected reward (influence) of the super arm S_A.
    """
    return np.sum(1 - np.prod(1 - G[S_A], axis=0))


def greedy(G, k_A):
    """
    Greedily select k_A arms to maximize the expected reward in the Combinatorial Logistic Bandit problem.

    Parameters:
    -----------
    G : numpy.ndarray
        Matrix of shape (n_arms, n_targets) where G[i, j] is the probability that arm i covers target j.
    k_A : int
        Number of arms to select for the super arm.

    Returns:
    --------
    tuple
        - list: Indices of the selected super arm (S_A).
        - float: Expected reward (influence) of the selected super arm.

    Raises:
    -------
    ValueError
        If G is not a 2D numpy array, k_A is negative, or k_A exceeds the number of arms.
    """
    # Input validation
    if not isinstance(G, np.ndarray) or G.ndim != 2:
        raise ValueError("G must be a 2D numpy array")
    if not isinstance(k_A, int) or k_A < 0:
        raise ValueError("k_A must be a non-negative integer")
    n_s = G.shape[0]  # Number of base arms
    if k_A > n_s:
        raise ValueError(f"k_A ({k_A}) cannot exceed the number of arms ({n_s})")

    S_A = []  # Initialize empty super arm
    available_arms = np.arange(n_s)  # Track available arms

    for _ in range(k_A):
        inf_out = np.zeros(len(available_arms))  # Store influence for candidate arms
        for idx, arm in enumerate(available_arms):
            # Compute influence if arm is added to current S_A
            inf_out[idx] = cal_inf(G, S_A + [arm])

        # Select arm with maximum influence
        best_arm_idx = np.argmax(inf_out)
        best_arm = available_arms[best_arm_idx]
        S_A.append(best_arm)

        # Remove selected arm from available arms
        available_arms = np.delete(available_arms, best_arm_idx)

    # Compute final influence
    influence = cal_inf(G, S_A)
    return S_A, influence