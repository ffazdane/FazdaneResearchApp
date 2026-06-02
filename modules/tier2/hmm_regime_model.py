import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger("HmmRegimeModel")

class GaussianHMM:
    """
    A robust, self-contained Gaussian Hidden Markov Model with diagonal covariance 
    for stable parameter estimation on financial time-series. Fully vectorized.
    """
    def __init__(self, n_states=3, max_iter=20, tol=1e-4):
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        
        # Parameters
        self.pi = None      # Initial state distribution (N,)
        self.A = None       # Transition probabilities matrix (N, N)
        self.means = None   # Emission means (N, D)
        self.vars = None    # Emission variances (N, D) (diagonal covariance)
        
        self.n_features = None
        self.is_fitted = False
        
    def _compute_emissions(self, X):
        """Calculate Gaussian pdf (diagonal covariance) for all data points in X across all states."""
        # X: (T, D), means: (N, D), vars: (N, D) -> diff: (T, N, D)
        diff = X[:, np.newaxis, :] - self.means[np.newaxis, :, :]
        exponent = -0.5 * (diff ** 2) / self.vars[np.newaxis, :, :]
        denom = np.sqrt(2.0 * np.pi * self.vars[np.newaxis, :, :])
        probs = np.exp(exponent) / denom
        
        # Product along features axis (axis 2)
        emissions = np.prod(probs, axis=2)
        return np.maximum(emissions, 1e-100)
        
    def _emission_density(self, x):
        """Calculate Gaussian pdf (diagonal covariance) for data point x across all states."""
        # x: (D,) -> diff: (N, D)
        diff = x - self.means
        exponent = -0.5 * (diff ** 2) / self.vars
        denom = np.sqrt(2.0 * np.pi * self.vars)
        probs = np.exp(exponent) / denom
        pdf = np.prod(probs, axis=1)
        return np.maximum(pdf, 1e-100)
        
    def fit(self, X):
        """
        Fit the HMM parameters using a robust K-Means initialization 
        followed by Baum-Welch (EM) iterations.
        """
        X = np.asarray(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        n_samples, self.n_features = X.shape
        if n_samples < self.n_states * 2:
            raise ValueError("Too few samples to fit HMM")
            
        # 1. Initialize using K-Means to ensure deterministic state assignment
        kmeans = KMeans(n_clusters=self.n_states, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X)
        
        self.means = np.zeros((self.n_states, self.n_features))
        self.vars = np.zeros((self.n_states, self.n_features))
        
        for i in range(self.n_states):
            state_data = X[labels == i]
            if len(state_data) > 1:
                self.means[i] = np.mean(state_data, axis=0)
                self.vars[i] = np.var(state_data, axis=0) + 1e-6  # variance floor
            else:
                self.means[i] = np.mean(X, axis=0)
                self.vars[i] = np.var(X, axis=0) + 1e-6
                
        # Initialize Transition matrix A with Laplace smoothing
        self.A = np.ones((self.n_states, self.n_states)) * 0.1
        for t in range(n_samples - 1):
            self.A[labels[t], labels[t+1]] += 1.0
        self.A /= np.sum(self.A, axis=1, keepdims=True)
        
        # Initialize initial state distribution pi
        self.pi = np.ones(self.n_states) * 0.1
        state_counts = np.bincount(labels, minlength=self.n_states)
        self.pi += state_counts
        self.pi /= np.sum(self.pi)
        
        # 2. Baum-Welch (EM) training
        old_log_likelihood = -np.inf
        
        for iteration in range(self.max_iter):
            # E-step: forward-backward
            emissions = self._compute_emissions(X)
            alpha, log_prob_alpha = self._forward(X, emissions=emissions)
            beta = self._backward(X, emissions=emissions)
            
            # Compute gammas (posterior probabilities of states)
            gamma = alpha * beta
            gamma_sum = np.sum(gamma, axis=1, keepdims=True)
            gamma_sum[gamma_sum == 0] = 1e-20
            gamma /= gamma_sum
            
            # Vectorized xi calculation (transitions)
            # Shape: (T-1, N, N)
            emit_beta = emissions[1:] * beta[1:]
            xi = (alpha[:-1, :, np.newaxis] * 
                  self.A[np.newaxis, :, :] * 
                  emit_beta[:, np.newaxis, :])
            xi_sum = np.sum(xi, axis=(1, 2), keepdims=True)
            xi = np.where(xi_sum > 0, xi / xi_sum, 0.0)
                    
            # M-step: update parameters
            self.pi = gamma[0] / np.sum(gamma[0])
            
            # Update transitions
            xi_sum_over_t = np.sum(xi, axis=0)
            gamma_sum_t_minus_1 = np.sum(gamma[:-1], axis=0, keepdims=True).T
            gamma_sum_t_minus_1[gamma_sum_t_minus_1 == 0] = 1e-20
            self.A = xi_sum_over_t / gamma_sum_t_minus_1
            self.A /= np.sum(self.A, axis=1, keepdims=True)
            
            # Update emissions
            gamma_sum_over_t = np.sum(gamma, axis=0, keepdims=True).T
            gamma_sum_over_t[gamma_sum_over_t == 0] = 1e-20
            
            # Update means
            self.means = np.dot(gamma.T, X) / gamma_sum_over_t
            
            # Update variances
            for i in range(self.n_states):
                diff = X - self.means[i]
                self.vars[i] = np.dot(gamma[:, i], diff ** 2) / gamma_sum_over_t[i, 0] + 1e-6
                
            # Check convergence
            log_likelihood = log_prob_alpha
            if np.abs(log_likelihood - old_log_likelihood) < self.tol:
                break
            old_log_likelihood = log_likelihood
            
        self.is_fitted = True
        
    def _forward(self, X, emissions=None):
        """Execute forward pass scaling alpha to prevent underflow."""
        n_samples = X.shape[0]
        alpha = np.zeros((n_samples, self.n_states))
        
        if emissions is None:
            emissions = self._compute_emissions(X)
            
        # t = 0
        alpha[0] = self.pi * emissions[0]
        sum_alpha = np.sum(alpha[0])
        if sum_alpha > 0:
            alpha[0] /= sum_alpha
            
        log_prob = np.log(max(sum_alpha, 1e-20))
        
        # t > 0
        for t in range(1, n_samples):
            alpha[t] = np.dot(alpha[t-1], self.A) * emissions[t]
            sum_alpha = np.sum(alpha[t])
            if sum_alpha > 0:
                alpha[t] /= sum_alpha
            log_prob += np.log(max(sum_alpha, 1e-20))
            
        return alpha, log_prob
        
    def _backward(self, X, emissions=None):
        """Execute backward pass scaling beta."""
        n_samples = X.shape[0]
        beta = np.zeros((n_samples, self.n_states))
        
        if emissions is None:
            emissions = self._compute_emissions(X)
            
        # t = T
        beta[-1] = 1.0
        
        # t < T
        for t in range(n_samples - 2, -1, -1):
            beta[t] = np.dot(self.A, emissions[t+1] * beta[t+1])
            sum_beta = np.sum(beta[t])
            if sum_beta > 0:
                beta[t] /= sum_beta
                
        return beta
        
    def predict_proba(self, X):
        """Compute state probabilities for each time step."""
        X = np.asarray(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        emissions = self._compute_emissions(X)
        alpha, _ = self._forward(X, emissions=emissions)
        beta = self._backward(X, emissions=emissions)
        gamma = alpha * beta
        gamma_sum = np.sum(gamma, axis=1, keepdims=True)
        gamma_sum[gamma_sum == 0] = 1e-20
        gamma /= gamma_sum
        return gamma
        
    def predict(self, X):
        """Decode the most likely state sequence using the Viterbi algorithm."""
        X = np.asarray(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        n_samples = X.shape[0]
        viterbi = np.zeros((n_samples, self.n_states))
        backpointer = np.zeros((n_samples, self.n_states), dtype=int)
        
        emissions = self._compute_emissions(X)
        
        # t = 0
        viterbi[0] = np.log(self.pi + 1e-20) + np.log(emissions[0] + 1e-20)
        
        # t > 0
        for t in range(1, n_samples):
            log_trans = viterbi[t-1][:, np.newaxis] + np.log(self.A + 1e-20)
            backpointer[t] = np.argmax(log_trans, axis=0)
            viterbi[t] = log_trans[backpointer[t], np.arange(self.n_states)] + np.log(emissions[t] + 1e-20)
            
        # Find best final state
        best_path = np.zeros(n_samples, dtype=int)
        best_path[-1] = np.argmax(viterbi[-1])
        
        # Backtrack
        for t in range(n_samples - 2, -1, -1):
            best_path[t] = backpointer[t+1, best_path[t+1]]
            
        return best_path


def train_hmm_model(df: pd.DataFrame, n_states: int = 3) -> tuple:
    """
    Train a Gaussian HMM model on returns and realized volatility features, 
    and dynamically assign human-readable labels to HMM states based on returns.
    """
    df = df.copy().dropna(subset=["daily_return", "realized_vol_20d"])
    features = df[["daily_return", "realized_vol_20d"]].values
    
    hmm = GaussianHMM(n_states=n_states)
    hmm.fit(features)
    
    # Decoded states sequence
    states = hmm.predict(features)
    probabilities = hmm.predict_proba(features)
    
    # Label states: map index -> name based on cluster emission properties
    # Let's sort states by return (Bull has high return, Bear has low return)
    # or realized vol (Sideways has low return & low vol)
    state_labels = {}
    
    # Gather average return and vol for each state index
    state_stats = []
    for i in range(n_states):
        avg_ret = hmm.means[i, 0]
        avg_vol = hmm.means[i, 1]
        state_stats.append((i, avg_ret, avg_vol))
        
    # Sort states by average return
    state_stats_sorted = sorted(state_stats, key=lambda x: x[1])
    
    if n_states == 3:
        # Lowest return = BEAR
        state_labels[state_stats_sorted[0][0]] = "BEAR"
        # Middle return = SIDEWAYS
        state_labels[state_stats_sorted[1][0]] = "SIDEWAYS"
        # Highest return = BULL
        state_labels[state_stats_sorted[2][0]] = "BULL"
    else:
        # Default numeric labeling for higher states
        for rank, (state_idx, _, _) in enumerate(state_stats_sorted):
            state_labels[state_idx] = f"STATE_{rank}"
            
    # Generate list of text states
    decoded_states = [state_labels[s] for s in states]
    
    # Map state probabilities
    prob_dict = {state_labels[i]: probabilities[:, i] for i in range(n_states)}
    
    # Store results back
    res_df = df.copy()
    res_df["hmm_state"] = decoded_states
    for state_name, prob in prob_dict.items():
        res_df[f"hmm_prob_{state_name.lower()}"] = prob
        
    return res_df, hmm, state_labels
