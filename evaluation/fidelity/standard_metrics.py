"""
Standard Metrics for Synthetic Time Series Fidelity Evaluation
"""

import numpy as np
from scipy.stats import chi2
from statsmodels.tsa.stattools import acf
from scipy.stats import wasserstein_distance
from pytorch_fid.fid_score import calculate_frechet_distance
from tslearn.metrics import cdist_dtw

# Normalized Wasserstein Distance 
def normalized_wasserstein_distance(real, holdout, synthetic):
    """
    Computes the Normalized Wasserstein Distance with respect to the holdout dataset.

    Interpretation:
    - ~1: The synthetic data is statistically indistinguishable from unseen real patient data.
    - >1: The synthetic data has poor fidelity
    - <1: The synthetic data has memorization warning
    """
    return wasserstein_distance(real, synthetic) / wasserstein_distance(real, holdout)

# MMD
def rbf_kernel(x, y, bandwidths):
    """
    Computes a multi-bandwidth RBF kernel matrix between matrices x and y.
    """
    # Expand dimensions to compute pairwise distance matrix via broadcasting
    x_expanded = np.expand_dims(x, axis=1)  # Shape: [N, 1, D]
    y_expanded = np.expand_dims(y, axis=0)  # Shape: [1, M, D]
    
    # Squared Euclidean distances between all pairs
    distances = np.sum((x_expanded - y_expanded) ** 2, axis=2)
    
    # Calculate kernels for all bandwidths and sum them
    kernel_matrix = np.zeros_like(distances)
    for bw in bandwidths:
        kernel_matrix += np.exp(-distances / (2 * (bw ** 2)))
        
    return kernel_matrix

def mmd(real_features, synthetic_features, bandwidths=[0.1, 1.0, 10.0]):
    """
    Calculates the Maximum Mean Discrepancy (MMD) between two sets of features.
    
    Args:
        real_features: Array of shape [Batch_Size, Features]
        synthetic_features: Array of shape [Batch_Size, Features]
        bandwidths: List of sigma values for the RBF kernel
        
    Returns:
        mmd_score: A scalar representing the MMD distance
    """
    # Ensure inputs are NumPy arrays
    real_features = np.asarray(real_features)
    synthetic_features = np.asarray(synthetic_features)
    
    # Ensure inputs are 2D (Flatten the 6x22 clinical feature matrix if necessary)
    if real_features.ndim > 2:
        real_features = real_features.reshape(real_features.shape[0], -1)
    if synthetic_features.ndim > 2:
        synthetic_features = synthetic_features.reshape(synthetic_features.shape[0], -1)
        
    # Compute the 3 kernel components
    xx_kernel = rbf_kernel(real_features, real_features, bandwidths)
    yy_kernel = rbf_kernel(synthetic_features, synthetic_features, bandwidths)
    xy_kernel = rbf_kernel(real_features, synthetic_features, bandwidths)
    
    # Compute the MMD^2 statistic (biased estimator)
    mmd_squared = xx_kernel.mean() + yy_kernel.mean() - 2 * xy_kernel.mean()
    
    # Return the square root for the actual distance
    # Add a small epsilon before sqrt for numerical stability if mmd_squared approaches 0
    return np.sqrt(np.maximum(mmd_squared, 0) + 1e-8)

# Autocorrelation Score
def autocorrelation_score(real_data, synthetic_data, max_lag=24):
    """
    Computes the Autocorrelation Score (ACS) between real and synthetic time series.
    
    Parameters:
    - real_data: np.ndarray of shape (num_samples, seq_len, num_features)
    - synthetic_data: np.ndarray of shape (num_samples, seq_len, num_features)
    - max_lag: int, the maximum time lag to compute the autocorrelation for.
    
    Returns:
    - acs: float, the Mean Absolute Error between the real and synthetic ACF profiles.
           A lower score indicates better preservation of temporal dynamics.
    """
    # Ensure inputs have the same shape
    assert real_data.shape == synthetic_data.shape, "Data shapes must match."

    if real_data.ndim == 2:
        n, l = real_data.shape
        real_data = real_data.reshape(n, l, 1)
        synthetic_data = synthetic_data.reshape(n, l, 1)

    N, T, F = real_data.shape
    
    # Cap max_lag if the sequence length is shorter than the requested lag
    max_lag = min(max_lag, T - 1)
    
    real_acf_profiles = np.zeros((F, max_lag + 1))
    synth_acf_profiles = np.zeros((F, max_lag + 1))
    
    # Iterate through each feature (channel)
    for f in range(F):
        
        # 1. Compute ACF for every sample in the real dataset and average them
        real_acfs = [
            acf(real_data[i, :, f], nlags=max_lag, fft=True) 
            for i in range(N)
        ]
        real_acf_profiles[f] = np.mean(real_acfs, axis=0)
        
        # 2. Compute ACF for every sample in the synthetic dataset and average them
        synth_acfs = [
            acf(synthetic_data[i, :, f], nlags=max_lag, fft=True) 
            for i in range(N)
        ]
        synth_acf_profiles[f] = np.mean(synth_acfs, axis=0)
        
    # 3. Compute the Mean Absolute Error (MAE) across all features and lags
    # We exclude lag 0 because autocorrelation at lag 0 is always exactly 1.0
    acs_score = np.mean(np.abs(real_acf_profiles[:, 1:] - synth_acf_profiles[:, 1:]))
    
    return acs_score

# Centroid Mahalanobis Distance
def _as_2d_samples(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    if X.ndim > 2:
        return X.reshape(X.shape[0], -1)
    return X

def centroid_mahalanobis_score(real, synthetic):
    """
    Normalized Mahalanobis distance [0, 1] from synthetic samples to the real distribution centroid.
    A score near 1.0 indicates perfect alignment with the real center; < 0.05 indicates severe outliers.
    """
    real = _as_2d_samples(real)
    synth = _as_2d_samples(synthetic)

    real_mean = np.mean(real, axis=0, keepdims=True)
    cov_matrix = np.atleast_2d(np.cov(real, rowvar=False))
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    cutoff = 1e-12 * (eigenvalues[-1] if eigenvalues[-1] > 0 else 1.0)
    supported = eigenvalues > cutoff
    
    if not np.any(supported):
        return 0.0
        
    df = np.sum(supported) # Degrees of freedom for Chi-Square
    scale = 1.0 / np.sqrt(eigenvalues[supported])
    whitener = eigenvectors[:, supported] * scale

    # Calculate distance to the real centroid in whitened space
    synth_whitened = synth @ whitener
    centroid_whitened = real_mean @ whitener
    
    distances = np.linalg.norm(synth_whitened - centroid_whitened, axis=1)
    squared_distances = distances ** 2
    
    # Convert to p-values via Chi-Square CDF
    p_values = 1.0 - chi2.cdf(squared_distances, df=df)
    
    return float(np.mean(p_values))


def frechet_inception_distance(real, synthetic, eps=1e-6):
    """
    Fréchet Inception Distance between the Gaussian fits of two sample sets.

    Each sample is flattened to a feature vector, so inputs may be (N, D),
    (N, T), or (N, T, F). Lower is better (0 when the two Gaussians match).
    """
    real = _as_2d_samples(real)
    synthetic = _as_2d_samples(synthetic)
    mu_real, mu_synth = np.mean(real, axis=0), np.mean(synthetic, axis=0)
    sigma_real = np.atleast_2d(np.cov(real, rowvar=False))
    sigma_synth = np.atleast_2d(np.cov(synthetic, rowvar=False))
    return float(calculate_frechet_distance(mu_real, sigma_real, mu_synth, sigma_synth, eps=eps))


def _as_tslearn_series(X):
    """tslearn wants (n_ts, sz) or (n_ts, sz, d)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        return X.reshape(1, -1)
    if X.ndim == 2:
        return X
    if X.ndim == 3:
        return X
    raise ValueError(f"Expected series of shape (T,), (N, T), or (N, T, F); got {X.shape}.")


def dtw_distance(real, synthetic):
    """
    Mean nearest-neighbor DTW from each synthetic series to the real set.

    Univariate inputs are (N, T); multivariate inputs are (N, T, F) and use
    dependent DTW (channels warped together). Lower is better.
    """
    real = _as_tslearn_series(real)
    synthetic = _as_tslearn_series(synthetic)
    distances = cdist_dtw(synthetic, real)
    return float(np.mean(np.min(distances, axis=1)))

### Evaluation Call
def evaluate_standard_metrics(real, synthetic, holdout):
    """
    Evaluates the standard metrics for synthetic time series fidelity.
    """
    return {
        'normalized_wasserstein_distance': normalized_wasserstein_distance(real, holdout, synthetic),
        'mmd': mmd(real, synthetic),
        'autocorrelation_score': autocorrelation_score(real, synthetic),
        'centroid_mahalanobis_distance': centroid_mahalanobis_score(real, synthetic),
        'frechet_inception_distance': frechet_inception_distance(real, synthetic),
        'dtw_distance': dtw_distance(real, synthetic),
    }