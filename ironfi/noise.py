import numpy as np

def sample_noise(rng: np.random.Generator, shape, dist: str = 'gaussian', df: float = 3.0):
    """Sample noise vector(s) with chosen distribution.
    Args:
      rng: np.random.Generator
      shape: tuple (n, m)
      dist: 'gaussian' | 'cauchy' | 'student-t'
      df: degrees of freedom for student-t
    Returns:
      ndarray of shape `shape`
    """
    if dist == 'gaussian':
        return rng.normal(size=shape)
    elif dist == 'cauchy':
        return rng.standard_cauchy(size=shape)
    elif dist == 'student-t':
        return rng.standard_t(df=df, size=shape)
    else:
        raise ValueError(f"Unknown dist: {dist}")

def sample_gaussian_xi(rng: np.random.Generator, sigma: float, alpha: float, tau: float, shape):
    """Center perturbation xi = (sqrt(alpha)/(1+tau)) * sigma * eta,  eta~N(0,I)."""
    eta = rng.normal(size=shape)
    scale = (np.sqrt(alpha) / (1.0 + tau)) * sigma
    return scale * eta
