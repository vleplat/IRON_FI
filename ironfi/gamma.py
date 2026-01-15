def update_gamma(gamma: float, alpha: float, mu: float) -> float:
    """One-step update for gamma: gamma_{k+1} = (gamma_k + alpha * mu) / (1 + alpha)."""
    return (gamma + alpha * mu) / (1.0 + alpha)
