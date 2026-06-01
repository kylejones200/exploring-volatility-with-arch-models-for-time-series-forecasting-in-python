"""GARCH(1,1) conditional variance recursion."""

from __future__ import annotations

import numpy as np


def garch11_variance(
    returns: np.ndarray, omega: float, alpha: float, beta: float
) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n == 0:
        return np.empty(0, dtype=float)
    h = np.zeros(n, dtype=float)
    h[0] = max(float(r.var()), 1e-12)
    for t in range(1, n):
        h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
    return h
