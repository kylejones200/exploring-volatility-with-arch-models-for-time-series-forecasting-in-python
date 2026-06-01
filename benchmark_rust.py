#!/usr/bin/env python3
"""Python vs Rust kernel benchmark."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from compute_kernel import garch11_variance  # noqa: E402

def main() -> None:
    r = np.ascontiguousarray(np.sin(np.arange(2000) * 0.001) * 0.01)
    o, a, b = 1e-6, 0.1, 0.85
    t0 = time.perf_counter()
    for _ in range(200):
        garch11_variance(r, o, a, b)
    py_s = time.perf_counter() - t0
    try:
        import exploring_volatility_with_arch_models_for_time_series_forecasting_in_python_rs as rs
    except ImportError:
        print("Build: maturin develop --release -m rust/py/Cargo.toml")
        print(f"Python {py_s:.3f}s")
        return
    rs_s = rs.bench_kernel_py(r, o, a, b, 500)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s / max(rs_s, 1e-9):.1f}x")
    np.testing.assert_allclose(
        garch11_variance(r, o, a, b),
        np.asarray(rs.garch11_variance_py(r, o, a, b)),
        rtol=1e-10,
    )
    print("Correctness: OK")

if __name__ == "__main__":
    main()
