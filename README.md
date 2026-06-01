# Exploring Volatility with ARCH Models for Time Series Forecasting

This project demonstrates ARCH and GARCH models for volatility forecasting.

## Business context

Sometimes in time series work we care about predicting specific value. Other times we just care about the value changing.

Sometimes in time series work we care about predicting specific value. Other times we just care about the value changing.

Autoregressive Conditional Heteroskedasticity (ARCH) models are popular in economics and finance for modeling and forecasting time-varying volatility. ARCH models don't assume constant variance (like most TS models) which is why they are often used for analyzing volatility clustering, such as stock prices or returns.

## Article

Medium article: [Exploring Volatility with ARCH Models for Time Series Forecasting in Python](https://medium.com/@kylejones_47003/exploring-volatility-with-arch-models-for-time-series-forecasting-in-python-53966b72c1ce)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # ARCH/GARCH functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
├── images/            # Generated plots and figures
├── rust/                   # Rust port (core + PyO3 + CLI bench)
├── benchmark_rust.py       # Python vs Rust benchmark
├── src/compute_kernel.py   # Python/numpy reference kernel
```

## Configuration

Edit `config.yaml` to customize:
- Data source or synthetic generation
- GARCH parameters (p, q, omega, alpha, beta)
- Forecast horizon
- Output settings

## ARCH/GARCH Models

### ARCH (Autoregressive Conditional Heteroskedasticity)
- Models volatility clustering
- Volatility depends on past squared errors

### GARCH (Generalized ARCH)
- Extends ARCH with lagged volatility
- More flexible and commonly used
- Captures persistence in volatility

## Caveats

- By default, generates synthetic returns with volatility clustering.
- GARCH models assume volatility follows specific dynamics.
- Model selection (p, q) requires careful consideration.

## Rust performance port

Side-by-side **Python vs Rust** implementation of the numeric hot loop — GARCH(1,1) conditional variance. Reference PyO3 benchmark: **see `benchmark_rust.py`** on a release build (local machine; run `benchmark_rust.py` to reproduce).

| Path | Role |
|------|------|
| `src/compute_kernel.py` | Python/numpy reference kernel |
| `rust/core/` | Pure Rust library |
| `rust/py/` | PyO3 bindings |
| `rust/bench/` | Standalone CLI benchmark |
| `benchmark_rust.py` | Python vs Rust timing + correctness check |

```bash
# Rust-only CLI benchmark
cd rust && cargo run --release -p exploring_volatility_with_arch_models_for_time_series_forecasting_in_python_bench

# Python vs Rust (PyO3)
pip install maturin numpy
maturin develop --release -m rust/py/Cargo.toml
python benchmark_rust.py
```

Python ML training, solvers, and orchestration stay in Python; Rust targets the numeric hot loops. Stochastic generators validate output shapes; deterministic kernels match at tight floating-point tolerance.


## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).