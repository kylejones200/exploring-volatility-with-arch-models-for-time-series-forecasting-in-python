use exploring_volatility_with_arch_models_for_time_series_forecasting_in_python_core::garch11_variance;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pyfunction]
fn garch11_variance_py<'py>(py: Python<'py>, returns: PyReadonlyArray1<f64>, omega: f64, alpha: f64, beta: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(garch11_variance(returns.as_slice()?, omega, alpha, beta).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (returns, omega, alpha, beta, iterations=500))]
fn bench_kernel_py(returns: PyReadonlyArray1<f64>, omega: f64, alpha: f64, beta: f64, iterations: usize) -> PyResult<f64> {
    let returns_buf = returns.as_slice()?.to_vec();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = garch11_variance(&returns_buf, omega, alpha, beta);
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn exploring_volatility_with_arch_models_for_time_series_forecasting_in_python_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(garch11_variance_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
