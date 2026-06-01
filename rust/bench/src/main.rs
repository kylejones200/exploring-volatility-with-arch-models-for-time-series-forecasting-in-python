use exploring_volatility_with_arch_models_for_time_series_forecasting_in_python_core::garch11_variance;

fn main() {
    let r: Vec<f64> = (0..2000).map(|i| (i as f64 * 0.001).sin() * 0.01).collect();
    for _ in 0..500 {
        let _ = garch11_variance(&r, 1e-6, 0.1, 0.85);
    }
}
