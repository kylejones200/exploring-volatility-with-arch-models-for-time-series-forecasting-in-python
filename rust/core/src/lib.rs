//! GARCH(1,1) conditional variance recursion.

pub fn garch11_variance(returns: &[f64], omega: f64, alpha: f64, beta: f64) -> Vec<f64> {
    let n = returns.len();
    if n == 0 {
        return vec![];
    }
    let mut h = vec![0.0; n];
    let init = returns.iter().map(|r| r * r).sum::<f64>() / n as f64;
    h[0] = init.max(1e-12);
    for t in 1..n {
        let r = returns[t - 1];
        h[t] = omega + alpha * r * r + beta * h[t - 1];
    }
    h
}
