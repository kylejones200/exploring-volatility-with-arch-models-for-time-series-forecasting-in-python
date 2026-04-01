# Description: Short example for Exploring Volatility with ARCH Models for Time Series Forecasting in Python.



from arch import arch_model
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)



# Set random seed for reproducibility
np.random.seed(42)

# Simulate returns with volatility clustering
n = 1000
omega = 0.1
alpha = 0.8

errors = np.random.normal(size=n)
volatility = np.zeros(n)
returns = np.zeros(n)

for t in range(1, n):
    volatility[t] = np.sqrt(omega + alpha * errors[t-1]**2)
    returns[t] = volatility[t] * np.random.normal()

# Create a DataFrame
data = pd.DataFrame({"returns": returns, "volatility": volatility})
data.plot(subplots=True, figsize=(10, 6), title="Simulated Returns and Volatility")
plt.show()



# Fit an ARCH(1) model
arch_model_fit = arch_model(data["returns"], vol="ARCH", p=1).fit()
logger.info(arch_model_fit.summary())

# Forecast volatility
forecast = arch_model_fit.forecast(horizon=10)
forecast_variance = forecast.variance.iloc[-1]

# Plot forecasted volatility
plt.figure(figsize=(10, 6))
plt.plot(forecast_variance, marker="o", label="Forecasted Variance")
plt.title("Forecasted Volatility")
plt.xlabel("Horizon")
plt.ylabel("Variance")
plt.legend()
plt.show()
