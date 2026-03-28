import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("data/sales.csv")
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# Train model
model = ARIMA(df["sales"], order=(3,1,2))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=12)

print(forecast)
