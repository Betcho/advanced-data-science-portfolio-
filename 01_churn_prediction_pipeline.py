import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# Load
df = pd.read_csv("data/churn.csv")

# Feature engineering
df["avg_value"] = df["total_value"] / (df["num_transactions"] + 1)

# Clean
df = df.dropna()

X = df.drop("churn", axis=1)
y = df["churn"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Evaluate
preds = model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, preds))
