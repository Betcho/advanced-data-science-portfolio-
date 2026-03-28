from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

pipeline = Pipeline([
    ("poly", PolynomialFeatures(degree=3)),
    ("model", Ridge(alpha=1.0))
])

scores = cross_val_score(pipeline, X_train, y_train, cv=5)

print("CV Score:", scores.mean())
