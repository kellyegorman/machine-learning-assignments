import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# df for correlation
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# correlation matrix
corr_matrix = df.corr()
print("Correlation Matrix:\n")
print(corr_matrix)

# correlation with target
target_corr = corr_matrix['target'].drop('target')

# get top 2 features
top2_features = target_corr.abs().sort_values(ascending=False).head(2).index.tolist()

print("2 features most correlated with target:")
print(top2_features)

X_selected = df[top2_features].values
X_train, X_test, y_train, y_test = train_test_split(
    # 80/20 split 
    X_selected, y, test_size=0.2, random_state=42
)

# OLS (linear regression)
model_linear = LinearRegression()
model_linear.fit(X_train, y_train)

y_pred_linear = model_linear.predict(X_test)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
print("MAE (Linear Regression):", mae_linear)

# quadratic regression
poly = PolynomialFeatures(degree=2, include_bias=False)

# transform features to include polynomial terms
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# fit linear regression on polynomial features
model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)

# predict and evaluate
y_pred_poly = model_poly.predict(X_test_poly)

# compute mae for quadratic 
mae_poly = mean_absolute_error(y_test, y_pred_poly)
print("MAE (Quadratic Model):", mae_poly)