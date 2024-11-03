
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, silhouette_score

# Load preprocessed training data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()  # Squeeze to convert DataFrame to Series
y_test = pd.read_csv('y_test.csv').squeeze()

# Task 1: Customer Classification (Clustering)
# Apply KMeans clustering to identify customer segments
kmeans = KMeans(n_clusters=3, random_state=42)
customer_segments = kmeans.fit_predict(X_train)
silhouette_avg = silhouette_score(X_train, customer_segments)
print(f"Silhouette Score for Customer Segmentation: {silhouette_avg}")

# Task 2: Sales Prediction (Regression)
# Train a linear regression model to predict total_order_value
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error for Sales Prediction: {mse}")

# Task 3: Average Cart Value Analysis
# Calculate average cart value from training set and analyze trends
average_cart_value = y_train.mean()
print(f"Average Cart Value (Training Set): {average_cart_value}")

# Save model outputs and metrics for reference
with open("model_metrics.txt", "w") as f:
    f.write(f"Silhouette Score for Customer Segmentation: {silhouette_avg}\n")
    f.write(f"Mean Squared Error for Sales Prediction: {mse}\n")
    f.write(f"Average Cart Value (Training Set): {average_cart_value}\n")

print("Model training and metrics calculation completed.")
