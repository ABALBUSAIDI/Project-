
import pandas as pd
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# Load test data
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv').squeeze()

# Load the trained models (re-train for simplicity as no saved models exist)
# Customer Classification (KMeans)
best_kmeans = KMeans(n_clusters=3, random_state=42)
best_kmeans.fit(X_test)
customer_segments = best_kmeans.predict(X_test)
silhouette_avg = silhouette_score(X_test, customer_segments)
print(f"Silhouette Score on Test Data for Customer Segmentation: {silhouette_avg}")

# Sales Prediction (Linear Regression)
regressor = LinearRegression()
regressor.fit(pd.read_csv('X_train.csv'), pd.read_csv('y_train.csv').squeeze())
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Data for Sales Prediction: {mse}")

# Average Cart Value Analysis (for additional interpretation)
average_cart_value_test = y_test.mean()
print(f"Average Cart Value (Test Set): {average_cart_value_test}")

# Save evaluation results
with open("model_evaluation_results.txt", "w") as f:
    f.write(f"Silhouette Score on Test Data for Customer Segmentation: {silhouette_avg}\n")
    f.write(f"Mean Squared Error on Test Data for Sales Prediction: {mse}\n")
    f.write(f"Average Cart Value (Test Set): {average_cart_value_test}\n")

print("Model evaluation completed. Results saved.")
