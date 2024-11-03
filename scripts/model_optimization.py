import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# Load preprocessed training data
X_train = pd.read_csv('X_train.csv')

# Task 1: Hyperparameter Tuning for Customer Classification (KMeans)
# Define a parameter grid for number of clusters
kmeans_param_grid = {'n_clusters': [2, 3, 4, 5, 6]}
kmeans = KMeans(random_state=42)
kmeans_grid_search = GridSearchCV(kmeans, kmeans_param_grid, cv=5)  # No scoring here
kmeans_grid_search.fit(X_train)

# Get the best model and calculate its silhouette score
best_kmeans = kmeans_grid_search.best_estimator_
customer_segments = best_kmeans.predict(X_train)
best_silhouette = silhouette_score(X_train, customer_segments)

print(f"Best KMeans Model: {best_kmeans}")
print(f"Best Silhouette Score: {best_silhouette}")

# Save optimization results to a text file
with open("model_optimization_results.txt", "w") as f:
    f.write(f"Best KMeans Model: {best_kmeans}\\n")
    f.write(f"Best Silhouette Score: {best_silhouette}\\n")

print("Model optimization and hyperparameter tuning completed.")
