import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN

# Schwefel Function Definition
def schwefel(x):
    return 418.9828872724339 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

# Generate Random Points in the Schwefel Domain
num_points = 25000  # Adjust for resolution
dim = 2
X_samples = np.random.uniform(-500, 500, (num_points, dim))
y_samples = np.array([schwefel(x) for x in X_samples])

# Apply DBSCAN Clustering to Identify Local Optima
dbscan = DBSCAN(eps=50, min_samples=5, metric='euclidean')  # Tune 'eps' and 'min_samples'
labels = dbscan.fit_predict(np.column_stack((X_samples, y_samples)))

# Extract Cluster Centers (Local Optima)
unique_labels = set(labels)
optima_centers = np.array([X_samples[labels == i].mean(axis=0) for i in unique_labels if i != -1])

# Compute Pairwise Distances and Average Distance Between Optima
if len(optima_centers) > 1:
    distances = distance_matrix(optima_centers, optima_centers)
    np.fill_diagonal(distances, np.nan)  # Ignore self-distances
    avg_distance = np.nanmean(distances)
else:
    avg_distance = np.nan

# Print Results
num_optima = len(optima_centers)
print(f"Number of Local Optima Detected: {num_optima}")
print(f"Average Distance Between Optima: {avg_distance:.2f}")

# Visualize Optima and Data Points
plt.figure(figsize=(8, 6))
plt.scatter(X_samples[:, 0], X_samples[:, 1], c=labels, cmap='tab10', s=5, alpha=0.5, label="Sample Points")
plt.scatter(optima_centers[:, 0], optima_centers[:, 1], c='red', marker='X', s=100, label="Local Optima")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("DBSCAN-Detected Local Optima in Schwefel Function")
plt.legend()
plt.show()