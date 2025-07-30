import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans

# Schwefel Function Definition
def schwefel(x):
    return 418.9828872724339 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

# Generate Random Points in the Schwefel Domain
num_points = 10000  # Adjust for resolution
dim = 2
X_samples = np.random.uniform(-500, 500, (num_points, dim))
y_samples = np.array([schwefel(x) for x in X_samples])

# Determine Optimal Number of Clusters Using the Elbow Method
def find_optimal_k(X, max_k=10):
    inertia = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    return inertia

# Prepare Data for Clustering (Include Schwefel Function Values)
data = np.column_stack((X_samples, y_samples))

# Find Optimal K using the Elbow Method (Optional, Uncomment to Use)
# inertia_values = find_optimal_k(data, max_k=10)
# plt.plot(range(1, 11), inertia_values, marker='o')
# plt.xlabel("Number of Clusters (K)")
# plt.ylabel("Inertia (Distortion)")
# plt.title("Elbow Method for Optimal K")
# plt.show()

# Set Number of Clusters (Can be Adjusted Manually or from Elbow Method)
num_clusters = 5  # Manually chosen, modify based on elbow method results

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
labels = kmeans.fit_predict(data)
optima_centers = kmeans.cluster_centers_[:, :2]  # Extract x1, x2 coordinates of cluster centers

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
plt.title("K-Means-Detected Local Optima in Schwefel Function")
plt.legend()
plt.show()