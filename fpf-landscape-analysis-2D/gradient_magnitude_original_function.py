import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def schwefel(x):
    return ((418.9828872724339 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))))/4000

def h1(x1, x2):
    term1 = np.sin(x1 - x2 / 8) ** 2
    term2 = np.sin(x2 + x1 / 8) ** 2
    denominator = np.sqrt((x1 - 8.6998) ** 2 + (x2 - 6.7665) ** 2) + 1
    return (-((term1 + term2) / denominator) + 2)/2

def schaffer(x1, x2):
    term1 = (x1 ** 2 + x2 ** 2) ** 0.25
    term2 = np.sin(50 * (x1 ** 2 + x2 ** 2) ** 0.10) ** 2
    return term1 * (term2 + 1.0)

# Generate random points
num_points = 25000  # Reduce for visualization purposes
X_samples = np.random.uniform(-100, 100, (num_points, 2))
y_samples = np.array([schaffer(x[0], x[1]) for x in X_samples])

# Build KDTree for nearest neighbor search
tree = KDTree(X_samples)


# Function to compute the gradient magnitude
def compute_gradient_magnitude(X_samples, y_samples, k=12):
    gradient_magnitudes = []

    for i, x in enumerate(X_samples):
        # Find k-nearest neighbors
        dists, indices = tree.query(x, k=k)
        neighbors = X_samples[indices]
        f_neighbors = y_samples[indices]

        # Compute gradient magnitudes for each neighbor
        grad_magnitudes = np.abs((f_neighbors - y_samples[i]) / (dists + 1e-8))  # Avoid division by zero

        # Compute average gradient magnitude
        avg_grad_magnitude = np.mean(grad_magnitudes)
        gradient_magnitudes.append(avg_grad_magnitude)

    return np.array(gradient_magnitudes)


# Compute gradient magnitudes for all points
gradient_magnitudes = compute_gradient_magnitude(X_samples, y_samples, k=12)

avg_gradient_magnitude = np.mean(gradient_magnitudes)
std_gradient_magnitude = np.std(gradient_magnitudes)

print(f"Average Gradient Magnitude: {avg_gradient_magnitude:.6f} ± {std_gradient_magnitude:.6f}")

# Create a heatmap of gradient magnitudes
plt.figure(figsize=(8, 6))
sc = plt.scatter(X_samples[:, 0], X_samples[:, 1], c=gradient_magnitudes, cmap='viridis', s=5, alpha=0.7)
plt.colorbar(sc, label="Gradient Magnitude")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Heatmap of Gradient Magnitude (Landscape Smoothness)")
plt.show()