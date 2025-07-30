import numpy as np
import json
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
    return (term1 * (term2 + 1.0))/25

def load_json_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    X_samples = []
    y_samples = []
    for entry in data:
        for individual in entry['individualsWithFPF']:
            if individual['fpfValue'] != 1.0:  # Exclude fpfValue of 1.0
                X_samples.append([individual['x1'], individual['x2']])
                y_samples.append(individual['fpfValue'])

    return np.array(X_samples), np.array(y_samples)

# Generate random points
# num_points = 10000
# X_samples = np.random.uniform(-100, 100, (num_points, 2))
# y_samples = np.array([schwefel(x) for x in X_samples])

json_file = "_schwefelFixedTarget.json"
X_samples, y_samples = load_json_data(json_file)

def compute_convexity_ratio(X_samples, y_samples, num_pairs=1000, num_interpolations=10):
    n = len(X_samples)
    cr_counter = 0
    total = 0
    tree = KDTree(X_samples)
    y_samples = np.array(y_samples)

    for _ in range(num_pairs):
        i, j = np.random.choice(n, size=2, replace=False)
        xa, xb = X_samples[i], X_samples[j]
        fa, fb = y_samples[i], y_samples[j]

        for l in range(1, num_interpolations + 1):
            lam = l / (num_interpolations + 1)
            x_lambda = (1 - lam) * xa + lam * xb
            f_interp = (1 - lam) * fa + lam * fb

            _, idx = tree.query(x_lambda)
            f_near = y_samples[idx]

            if f_near <= f_interp:
                cr_counter += 1
            total += 1

    convexity_ratio = cr_counter / total if total > 0 else 0
    return convexity_ratio


convexity_ratio = compute_convexity_ratio(X_samples, y_samples)
print(f"Convexity Ratio: {convexity_ratio:.6f}")