from scipy.stats import entropy
import numpy as np
import json

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
# y_samples = np.array([schaffer(x[0], x[1]) for x in X_samples])

json_file = "_hpi_schwefelFixedTarget.json"
X_samples, y_samples = load_json_data(json_file)

# Compute histogram of fitness values
hist, _ = np.histogram(y_samples, bins=50, density=True)
landscape_entropy = entropy(hist)

print(f"Landscape Entropy (Fitness Diversity): {landscape_entropy:.6f}")