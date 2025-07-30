import json
import numpy as np
import matplotlib.pyplot as plt

with open('_hpi_schwefelFixedTarget.json', 'r') as file:
    data = json.load(file)

tile_size = 50

x1_values = [ind['x1'] for entry in data for ind in entry['individualsWithFPF']]
x2_values = [ind['x2'] for entry in data for ind in entry['individualsWithFPF']]
x1_min, x1_max = min(x1_values), max(x1_values)
x2_min, x2_max = min(x2_values), max(x2_values)

x1_bins = np.arange(x1_min, x1_max + tile_size, tile_size)
x2_bins = np.arange(x2_min, x2_max + tile_size, tile_size)

grid_values = np.full((len(x1_bins), len(x2_bins)), np.nan)
count_values = np.zeros((len(x1_bins), len(x2_bins)))

# Accumulate values into grid
for entry in data:
    for individual in entry['individualsWithFPF']:
        if individual['fpfValue'] != 1.0:
            x1_idx = np.searchsorted(x1_bins, individual['x1'], side='right') - 1
            x2_idx = np.searchsorted(x2_bins, individual['x2'], side='right') - 1

            if 0 <= x1_idx < len(x1_bins) and 0 <= x2_idx < len(x2_bins):
                if np.isnan(grid_values[x1_idx, x2_idx]):
                    grid_values[x1_idx, x2_idx] = 0
                grid_values[x1_idx, x2_idx] += individual['fpfValue']
                count_values[x1_idx, x2_idx] += 1

# Compute the average
with np.errstate(invalid='ignore'):
    grid_values /= count_values

fig, ax = plt.subplots(figsize=(10, 8))
c = ax.pcolormesh(x1_bins, x2_bins, grid_values.T, cmap='viridis', shading='auto')
plt.colorbar(c, label='Average fpfValue')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title(f'HPI Schwefel Fixed Target FPF Heatmap with Tile Size {2*tile_size}x{2*tile_size}')

plt.show()