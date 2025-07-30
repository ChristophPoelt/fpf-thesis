import json
import numpy as np
import matplotlib.pyplot as plt

with open('_hpi_schwefelFixedTarget.json', 'r') as file:
    data = json.load(file)

x1 = []
x2 = []
fpfValue = []
generation_fitness = {}

total_runs = len(data)

total_generations = []
final_fitness_values = []

for entry in data:
    last_fitness_value = None

    for individual in entry['individualsWithFPF']:
        if individual['fpfValue'] != 1.0:  # Exclude fpfValue of 1.0
            x1.append(individual['x1'])
            x2.append(individual['x2'])
            fpfValue.append(individual['fpfValue'])

        total_generations.append(entry['totalGenerations'])
        final_fitness_values.append(entry['finalBestFitness'])

    for gen in range(50):
        if gen < len(entry['bestIndividualsPerGeneration']):
            last_fitness_value = entry['bestIndividualsPerGeneration'][gen]['targetValue']
        if gen not in generation_fitness:
            generation_fitness[gen] = []
        if last_fitness_value is not None:
            generation_fitness[gen].append(last_fitness_value)

generations = sorted(generation_fitness.keys())
avg_fitness = [np.mean(generation_fitness[gen]) for gen in generations]
std_dev = [np.std(generation_fitness[gen]) for gen in generations]

avg_generations = np.mean(total_generations)
avg_final_fitness = np.mean(final_fitness_values)
title_text = f'Schwefel Fixed Generations; Average {avg_generations:.2f} generations; Average Result: {avg_final_fitness:.6f}; Total Runs: {total_runs}'

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x1, x2, fpfValue, c=fpfValue, cmap='viridis', marker='o')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('fpfValue')
ax.set_title(title_text)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(generations, avg_fitness, marker='o', label='Average Best Fitness')
plt.fill_between(generations, np.array(avg_fitness) - np.array(std_dev), np.array(avg_fitness) + np.array(std_dev), color='b', alpha=0.2, label='Standard Deviation')
plt.xlabel('Generation')
plt.ylabel('Average Best Fitness')
plt.title('Average Best Individual Fitness Over Generations')
plt.legend()
plt.grid()
plt.show()