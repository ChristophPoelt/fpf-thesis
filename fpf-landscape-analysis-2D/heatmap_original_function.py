import numpy as np
import matplotlib.pyplot as plt

def schwefel(x1, x2):
    return 418.9828872724339 * 2 - (x1 * np.sin(np.sqrt(np.abs(x1))) + x2 * np.sin(np.sqrt(np.abs(x2))))

def h1(x1, x2):
    term1 = np.sin(x1 - x2 / 8) ** 2
    term2 = np.sin(x2 + x1 / 8) ** 2
    denominator = np.sqrt((x1 - 8.6998) ** 2 + (x2 - 6.7665) ** 2) + 1
    return (term1 + term2) / denominator

def schaffer(x1, x2):
    term1 = (x1 ** 2 + x2 ** 2) ** 0.25
    term2 = np.sin(50 * (x1 ** 2 + x2 ** 2) ** 0.10) ** 2
    return term1 * (term2 + 1.0)

# Definiere den Bereich der Heatmap, unterschiedlich je nach Funktion
x_min, x_max = -100, 100
y_min, y_max = -100, 100

# Erstelle ein Gitter von x- und y-Werten
resolution = 500  # Anzahl der Punkte pro Achse
x = np.linspace(x_min, x_max, resolution)
y = np.linspace(y_min, y_max, resolution)
X, Y = np.meshgrid(x, y)

# Verwende die gewünschte Funktion hier
Z = schaffer(X, Y)

plt.figure(figsize=(8, 6))
plt.imshow(Z, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis')
plt.colorbar(label='Schaffer target value')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Heatmap of the original Schaffer Function')
plt.show()