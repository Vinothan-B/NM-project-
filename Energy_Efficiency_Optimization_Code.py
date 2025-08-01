import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
data = {
    'hour': list(range(24)),
    'temperature': np.random.randint(20, 35, size=24),
    'energy_consumption': [100 + i * 3 + np.random.randint(-10, 10) for i in range(24)]
}
df = pd.DataFrame(data)
x = df[['hour', 'temperature']]
y = df['energy_consumption']
model = LinearRegression()
model.fit(x, y)
future_hours = pd.DataFrame({
    'hour': list(range(24)),
    'temperature': np.random.randint(20, 35, size=24)
})
predicted_energy = model.predict(future_hours)
optimized_energy = predicted_energy * 0.9
plt.figure(figsize=(10, 6))
plt.plot(future_hours['hour'], predicted_energy, label='Predicted Usage', color='blue', marker='o')
plt.plot(future_hours['hour'], optimized_energy, label='Optimized Usage', color='green', marker='x')
plt.title("Energy Usage Prediction & Optimization")
plt.xlabel("Hour of Day")
plt.ylabel("Energy Consumption (kWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("energy_optimization_plot.png")
plt.show()
import matplotlib.pyplot as plt
time = list(range(0, 24))
temperature = [22, 21, 20, 20, 21, 23, 25, 28, 30, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 21, 22, 23]
occupancy = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
energy_usage = [0.2, 0.2, 0.2, 0.3, 1.5, 2.0, 2.3, 3.0, 3.5, 3.8, 3.7, 3.6, 3.4, 3.3, 3.2, 3.0, 2.8, 2.5, 2.3, 2.0, 1.0, 0.8, 0.6, 0.4, 0.3]
plt.figure(figsize=(10, 6))
plt.plot(time, temperature, label='Temperature (°C)', color='orange', marker='o')
plt.plot(time, occupancy, label='Occupancy (0/1)', color='green', linestyle='--')
plt.plot(time, energy_usage, label='Energy Usage (kWh)', color='blue', marker='s')
plt.title('Energy Efficiency Data Visualization')
plt.xlabel('Time (Hour of Day)')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.xticks(time)
plt.tight_layout()
plt.show()
import torch
import time
import psutil
import os
model = torch.nn.Sequential(
    torch.nn.Linear(10, 5),
    torch.nn.ReLU(),
    torch.nn.Linear(5, 2)
)
model.eval()
input_data = torch.randn(1, 10)
process = psutil.Process(os.getpid())
cpu_before = process.cpu_percent(interval=None)
mem_before = process.memory_info().rss / (1024 ** 2)  
start = time.time()
with torch.no_grad():
    output = model(input_data)
end = time.time()
cpu_after = process.cpu_percent(interval=None)
mem_after = process.memory_info().rss / (1024 ** 2)
print("=== Simple Performance Metrics ===")
print(f"Inference Time: {(end - start) * 1000:.2f} ms")
print(f"CPU Usage Change: {cpu_after - cpu_before:.2f} %")
print(f"Memory Usage Change: {mem_after - mem_before:.2f} MB")
