import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULTS_DIR = "."  # folder holding the output_<schedule>_<threads>.txt files

# Mapping filenames to experiment configurations
files = {
    "output_static_4.txt": ("static", 4),
    "output_static_8.txt": ("static", 8),
    "output_static_16.txt": ("static", 16),
    "output_static_32.txt": ("static", 32),
    "output_dynamic_4.txt": ("dynamic", 4),
    "output_dynamic_8.txt": ("dynamic", 8),
    "output_dynamic_16.txt": ("dynamic", 16),
    "output_dynamic_32.txt": ("dynamic", 32),
}

# Prepare DataFrame to collect results
results = []

# Process each file
for filename, (schedule, threads) in files.items():
    with open(os.path.join(RESULTS_DIR, filename), "r") as file:
        lines = file.readlines()
        time_line = lines[0]
        avg_time = float(time_line.split(":")[1].strip())
        results.append({"Threads": threads, "Schedule": schedule, "AvgTime": avg_time})

# Create DataFrame
df = pd.DataFrame(results)

# Sort for consistent plotting
df.sort_values(by=["Schedule", "Threads"], inplace=True)

# Speedup calculation
seq_time = df["AvgTime"].max()
df["Speedup"] = seq_time / df["AvgTime"]

# Plotting
sns.set(style="whitegrid")

# Execution Time Plot
plt.figure(figsize=(10, 6))
sns.barplot(x="Threads", y="AvgTime", hue="Schedule", data=df)
plt.title("Average Execution Time vs Threads")
plt.ylabel("Avg Time (s)")
plt.xlabel("Number of Threads")
plt.legend(title="Schedule")
plt.tight_layout()
plt.show()

# Speedup Plot
plt.figure(figsize=(10, 6))
sns.barplot(x="Threads", y="Speedup", hue="Schedule", data=df)
plt.title("Speedup vs Threads")
plt.ylabel("Speedup")
plt.xlabel("Number of Threads")
plt.legend(title="Schedule")
plt.tight_layout()
plt.show()