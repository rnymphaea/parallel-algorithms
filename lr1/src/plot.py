import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python3 plot.py <csv_file> [output_file]")
    sys.exit(1)

csv_file = sys.argv[1]
output_file = sys.argv[2] if len(sys.argv) > 2 else "performance.png"

data = pd.read_csv(csv_file)

threads = data["threads"].tolist()
x = list(range(len(threads)))

plt.figure(figsize=(10,6))
plt.plot(x, data["single"], marker="o", label="Single-thread")
plt.plot(x, data["multi"], marker="o", label="Multi-thread")
plt.plot(x, data["async"], marker="o", label="Async")

plt.xlabel("Number of threads / tasks")
plt.ylabel("Execution time (seconds)")
plt.title("Matrix multiplication performance")
plt.legend()
plt.grid(True)

plt.xticks(x, threads)

plt.savefig(output_file, dpi=300)
plt.show()

