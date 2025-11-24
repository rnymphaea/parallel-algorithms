#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
import os

if len(sys.argv) < 2:
    print("Usage: plot_results.py results.csv")
    sys.exit(1)

csv = sys.argv[1]
df = pd.read_csv(csv)

# simple line plot: threads vs ops_per_sec for each impl
pivot = df.pivot(index='threads', columns='impl', values='ops_per_sec')
ax = pivot.plot(marker='o', title='Throughput (ops/s) vs threads')
ax.set_xlabel('threads')
ax.set_ylabel('ops/s')
plt.grid(True)
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/throughput_vs_threads.png', dpi=200)
print("Saved plots/throughput_vs_threads.png")

