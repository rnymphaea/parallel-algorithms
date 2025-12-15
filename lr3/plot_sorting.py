import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('sorting_results.csv')

fig, ax = plt.subplots(figsize=(10, 6))

single_thread_data = df[df['Algorithm'] == 'single_thread_merge_sort']
parallel_data = df[df['Algorithm'] == 'parallel_merge_sort']

if not single_thread_data.empty:
    ax.plot(single_thread_data['Size'], single_thread_data['Time'], 'ko-', 
            linewidth=2, markersize=8, label='Single-thread merge sort')

if not parallel_data.empty:
    ax.plot(parallel_data['Size'], parallel_data['Time'], 'ro-', 
            linewidth=2, markersize=8, label='Parallel merge sort (2 threads)')

ax.set_xlabel('Array Size')
ax.set_ylabel('Time (seconds)')
ax.set_title('Sorting Performance: Single-thread vs Parallel')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('sorting_plot.png', dpi=300)
plt.show()
