import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('matrix_results.csv')
sizes = sorted(df['Size'].unique())
threads = [1, 2]

fig, ax = plt.subplots(figsize=(10, 6))

colors = {'Strassen': 'blue', 'Block': 'red'}
markers = {1: 'o', 2: 's'}

for algorithm in ['Strassen', 'Block']:
    for thread_count in threads:
        data = df[(df['Algorithm'] == algorithm) & (df['Threads'] == thread_count)]
        if not data.empty:
            data = data.sort_values('Size')
            label = f'{algorithm} ({thread_count} thread{"s" if thread_count > 1 else ""})'
            ax.plot(data['Size'], data['Time'], 
                   marker=markers[thread_count],
                   color=colors[algorithm],
                   linewidth=2,
                   markersize=8,
                   label=label)

ax.set_xlabel('Matrix Size (N)')
ax.set_ylabel('Time (seconds)')
ax.set_title('Matrix Multiplication Performance')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('matrix_plot.png', dpi=300)
plt.show()
