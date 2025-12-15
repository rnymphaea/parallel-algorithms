import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_sort_results():
    if not os.path.exists('sort_benchmarks.csv'):
        print("Benchmark data not found. Run benchmarks first.")
        return
    
    df = pd.read_csv('sort_benchmarks.csv')
    
    cpu_data = df[df['Implementation'] == 'CPU']
    gpu_data = df[df['Implementation'] == 'GPU']
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    if not cpu_data.empty:
        cpu_configs = cpu_data['Config'].unique()
        
        for config in sorted(cpu_configs):
            config_data = cpu_data[cpu_data['Config'] == config]
            config_data = config_data.groupby('ArraySize')['Time'].mean().reset_index()
            
            if 'threads' in str(config):
                thread_count = str(config).split()[0]
                label = f'CPU {thread_count} threads'
            elif 'std::sort' in str(config):
                label = 'CPU std::sort'
            else:
                label = f'CPU {config}'
            
            ax.plot(config_data['ArraySize'], config_data['Time'], 
                   marker='o', linewidth=2, markersize=8, linestyle='-',
                   label=label)
    
    if not gpu_data.empty:
        gpu_configs = gpu_data['Config'].unique()
        
        for config in sorted(gpu_configs):
            config_data = gpu_data[gpu_data['Config'] == config]
            config_data = config_data.groupby('ArraySize')['Time'].mean().reset_index()
            
            if 'WG' in str(config):
                wg_size = str(config).replace('WG', '')
                label = f'GPU WG{wg_size}'
            else:
                label = f'GPU {config}'
            
            ax.plot(config_data['ArraySize'], config_data['Time'], 
                   marker='s', linewidth=2, markersize=8, linestyle='--',
                   label=label)
    
    ax.set_xlabel('Array Size', fontsize=12)
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.set_title('Sorting Algorithm Performance: CPU vs GPU', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('sort_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Sort performance plot saved as 'sort_performance.png'")

if __name__ == "__main__":
    plot_sort_results()
