import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_matrix_results():
    if not os.path.exists('matrix_benchmarks.csv'):
        print("Benchmark data not found")
        return
        
    df = pd.read_csv('matrix_benchmarks.csv')
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    cpu_data = df[df['TestType'] == 'CPU']
    gpu_data = df[df['TestType'] == 'GPU']
    
    if not cpu_data.empty:
        for threads in sorted(cpu_data['Threads'].unique()):
            thread_data = cpu_data[cpu_data['Threads'] == threads]
            ax.plot(thread_data['MatrixSize'], thread_data['Time'], 
                   marker='o', linewidth=2, markersize=8,
                   label=f'CPU {threads} threads')
    
    if not gpu_data.empty:
        for workgroup in sorted(gpu_data['WorkgroupSize'].unique()):
            wg_data = gpu_data[gpu_data['WorkgroupSize'] == workgroup]
            ax.plot(wg_data['MatrixSize'], wg_data['Time'], 
                   marker='s', linewidth=2, markersize=8, linestyle='--',
                   label=f'GPU WG {workgroup}')
    
    ax.set_xlabel('Matrix Size (N x N)', fontsize=12)
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.set_title('Matrix Multiplication Performance: CPU vs GPU', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('matrix_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Matrix performance plot saved as 'matrix_performance.png'")

if __name__ == "__main__":
    plot_matrix_results()
