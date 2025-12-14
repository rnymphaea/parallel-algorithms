import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_all_results():
    if not os.path.exists('matrix_benchmarks.csv'):
        print("Benchmark data not found")
        return
        
    df = pd.read_csv('matrix_benchmarks.csv')
    print("Columns in CSV:", df.columns.tolist())
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    cpu_data = df[df['TestType'] == 'CPU']
    if not cpu_data.empty:
        for threads in cpu_data['Threads'].unique():
            thread_data = cpu_data[cpu_data['Threads'] == threads]
            axes[0, 0].plot(thread_data['MatrixSize'], thread_data['Time'], 
                           marker='o', label=f'{threads} threads')
        
        axes[0, 0].set_xlabel('Matrix Size')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].set_title('CPU Blocked Matrix Multiplication')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    gpu_data = df[df['TestType'] == 'GPU']
    if not gpu_data.empty:
        for workgroup in gpu_data['WorkgroupSize'].unique():
            wg_data = gpu_data[gpu_data['WorkgroupSize'] == workgroup]
            axes[0, 1].plot(wg_data['MatrixSize'], wg_data['Time'], 
                           marker='s', label=f'WG {workgroup}')
        
        axes[0, 1].set_xlabel('Matrix Size')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].set_title('GPU Blocked Matrix Multiplication')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    if not cpu_data.empty and not gpu_data.empty:
        best_cpu_times = cpu_data.groupby('MatrixSize')['Time'].min()
        best_gpu_times = gpu_data.groupby('MatrixSize')['Time'].min()
        
        common_sizes = best_cpu_times.index.intersection(best_gpu_times.index)
        speedup = best_cpu_times[common_sizes] / best_gpu_times[common_sizes]
        
        axes[1, 0].plot(speedup.index, speedup.values, 
                       marker='D', color='green', linewidth=2, markersize=8)
        
        axes[1, 0].set_xlabel('Matrix Size')
        axes[1, 0].set_ylabel('Speedup (CPU Time / GPU Time)')
        axes[1, 0].set_title('GPU Speedup over Best CPU Configuration')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=1, color='r', linestyle='--', alpha=0.5, label='No Speedup')
        axes[1, 0].legend()
        
        for size, speed in zip(speedup.index, speedup.values):
            axes[1, 0].annotate(f'{speed:.1f}x', 
                               xy=(size, speed),
                               xytext=(5, 5),
                               textcoords='offset points',
                               fontsize=9,
                               ha='left')
    
    comparison_data = df[df['TestType'] == 'CPU_GPU_Comparison']
    if not comparison_data.empty:
        axes[1, 1].plot(comparison_data['MatrixSize'], comparison_data['CPUTime'], 
                       marker='o', label='CPU (8 threads)', linewidth=2)
        axes[1, 1].plot(comparison_data['MatrixSize'], comparison_data['GPUTime'], 
                       marker='s', label='GPU (WG 16)', linewidth=2)
        
        axes[1, 1].set_xlabel('Matrix Size')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].set_title('CPU vs GPU Performance Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('matrix_benchmarks.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Matrix benchmarks plot saved as 'matrix_benchmarks.png'")
    
    if not cpu_data.empty and not gpu_data.empty:
        print("\nGPU Speedup Statistics:")
        print(f"Average speedup: {speedup.mean():.2f}x")
        print(f"Maximum speedup: {speedup.max():.2f}x")
        print(f"Minimum speedup: {speedup.min():.2f}x")
        print(f"Speedup for largest matrix ({speedup.index[-1]}): {speedup.iloc[-1]:.2f}x")

if __name__ == "__main__":
    plot_all_results()
