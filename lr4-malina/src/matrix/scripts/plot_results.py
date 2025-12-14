import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_all_results():
    if not os.path.exists('all_benchmarks.csv'):
        print("Benchmark data not found")
        return
        
    df = pd.read_csv('all_benchmarks.csv')
    print("Columns in CSV:", df.columns.tolist())
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: CPU Performance
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
        # Убрана логарифмическая шкала
    
    # Plot 2: GPU Performance
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
        # Убрана логарифмическая шкала
    
    # Plot 3: GPU Speedup over Best CPU
    if not cpu_data.empty and not gpu_data.empty:
        # Находим лучшее время для CPU (минимальное время для каждого размера матрицы)
        best_cpu_times = cpu_data.groupby('MatrixSize')['Time'].min()
        
        # Находим лучшее время для GPU (минимальное время для каждого размера матрицы)
        best_gpu_times = gpu_data.groupby('MatrixSize')['Time'].min()
        
        # Вычисляем ускорение GPU над CPU
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
        
        # Добавляем аннотации с значениями ускорения
        for size, speed in zip(speedup.index, speedup.values):
            axes[1, 0].annotate(f'{speed:.1f}x', 
                               xy=(size, speed),
                               xytext=(5, 5),
                               textcoords='offset points',
                               fontsize=9,
                               ha='left')
    
    # Plot 4: CPU-GPU Comparison
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
        # Убрана логарифмическая шкала
    
    plt.tight_layout()
    plt.savefig('all_benchmarks.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("All benchmarks plot saved as 'all_benchmarks.png'")
    
    # Выводим статистику по ускорению
    if not cpu_data.empty and not gpu_data.empty:
        print("\nGPU Speedup Statistics:")
        print(f"Average speedup: {speedup.mean():.2f}x")
        print(f"Maximum speedup: {speedup.max():.2f}x")
        print(f"Minimum speedup: {speedup.min():.2f}x")
        print(f"Speedup for largest matrix ({speedup.index[-1]}): {speedup.iloc[-1]:.2f}x")

if __name__ == "__main__":
    plot_all_results()