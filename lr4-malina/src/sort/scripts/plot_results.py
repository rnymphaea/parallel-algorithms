import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def plot_benchmark_results():
    """Генерирует графики из benchmark результатов"""
    
    results_dir = Path("results")
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    try:
        df = pd.read_csv(results_dir / "benchmark_results.csv")
    except FileNotFoundError:
        print("Benchmark results not found. Run benchmarks first.")
        return
    
    # Разделяем данные на CPU и GPU
    cpu_data = df[df['Implementation'] == 'CPU']
    gpu_data = df[df['Implementation'] == 'GPU']
    
    # Настройка стиля
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # График 1: Сравнение CPU потоков
    cpu_pivot = cpu_data.pivot_table(index='ArraySize', columns='Config', values='Time', aggfunc='mean')
    if not cpu_pivot.empty:
        cpu_pivot.plot(ax=axes[0,0], marker='o', linewidth=2, markersize=6)
        axes[0,0].set_xlabel('Array Size')
        axes[0,0].set_ylabel('Time (seconds)')
        axes[0,0].set_title('CPU Performance: Different Thread Counts')
        axes[0,0].legend(title='CPU Config')
        axes[0,0].grid(True, alpha=0.3)
    
    # График 2: Сравнение GPU конфигураций
    gpu_pivot = gpu_data.pivot_table(index='ArraySize', columns='Config', values='Time', aggfunc='mean')
    if not gpu_pivot.empty:
        gpu_pivot.plot(ax=axes[0,1], marker='s', linewidth=2, markersize=6)
        axes[0,1].set_xlabel('Array Size')
        axes[0,1].set_ylabel('Time (seconds)')
        axes[0,1].set_title('GPU Performance: Different Work Group Configurations')
        axes[0,1].legend(title='GPU Config')
        axes[0,1].grid(True, alpha=0.3)
    
    # График 3: Лучший CPU vs Лучший GPU
    best_cpu = cpu_data.groupby('ArraySize')['Time'].min()
    best_gpu = gpu_data.groupby('ArraySize')['Time'].min()
    
    if not best_cpu.empty and not best_gpu.empty:
        axes[1,0].plot(best_cpu.index, best_cpu.values, 'o-', label='Best CPU', linewidth=3, markersize=8)
        axes[1,0].plot(best_gpu.index, best_gpu.values, 's-', label='Best GPU', linewidth=3, markersize=8)
        axes[1,0].set_xlabel('Array Size')
        axes[1,0].set_ylabel('Time (seconds)')
        axes[1,0].set_title('Best CPU vs Best GPU Performance')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # График 4: Ускорение GPU vs CPU
    if not best_cpu.empty and not best_gpu.empty:
        speedup = best_cpu / best_gpu
        colors = ['green' if x > 1 else 'red' for x in speedup]
        
        bars = axes[1,1].bar(range(len(speedup)), speedup.values, color=colors, alpha=0.7)
        axes[1,1].set_xlabel('Array Size')
        axes[1,1].set_ylabel('Speedup (CPU Time / GPU Time)')
        axes[1,1].set_title('GPU Speedup vs Best CPU Configuration')
        axes[1,1].set_xticks(range(len(speedup)))
        axes[1,1].set_xticklabels([f"{size:,}" for size in speedup.index], rotation=45)
        axes[1,1].axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Break-even')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Добавляем значения на столбцы
        for bar, value in zip(bars, speedup.values):
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                          f'{value:.2f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'comprehensive_benchmark.png', dpi=300, bbox_inches='tight')
    plt.savefig(plots_dir / 'comprehensive_benchmark.pdf', bbox_inches='tight')
    plt.show()
    
    print(f"Plots saved to {plots_dir}/")

if __name__ == "__main__":
    print("Generating comprehensive benchmark plots...")
    plot_benchmark_results()