#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results(csv_file):
    df = pd.read_csv(csv_file)
    
    config_name = os.path.splitext(os.path.basename(csv_file))[0].replace('results_', '')
    if config_name == 'results':
        config_name = 'default'
    
    ratios = df[['p_insert', 'p_remove']].iloc[0]
    find_ratio = 1.0 - ratios['p_insert'] - ratios['p_remove']
    ops_per_thread = df['ops_per_thread'].iloc[0]
    key_range = df['key_range'].iloc[0]
    
    plt.figure(figsize=(10, 6))
    
    pivot = df.pivot(index='threads', columns='impl', values='ops_per_sec')
    pivot.plot(marker='o', linewidth=2, markersize=8)
    
    plt.title(f'Linked List Throughput\n'
              f'Insert: {ratios["p_insert"]:.1%}, Remove: {ratios["p_remove"]:.1%}, Find: {find_ratio:.1%}\n'
              f'Ops/thread: {ops_per_thread:,}, Key range: {key_range:,}')
    plt.xlabel('Number of Threads')
    plt.ylabel('Throughput (operations/second)')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Implementation')
    
    os.makedirs('plots', exist_ok=True)
    output_file = f'plots/throughput_{config_name}.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"Saved {output_file}")
    
    print(f"\n{config_name.upper()} WORKLOAD SUMMARY:")
    print(f"Operation ratios: Insert={ratios['p_insert']:.1%}, "
          f"Remove={ratios['p_remove']:.1%}, Find={find_ratio:.1%}")
    print("Throughput results (ops/sec):")
    for threads in pivot.index:
        print(f"  Threads {threads}:")
        for impl in pivot.columns:
            throughput = pivot.loc[threads, impl]
            print(f"    {impl:6}: {throughput:12,.0f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        result_files = [f for f in os.listdir('.') if f.startswith('results') and f.endswith('.csv')]
        if not result_files:
            print("Usage: plot_results.py results.csv")
            print("Or run without arguments to plot all result files in current directory")
            sys.exit(1)
        
        for csv_file in result_files:
            plot_results(csv_file)
    else:
        for csv_file in sys.argv[1:]:
            plot_results(csv_file)
