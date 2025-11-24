#!/usr/bin/env bash
set -e

mkdir -p plots
make

echo "=== Default Benchmark ==="
./bin/list_bench

echo -e "\n=== High Read Workload (90% find) ==="
./bin/list_bench -t 1,2,4,8 -i 0.05 -r 0.05 -f 0.9 -O results_high_read.csv

echo -e "\n=== High Write Workload (50% updates) ==="
./bin/list_bench -t 1,2,4,8 -i 0.25 -r 0.25 -f 0.5 -O results_high_write.csv

echo -e "\n=== Mixed Workload ==="
./bin/list_bench -t 1,4,16 -i 0.1 -r 0.2 -f 0.7 -o 50000 -O results_mixed.csv

echo -e "\n=== Generating plots ==="
for csv in results.csv results_high_read.csv results_high_write.csv results_mixed.csv; do
    if [ -f "$csv" ]; then
        python3 plot_results.py "$csv"
    fi
done

echo "Done. See results_*.csv and plots/"
