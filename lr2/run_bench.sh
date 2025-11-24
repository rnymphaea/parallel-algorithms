#!/usr/bin/env bash
set -e
mkdir -p plots
make
./bin/list_bench
python3 plot_results.py results.csv
echo "Done. See results.csv and plots/"
