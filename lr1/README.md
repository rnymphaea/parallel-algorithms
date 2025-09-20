# Лабораторная работа №1  
**Тема:** Основы работы с потоками  

## Задание
Реализовать многопоточное умножение матриц с блочным разбиением.  

1.1 Использовать `std::thread` для создания потоков. Провести исследование метрик производительности в зависимости от количества потоков и в зависимости от размера задачи, выбрать количество потоков, при котором достигается оптимальная производительность.  

1.2 Использовать `std::async` и `std::future`.  

Реализация должна содержать тесты на корректность вычислений: состоящие в сравнении с однопоточной реализацией перемножения матриц.  

Метрики производительности приводить с осреднением, в отчете построить графики.  

## Установка и запуск
```bash
git clone https://github.com/rnymphaea/parallel-algorithms.git && \
cd parallel-algorithms/lr1
```
Сборка осуществляется с помощью `make`.  

```bash
make
```
Запуск программы:
```bash
./mm [OPTIONS]
```

Очистка артефактов сборки:
```bash
make clean
```

## CLI
Программа поддерживает **CLI-аргументы**:
```
Usage: ./mm [OPTIONS]

Matrix multiplication program with single-threaded, multi-threaded, and async implementations.

Options:
  -r, --rows M            Number of rows for randomly generated matrices (default: 4)
  -c, --columns N         Number of columns for randomly generated matrices (default: 4)
  -a, --path-a FILE       Load matrix A from the specified file
  -b, --path-b FILE       Load matrix B from the specified file
  -T, --time              Measure execution time for multi-threaded and async multiplication
  -n, --repeats N         Number of repetitions to average timing results (default: 3)
  -t, --threads N         Number of threads (tasks) to for multi-threaded (async) multiplication (default: number of hardware threads available on the system)
  -o, --output FILE       Specify output file to save result
  -d, --debug             Enable debug mode
  -e, --export-csv FILE   Export timing results to CSV file (append mode).
                          Format: threads,single,multi,async
  -h, --help              Display this help message and exit

Notes:
- If --path-a or --path-b are not specified, the matrices will be generated randomly.
- If --time is not specified, the program will compute the result without measuring execution time,
  even if --repeats is given.
- Matrices larger than 10x10 will not be displayed on the console, unless the --debug flag is used.
- You can optionally add --output FILE to save the result matrix to a file.

```
```
