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
  --rows [-r] M            Number of rows for randomly generated matrices (default: 4)
  --columns [-c] N         Number of columns for randomly generated matrices (default: 4)
  --path-a [-a] FILE       Load matrix A from the specified file
  --path-b [-b] FILE       Load matrix B from the specified file
  --time [-t]              Measure execution time for multi-threaded and async multiplication
  --repeats [-n] N         Number of repetitions to average timing results (default: 3)
  --threads [-p] N         Number of threads (tasks) to use for multi-threaded (async) multiplication
  --output [-o] FILE       Specify output file to save result
  --help [-h]              Display this help message and exit

Notes:
- If --path-a or --path-b are not specified, the matrices will be generated randomly.
- If --time is not specified, the program will compute the result without measuring execution time,
  even if --repeats is given.
- Matrices larger than 10x10 are not printed to the console.
- You can optionally add --output FILE to save the result matrix to a file.
```

