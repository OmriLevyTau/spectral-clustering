# C - compile and run #

## compile and run with user input ##

general:
1. `gcc -Wall .\spkmeans.c -o run `
2. `.\run.exe ddg input `

specific:
1. `gcc -Wall .\spkmeans.c -o run `
2. `.\run.exe jacobi C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\inputs\\sym_matrix\\sym_matrix_input_9.txt`
3. `.\run.exe lnorm C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\inputs\\data_points\\input_0.txt`


## connect two files - compile and run ##
For invoking method foo() defined in spkmeans.c in test.c:
1. set `#include "spkmeans.h"` in the top of test.c file
2. run the program using IDE

Alternatively, using the command line:
1. compile both files:
   1. `gcc -Wall -c .\spkmeans.c`
   2. `gcc -Wall -c .\test.c`
2. link files:
   1. `gcc -o run spkmeans.o tests.o`
3. execute file:
   1. `./run.exe`


# python - run #
1. a
2. b
3. c
4. 