# Compile and Run #

For invoking method foo() defined in spkmeans.c in test.c:
1. set `#include "spkmeans.h"` in the top of test.c file
2. run the program using IDE

Alternatively, using the command line:
1. compile both files:
   1. `gcc -Wall -c .\spkmeans.c`
   2. `gcc -Wall -c .\spkmeans.c`
2. link files:
   1. `gcc -o  run spkmeans.o tests.o`
3. execute file:
   1. `./run.exe`

