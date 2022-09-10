# Python and C #

## compile and run spkemans.c with user input ##

1. `gcc -Wall .\spkmeans.c -o run `
2. `.\run.exe jacobi C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\inputs\\sym_matrix\\sym_matrix_input_9.txt`
3. `.\run.exe lnorm C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\inputs\\data_points\\input_0.txt`


# python - run #
1. On Nova:
   1. `python3 spkmeans.py 3 spk input_1.txt`
2. On local machine:
   1. `python spkmeans.py 3 spk input_1.txt`

# Build and run all project #
1. `python3 setup.py build_ext --inplace`
   1. or: `python setup.py build_ext --inplace` locally
2. `comp.sh`
   1. or `gcc -ansi -Wall -Wextra -Werror -pedantic-errors spkmeans.c -lm -o spkmeans
      `
