#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "spkmeans.h"
#include "tests.h"
#include <string.h>

int test1_create_wam();
int test2_create_ddg();
int test3_create_l_norm();


int main(){

//    int test1 = test1_create_wam();
//    int test2 = test2_create_ddg();
    int test3 = test3_create_l_norm();


//    printf("test_wam: %d", test1);
//    printf("test_ddg: %d", test2);
    printf("test_lnorm: %d", test3);
    return 0;

}

int test1_create_wam(){

    int n_input = count_rows("C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\inputs\\input_8.txt");
    int d_input = count_cols("C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\inputs\\input_8.txt");
    double** X = read_data_from_file(n_input, d_input, "C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\inputs\\input_8.txt");

//    printMatrix(X, n_input, d_input);
//    printf("\n");

    int n_output = count_rows("C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\outputs\\wam\\wam_c_input_8.txt");
    int d_output = count_cols("C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\outputs\\wam\\wam_c_input_8.txt");
    double** expected = read_data_from_file(n_output, d_output, "C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\outputs\\wam\\wam_c_input_8.txt");

//    printMatrix(expected, n_output, n_output);
//    printf("\n");

    int result = testD1_create_wam(n_input, d_input, expected, X);

    return result;

}

int test2_create_ddg(){

    int n_input = count_rows("C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\inputs\\input_8.txt");
    int d_input = count_cols("C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\inputs\\input_8.txt");
    double** X = read_data_from_file(n_input, d_input, "C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\inputs\\input_8.txt");

//    printMatrix(X, n_input, d_input);
//    printf("\n");

    int n_output = count_rows("C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\outputs\\ddg\\ddg_c_input_8.txt");
    int d_output = count_cols("C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\outputs\\ddg\\ddg_c_input_8.txt");
    double** expected = read_data_from_file(n_output, d_output, "C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\outputs\\ddg\\ddg_c_input_8.txt");

//    printMatrix(expected, n_output, n_output);
//    printf("\n");

    int result = testE_create_ddg(n_input, d_input, expected, X);

    return result;

}

int test3_create_l_norm(){

    int n_input = count_rows("C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\inputs\\input_13.txt");
    int d_input = count_cols("C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\inputs\\input_13.txt");
    double** X = read_data_from_file(n_input, d_input, "C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\inputs\\input_13.txt");

    printMatrix(X, n_input, d_input);
    printf("\n");

    int n_output = count_rows("C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\outputs\\lnorm\\lnorm_c_input_13.txt");
    int d_output = count_cols("C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\outputs\\lnorm\\lnorm_c_input_13.txt");
    double** expected = read_data_from_file(n_output, d_output, "C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\outputs\\lnorm\\lnorm_c_input_13.txt");

    printMatrix(expected, n_output, n_output);
    printf("\n");

    int result = testG_create_Lnorm(n_input, d_input, expected, X);

    return result;

}