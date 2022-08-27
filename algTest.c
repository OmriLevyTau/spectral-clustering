#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "spkmeans.h"
#include "tests.h"
#include <string.h>

int test1_create_wam();
int test2_create_ddg();
int test3_create_l_norm();
void test4_create_jacobi();
void test5_create_U();


int main(){

//    int test1 = test1_create_wam();
//    int test2 = test2_create_ddg();
//    int test3 = test3_create_l_norm();


//    printf("test_wam: %d", test1);
//    printf("test_ddg: %d", test2);
//    printf("test_lnorm: %d", test3);

//    test4_create_jacobi();

    test5_create_U();


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

    int n_input = count_rows("C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\inputs\\input_0.txt");
    int d_input = count_cols("C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\inputs\\input_0.txt");
    double** X = read_data_from_file(n_input, d_input, "C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\inputs\\input_0.txt");

    printMatrix(X, n_input, d_input);
    printf("\n");

    int n_output = count_rows("C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\outputs\\lnorm\\lnorm_c_input_0.txt");
    int d_output = count_cols("C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\outputs\\lnorm\\lnorm_c_input_0.txt");
    double** expected = read_data_from_file(n_output, d_output, "C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\outputs\\lnorm\\lnorm_c_input_0.txt");

    printMatrix(expected, n_output, n_output);
    printf("\n");

    int result = testG_create_Lnorm(n_input, d_input, expected, X);

    return result;

}

void test4_create_jacobi(){

    int n_input = count_rows("C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\inputs\\sym_matrix\\sym_matrix_input_11.txt");
    int d_input = count_cols("C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\inputs\\sym_matrix\\sym_matrix_input_11.txt");
    double** L_norm = read_data_from_file(n_input, n_input, "C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\inputs\\sym_matrix\\sym_matrix_input_11.txt");

    printf("L_norm: \n");
    printMatrix(L_norm, n_input, n_input);
    printf("\n");

    double*** VandA = create_jacobi_matrix(n_input, L_norm);
    double* eig = extract_diagonal(n_input, VandA[1]);
    print_2_matrices(n_input, VandA);
    printf("\n");
    print_double_vector(eig, n_input);
}


void test5_create_U(){
    int n_input = count_rows("C:\\Users\\idani\\SW_PROJ_FINAL\\test_files\\inputs\\sym_matrix\\sym_matrix_input_9.txt");
    int d_input = count_cols("C:\\Users\\idani\\SW_PROJ_FINAL\\test_files\\inputs\\sym_matrix\\sym_matrix_input_9.txt");
    double** X = read_data_from_file(n_input, d_input, "C:\\Users\\idani\\SW_PROJ_FINAL\\test_files\\inputs\\sym_matrix\\sym_matrix_input_9.txt");

    printf("X: \n");
    printMatrix(X, n_input, d_input);
    printf("\n");

    printf("\n");
    double** V = create_jacobi_matrix(n_input, X)[0];
    printf("V: \n");
    printMatrix(V, n_input, d_input);
    printf("\n");
    double** A = create_jacobi_matrix(n_input, X)[1];
    printf("A: \n");
    printMatrix(A, n_input, d_input);
//    double** T = create_T(n_input, 4, U);
//    printf("T: \n");
//    printMatrix(T, n_input, 4);


}