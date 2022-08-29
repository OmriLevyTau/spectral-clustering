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
void test_spk_Memory();


int main(){

//    int test1 = test1_create_wam();
//    int test2 = test2_create_ddg();
//    int test3 = test3_create_l_norm();
//
//    printf("test_wam: %d", test1);
//    printf("test_ddg: %d", test2);
//    printf("test_lnorm: %d", test3);
//
//    test4_create_jacobi();

//    test5_create_U();

//    test_spk_Memory();

    test_Wam_Memory();
    return 0;

}

/*
int test1_create_wam(){

    int n_input = count_rows("C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\inputs\\input_8.txt");
    int d_input = count_cols("C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\inputs\\input_8.txt");
    double** X = read_data_from_file(n_input, d_input, "C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\inputs\\input_8.txt");

    printMatrix(X, n_input, d_input);
    printf("\n");

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
    int n_input = count_rows("C:\\Users\\idani\\SW_PROJ_FINAL\\test_files\\project_comprehensive_test\\testfiles\\jacobi_1.txt");
    int d_input = count_cols("C:\\Users\\idani\\SW_PROJ_FINAL\\test_files\\project_comprehensive_test\\testfiles\\jacobi_1.txt");
    double** L_norm = read_data_from_file(n_input, d_input, "C:\\Users\\idani\\SW_PROJ_FINAL\\test_files\\project_comprehensive_test\\testfiles\\jacobi_1.txt");


    double*** VandA = create_jacobi_matrix(n_input,L_norm);
//    print_2_matrices(n_input,VandA);
    printf("V: \n");
    printMatrix(VandA[0], n_input, n_input);

    printf("A: \n");
    printMatrix(VandA[1], n_input, n_input);
    printf("\n");
    int* indices = find_k_max_indices(n_input,3,VandA[1]);
    double** U = create_U(n_input,3,indices,VandA[0]);
    printMatrix(U,n_input,3);
    printf("\n");
    double** T = create_T(n_input,3,U);
    printMatrix(T,n_input,3);
}
/*
void test_spk_Memory(){

    char* input_file = "C:\\Users\\idani\\SW_PROJ_FINAL\\test_files\\project_comprehensive_test\\testfiles\\jacobi_1.txt";

//    char* input_file = "jacobi_1.txt";
    int k = 3;
    spk(k, input_file);
}
 */

