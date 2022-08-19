#include <stdio.h>
#include <stdlib.h>
#include "spkmeans.h"

int assert_array_are_equal(int n, const double *expected, const double *actual);
int assert_matrix_are_equal(int n, const double **expected, const double **actual);
int assert_equals(double expected, double actual);
int testA_sum_array(int n, double expected, double* arr);




int main(){
    /*
     * Generate Data
     * */
    int n =4;
    double arr1[] = {1,2,3,4};
    double arr2[] = {1,1,1,1};
    double arr3[] = {2,2,2,2};

    double X[4][3] = {{1.0,1.0,1.0},
                      {2.0,2.0,2.0},
                      {1.5,1.5,1.5},
                      {0.1,0.2,0.3}};

    int testA = testA_sum_array(n, 10, arr1);

    printf("testA: %d", testA);

    return 0;

}

/*
 * Helpers methods
 * */

int assert_equals(double expected, double actual){
    return (expected==actual);
}

int assert_array_are_equal(int n, const double *expected, const double *actual){
    int i;
    for (i=0; i<n; i++){
        if (expected[i] != actual[i]) { return 0 ;}
    }
    return 1;
}

int assert_matrix_are_equal(int n, const double **expected, const double **actual){
    int i, j;
    for (i=0; i<n; i++){
        for(j=0; j<n; j++){
            if(expected[i][j] != actual[i][j]) {return 0 ;}
        }
    }
    return 1;
}

/*
 * Test Methods
 * ------------
 * Best practice is tests are self-contained and do not get elements from the outside.
 * In order to relief the data generations and allow flexibility, it will get its inputs
 * from the outside.
 * */

int testA_sum_array(int n, double expected, double* arr){
    int sum = sum_array(n, arr);
    return assert_equals(expected, sum);
}

int testB_l2_norm_dist(int n, double expected, const double *A, const double *B){
    double norm = l2_norm_dist(n, A, B);
    return assert_equals(expected, norm);
}

int testC_weighted_distance(int n, double expected, const double *A, const double *B){
    double weight = weighted_distance(n, A, B);
    return assert_equals(expected, weight);
}

int testD_create_wam(int n, double** expected, double** X){
    double** wam = create_wam(n, X);
    int result = assert_matrix_are_equal(n, expected, wam);
    free_matrix(n, wam);
    return result;
}

int testD_create_identity_matrix(int n, const double** expected){
    double** eye = create_identity_matrix(n);
    int result = assert_matrix_are_equal(n, expected, eye);
    free_matrix(n, eye);
    return result;
}

int testE_create_ddg(int n, const double** expected, double** X){
    double** ddg = create_ddg(n, X);
    int result = assert_matrix_are_equal(n, expected, ddg);
    free_matrix(n, ddg);
    return result;
}

int testF_create_ddg_inverse(int n, const double** expected, double** X){
    double** ddg_inv = create_ddg_inverse(n, X);
    int result = assert_matrix_are_equal(n, expected, ddg_inv);
    free_matrix(n, ddg_inv);
    return result;
}


int testG_create_Lnorm(int n, const double** expected, double** X){
    double** l_norm = create_Lnorm(n, X);
    int result = assert_matrix_are_equal(n, expected, l_norm);
    free_matrix(n, l_norm);
    return result;
}

int testH_extract_diagonal(int n, const double* expected, double** A){
    double* diag = extract_diagonal(n, A);
    int result = assert_array_are_equal(n, expected, diag);
    free(diag);
    return result;
}

int testI_find_eigengap(int n, int expected, double** A){
    int gap = find_eigengap(n, A);
    return assert_equals(expected, gap);
}

int testJ_sort(int n, double* expected, double* arr){
    qsort(arr, n, sizeof(double), compare_int_reversed_order);
    return assert_array_are_equal(n, expected, arr);
}

int testK_find_ij(int n, int* expected, double** A){
    int* indices = find_ij(n, A);
    int result = assert_array_are_equal(n, expected, indices);
    free(indices);
    return result;
}

int testL_compute_off_diag(int n, double expected, const double **A){
    double off = compute_off_diag(n, A);
    return assert_equals(expected, off);
}

int testM_reset_matrix(int n, double**A){
    double** eye = create_identity_matrix(n);
    reset_matrix(n, A);
    int result = assert_matrix_are_equal(n, eye, A);
    free_matrix(n, eye);
    return result;
}















