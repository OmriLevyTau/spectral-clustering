#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "spkmeans.h"

int assert_array_are_equal(int n, double *expected, double *actual);
int assert_matrix_are_equal(int n, double **expected, double **actual);
int assert_equals(double expected, double actual);
int testA_sum_array(int n, double expected, double* arr);
int testB_l2_norm_dist(int n, double expected, const double *A, const double *B);
int testC_weighted_distance(int n, double expected, const double *A, const double *B);
int testD1_create_wam(int n, int d, double** expected, double** X);
int testD2_create_identity_matrix(int n, double** expected);
int testE_create_ddg(int n, int d, const double** expected, double** X);
int testF_create_ddg_inverse(int n, int d, const double** expected, double** X);
int testG_create_Lnorm(int n, int d, const double** expected, double** X);
int testH_extract_diagonal(int n, const double* expected, double** A);
int testI_find_eigengap(int n, int expected, double** A);
int testJ_sort(int n, double* expected, double* arr);
int testK_find_ij(int n, int* expected, double** A);
int testL_compute_off_diag(int n, double expected, const double **A);
int testM_reset_matrix(int n, double**A);
void printMatrix(double** mat, int rows, int cols);
double** build_calloc_matrix(int rows, int cols, double M[rows][cols]);
double* build_calloc_array(int n, double v[n]);
void print_vector(double* pointer, int cols);
int* build_calloc_int_array(int n, int v[n]);

int main(){
    /*
     * Generate Data
     * */
    int n =4;
    int d =3;
    int i,j;
    double arr1[] = {1,2,3,4};
    double arr2[] = {1,1,1,1};
    double arr3[] = {0.1,0.2,0.3,0.4};

    double X_temp[4][3] = {{1.0,1.0,1.0},
                           {2.0,2.0,2.0},
                           {1.5,1.5,1.5},
                           {0.1,0.2,0.3}};

    double eye_temp[4][4] = {{1.0, 0.0, 0.0, 0.0},
                        {0.0, 1.0, 0.0, 0.0},
                        {0.0, 0.0, 1.0, 0.0},
                        {0.0, 0.0, 0.0, 1.0}};

    double W_temp[4][4] = { {0.0, 1.73205081, 0.8660254, 1.39283883},
                            {1.73205081, 0.0, 0.8660254, 3.12089731},
                            {0.8660254, 0.8660254, 0.0, 2.25610283},
                            {1.39283883, 3.12089731, 2.25610283, 0.0}};

    double D_temp[4][4] = {{3.99091504, 0.0, 0.0, 0.0},
                           {0.0, 5.71897352, 0.0, 0.0},
                           {0.0, 0.0, 3.98815364, 0.0},
                           {0.0, 0.0, 0.0, 6.76983897}};

    double D_inv_temp[4][4] = {{0.50056878, 0.0, 0.0, 0.0},
                              {0.0, 0.41815853, 0.0, 0.0},
                              {0.0, 0.0, 0.50074205, 0.0},
                              {0.0, 0.0, 0.0, 0.38433579}};

    double L_norm_temp[4][4] = {{1.0, -0.36254786, -0.21707432, -0.26796338},
                               {-0.36254786, 1.0, -0.18133668, -0.50156967},
                               {-0.21707432, -0.18133668, 1.0, -0.43419396},
                               {-0.26796338, -0.50156967, -0.43419396, 1.0}};

    double gap1_temp[8][8] = {  {22.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
                                {0.0,21.1,0.0,0.0,0.0,0.0,0.0,0.0},
                                {0.0,0.0,15.6,0.0,0.0,0.0,0.0,0.0},
                                {0.0,0.0,0.0,15.0,0.0,0.0,0.0,0.0},
                                {0.0,0.0,0.0,0.0,14.0,0.0,0.0,0.0},
                                {0.0,0.0,0.0,0.0,0.0,13.0,0.0,0.0},
                                {0.0,0.0,0.0,0.0,0.0,0.0,12.0,0.0},
                                {0.0,0.0,0.0,0.0,0.0,0.0,0.0,11.0}};
//    {22.1, 21.1, 15.6, 15.0, 14.0, 13.0, 12.0, 11.0};

    double gap2_temp[8][8] = {  {22.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
                                {0.0,11.0,0.0,0.0,0.0,0.0,0.0,0.0},
                                {0.0,0.0,9.0,0.0,0.0,0.0,0.0,0.0},
                                {0.0,0.0,0.0,8.5,0.0,0.0,0.0,0.0},
                                {0.0,0.0,0.0,0.0,7.5,0.0,0.0,0.0},
                                {0.0,0.0,0.0,0.0,0.0,6.5,0.0,0.0},
                                {0.0,0.0,0.0,0.0,0.0,0.0,6.0,0.0},
                                {0.0,0.0,0.0,0.0,0.0,0.0,0.0,5.12}};

    double find_ij_temp[4][4] = {{1,2,3,4},
                                 {5,6,7,8},
                                 {9,10,11,12},
                                 {1,1,1,1}};
    int find_ij_real1_temp[2] = {2, 3};
    int find_ij_real2_temp[2] = {1, 2};

//    {22.1, 11.0, 9.0, 8.5, 7.5, 6.5, 6.00, 5.12};



    double sorted_temp[5] = {5,4,3,2,1};
    double sort1_temp[5] = {1,2,3,4,5};
    double sort2_temp[5] = {5,4,2,1,3};
    double diag1_temp[4] = {1.0,1.0,1.0,1.0};
    double diag2_temp[4] = {3.99091504,5.71897352,3.98815364,6.76983897};



    double** X = build_calloc_matrix(n,d,X_temp);
    double** eye = build_calloc_matrix(n,n,eye_temp);
    double** W = build_calloc_matrix(n,n,W_temp);
    double** D = build_calloc_matrix(n,n,D_temp);
    double** D_inv = build_calloc_matrix(n,n,D_inv_temp);
    double** L_norm = build_calloc_matrix(n,n,L_norm_temp);
    double* sort1 = build_calloc_array(5, sort1_temp);
    double* sort2 = build_calloc_array(5, sort2_temp);
    double* sorted = build_calloc_array(5, sorted_temp);
    double* diag1 = build_calloc_array(n, diag1_temp);
    double* diag2 = build_calloc_array(n, diag2_temp);
    double* gap1 = build_calloc_matrix(8,8, gap1_temp);
    double* gap2 = build_calloc_matrix(8,8, gap2_temp);
    double** find_ij = build_calloc_matrix(4,4, find_ij_temp);
    int* find_ij_real1 = build_calloc_int_array(2, find_ij_real1_temp);
    int* find_ij_real2 = build_calloc_int_array(2, find_ij_real2_temp);

    int testA = testA_sum_array(n, 10, arr1);
    int testB1 = testB_l2_norm_dist(n, 0.0 ,arr1, arr1);
    int testB2 = testB_l2_norm_dist(n, 1.51657508881031, arr2, arr3);
    int testC1 = testC_weighted_distance(n, 1.0, arr1, arr1);
    int testC2 = testC_weighted_distance(n, 0.15399599, arr1, arr2);
    int testD2 = testD2_create_identity_matrix(n, eye);
    int testD1 = testD1_create_wam(n, d, W, X);
    int testE = testE_create_ddg(n, d, D, X);
    int testF = testF_create_ddg_inverse(n, d, D_inv, X);
    int testG = testG_create_Lnorm(n, d, L_norm, X);
    int testH1 = testH_extract_diagonal(n, diag1, eye);
    int testH2 = testH_extract_diagonal(n, diag2, D);
    int testI1 = testI_find_eigengap(8, 1, gap1);
    int testI2 = testI_find_eigengap(8, 0, gap2);
    int testJ1 = testJ_sort(5, sorted, sort1);
    int testJ2 = testJ_sort(5, sorted, sort2);
    int testK1 = testK_find_ij(n, find_ij_real1, find_ij);
    int testK2 = testK_find_ij(n, find_ij_real2, L_norm);
    int testL1 = testL_compute_off_diag(n, 0.0, eye);
    int testL2 = testL_compute_off_diag(n, 42.54, W);


    printf("A:  %d \n", testA);
    printf("B1: %d \n", testB1);
    printf("B2: %d \n", testB2);
    printf("C1: %d \n", testC1);
    printf("C2: %d \n", testC2);
    printf("D2: %d \n", testD2);
    printf("D1: %d \n", testD1);
    printf("E:  %d \n",  testE);
    printf("F:  %d \n",  testF);
    printf("G:  %d \n",  testG);
    printf("H1: %d \n",  testH1);
    printf("H2: %d \n",  testH2);
    printf("I1: %d \n",  testI1);
    printf("I2: %d \n",  testI2);
    printf("J1: %d \n",  testJ1);
    printf("J2: %d \n",  testJ2);
    printf("K1: %d \n",  testK1);
    printf("K2: %d \n",  testK2);
    printf("L1: %d \n",  testL1);
    printf("L2: %d \n",  testL2);


    free_matrix(n, X);
    free_matrix(n, eye);
    free_matrix(n, W);
    free_matrix(n, D);
    free_matrix(n, D_inv);
    free_matrix(n, L_norm);
    free(sort1);
    free(sort2);
    free(sorted);
    free(diag1);
    free(diag2);
    free_matrix(8, gap1);
    free_matrix(8, gap2);
    free_matrix(4, find_ij);
    free(find_ij_real1);
    free(find_ij_real2);


    return 0;

}

/*
 * Helpers methods
 * */

int assert_equals(double expected, double actual){
    if ((fabs(expected-actual))<0.00001){
        return 1;
    }
    return 0;
}

int assert_array_are_equal(int n, double *expected, double *actual){
    int i;
    for (i=0; i<n; i++){
        if (assert_equals(expected[i],actual[i])==0) { return 0 ;}
    }
    return 1;
}

int assert_matrix_are_equal(int n, double **expected, double **actual){
    int i, j;
    for (i=0; i<n; i++){
        for(j=0; j<n; j++){
            if(assert_equals(expected[i][j],actual[i][j])==0) {return 0 ;}
        }
    }
    return 1;
}

void print_vector(double* pointer, int cols){
    int i;
    for (i=0; i<cols;i++){
        printf("  %.4f",pointer[i]);
    }
}

void printMatrix(double** mat, int rows, int cols){
    int i,j;
    for (i=0; i<rows;i++){
        for (j=0;j<cols;j++){
            printf("  %.4f",mat[i][j]);
        }
        printf("\n");
    }
}

double* build_calloc_array(int n, double v[n]){
    int i;
    double* arr = calloc(n, sizeof(double));

    for (i=0; i<n; i++){
        arr[i]=v[i];
    }
    return arr;
}

int* build_calloc_int_array(int n, int v[n]){
    int i;
    int* arr = calloc(n, sizeof(int));

    for (i=0; i<n; i++){
        arr[i]=v[i];
    }
    return arr;
}

double** build_calloc_matrix(int rows, int cols, double M[rows][cols]){
    int i,j;
    double** X = calloc(rows, sizeof(int*));

    for (i=0; i<rows; i++){
        X[i] = calloc(cols, sizeof(double));
    }

    for (i=0; i<rows; i++){
        for (j=0; j<cols; j++){
            X[i][j] = M[i][j];
        }
    }
    return X;
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

int testD1_create_wam(int n, int d, double** expected, double** X){
    double** wam = create_wam(n,d, X);
//    printMatrix(wam, n, n);
    int result = assert_matrix_are_equal(n, expected, wam);
    free_matrix(n, wam);
    return result;
}

int testD2_create_identity_matrix(int n, double** expected){
    double** eye = create_identity_matrix(n);
    int result = assert_matrix_are_equal(n, expected, eye);
    free_matrix(n, eye);
    return result;
}

int testE_create_ddg(int n, int d, const double** expected, double** X){
    double** ddg = create_ddg(n, d, X);
    int result = assert_matrix_are_equal(n, expected, ddg);
    free_matrix(n, ddg);
    return result;
}

int testF_create_ddg_inverse(int n, int d, const double** expected, double** X){
    double** ddg_inv = create_ddg_inverse(n, d, X);
    int result = assert_matrix_are_equal(n, expected, ddg_inv);
    free_matrix(n, ddg_inv);
    return result;
}


int testG_create_Lnorm(int n, int d, const double** expected, double** X){
    double** l_norm = create_Lnorm(n, d, X);
    int result = assert_matrix_are_equal(n, expected, l_norm);
//    printMatrix(l_norm, n, n);
    free_matrix(n, l_norm);
    return result;
}

int testH_extract_diagonal(int n, const double* expected, double** A){
    double* diag = extract_diagonal(n, A);
//    print_vector(diag, n); printf("\n");
    int result = assert_array_are_equal(n, expected, diag);
    free(diag);
    return result;
}

int testI_find_eigengap(int n, int expected, double** A){
    int gap = find_eigengap(n, A);
    return assert_equals(expected, gap);
}

int testJ_sort(int n, double* expected, double* arr){
//    print_vector(arr, n); printf("\n");
    qsort(arr, n, sizeof(double), compare_reversed_order);
//    print_vector(arr, n); printf("\n");
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
//    printf("%f \n", off);
    return assert_equals(expected, off);
}

int testM_reset_matrix(int n, double**A){
    double** eye = create_identity_matrix(n);
    reset_matrix(n, A);
    int result = assert_matrix_are_equal(n, eye, A);
    free_matrix(n, eye);
    return result;
}



