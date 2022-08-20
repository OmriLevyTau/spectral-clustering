
#ifndef SPECTRAL_CLUSTERING_TESTS_H
#define SPECTRAL_CLUSTERING_TESTS_H

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
void run_tests();

#endif //SPECTRAL_CLUSTERING_TESTS_H
