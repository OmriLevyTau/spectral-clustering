

/*
 * L2 Norm
 * */
double l2_norm(int n, const double *A, const double *B);

/*
 * calculates the weighted distance between points A and B
 * */
double weighted_distance(int n, const double *A, const double *B);

/*
 * creates the Weighted Adjacency Matrix
 * */
double** create_wam(int n, double** X);

/*
 * creates identity matrix
 * */
double** create_identity_matrix(int n);

/*
 * creates Diagonal Degree Matrix
 * uses sum_array
 * */
double** create_ddg(int n, double** W);

/*
 * creates D^(-0.5)
 * */
double** create_ddg_inverse(double** D);

/*
 * creates Normalized Graph Laplacian
 * */
double** create_lnorm(double** D);

/*
 * finds eigengap heuristic
 * */
int find_eigengap(int n, double* eigenvals);



/*
 * finds the indices i,j
 * */
int* find_ij (int n, double** A);  


/*
 * uses find_ij and create P_m
 * */
double** create_P (int n, int i, int j, double** A, double** temp);
/*
1. 
2. compute t, s, c, theta (calls helpers)
3. reset temp (I) (reset_matrix)
4. update matrix temp and return it
*/


/*
 * creates V (eigenvectors as columns) and A' (diagonal, eigenvalues of original A) 
 * */
 double*** create_jacobi_matrix (int n, double** L_norm);

 /*
 * 
 V = I (create_identity);
 A = copy of L_norm
 initialize temp(create_identity);
 for(int m=0; m<100; m++){
    1. if(off(A)^2-off(A')^2<=eps){ break;}
    2. compute i,j
    3. P_m = create_P (updated P_zona)
    4. set A=A' (call compute_A_tag) 
    6. V =V*P_m (matrix_mult)
 }
now V holds eigenvectors and A holds eigenvalues
free(temp);
return V and A
 */


/*
 * run jacobi algorithm
 * */
double** jacobi_algorithm (int k, int n, double** L_norm);
/*
 * 1. call create_jacobi_matrix and compute V and A
 * 2. if k=0, calls find_eigengap and compute that required k
 * 3. indices= find_k_max_indices 
 * 4. create and return U (call create_U)
 * */


/*
 * normalizes U and returns it
 * */
double** create_T(int n, double** U);



/*
 ***** HELPERS METHODS ***
 * */

/*
 * reads data from txt file into a matrix
 * */
double** read_data(int rows, int cols, char* filePath);
int count_cols(char* filePath);
int count_rows(char* filePath);

/*
 * sum of array
 * */
double sum_array(double* arr);

/*
 * Matrix multiplication
 * */
double** matrix_mult(int n, const double **A, const double **B);

/*
 * creates a copy of the matrix A
 * */
double** matrix_copy(int n, const double **A);

/*
 * computes the sum of all the off-diagonal cells in A
 * */
double compute_off_diag(int n, const double **A);

/*
 * computes theta
 * */
double compute_theta(double A_ij, double A_ii, double A_jj);

/*
 * computes t
 * */
double compute_t(double theta);

/*
 * computes c
 * */
double compute_c(double t);

/*
 * recieves a matrix and returns it as identity
 * */
double** reset_matrix(int n, double**A);

/*
 * updates A by formula
  for testing - A' is always symmetric
 * */
double** compute_A_tag(int n, int i, int j, double c, double s, double**A);

/*
 * find_k_max_indices
 * */
int* find_k_max_indices(int n, int k, double** A);
/*
K times find the next maximum from eigenvalues in A, and returns those indices 
*/

/*
 * creates U from V bu largest K eigenvalues
 * */
double** create_U (int n, int k, int* indices, double** V);



