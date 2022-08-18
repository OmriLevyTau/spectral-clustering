

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
 * */
double** create_ddg(int n, double** W);

/*
 * creates Diagonal Degree Matrix
 * uses sum_array
 * */
double** create_ddg(int n, double** W);

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
double** find_eigengap(int n, double* eigenvals);







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


