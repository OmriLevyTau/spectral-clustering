double l2_norm_dist(int n, const double *A, const double *B);
double weighted_distance(int n, const double *A, const double *B);
double** create_wam(int n, double** X);
double** create_identity_matrix(int n);
double** create_ddg(int n, double** X);
double** create_ddg_inverse(int n, double** X);
double** create_Lnorm(int n, double** X);
double* extract_diagonal(int n, double** A);
int find_eigengap(int n, double** A);
int compare_int_reversed_order(const void *p1, const void *p2);
int* find_ij (int n, double** A);
double** create_P (int n, int i, int j, double** A, double** P_prev);
double*** create_jacobi_matrix (int n, double** L_norm);
double** jacobi_algorithm (int k, int n, double** L_norm);
double** create_T(int n, double** U);
void print_error_and_exit();
double** buildMatrix(int rows, int cols);
double sum_array(int n, double* arr);
double** matrix_mult(int n, const double **A, const double **B);
int free_matrix( int rows, double** pointer,);
double compute_off_diag(int n, const double **A);
double compute_theta(double A_ij, double A_ii, double A_jj);
double compute_t(double theta);
double compute_c(double t);
void reset_matrix(int n, double**A);
void update_A_to_A_tag(int n, int i, int j, double**A);
int* find_k_max_indices(int n, int k, double** A);
double** create_U (int n, int k, int* indices, double** V);
double** read_data(int rows, int cols, char* filePath);
int count_cols(char* filePath);
int count_rows(char* filePath);
