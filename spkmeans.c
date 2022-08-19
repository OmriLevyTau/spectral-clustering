#include <math.h>
#include <stdlib.h>
#include <stdio.h>

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



/*
 * L2 Norm distance of the vector A - B
 * */
double l2_norm_dist(int n, const double *A, const double *B){
    double sum = 0;
    for(int i=0; i<n; i++){
        sum += (A[i]-B[i])*(A[i]-B[i]);
    }
    return math.sqrt(sum);
}

/*
 * calculates the weighted distance between points A and B
 * */
double weighted_distance(int n, const double *A, const double *B){
    double diff = l2_norm_dist(n,A,B);
    return math.exp(-0.5 * diff);
}

/*
 * creates the Weighted Adjacency Matrix
 1. build matrix nxn
 2. nested for loop - for every cell in A - compute lw_norm_dist
 * */
double** create_wam(int n, double** X){
    int i, j;
    double ** W = buildMatrix(n,n);

    for(i=0; i<n; i++){
        for(j=0; j<n; j++){
            if(i==j){
                W[i][j] = 0.0;
            }
            else{
                W[i][j] = l2_norm_dist(n,X[i], X[j]);
            }
        }
    }
    return W;
}

/*
 * creates identity matrix
 * */
double** create_identity_matrix(int n){
    double** eye = buildMatrix(n,n);
    for(int i=0; i<n; i++){
        eye[i][i] = 1.0;
    }
    return eye;
}

/*
 * creates Diagonal Degree Matrix
 * uses sum_array
 * */
double** create_ddg(int n, double** X){
    double** W = create_wam(n,X);
    double** D = create_identity_matrix(n);
    for(int i=0; i<n; i++){
        D[i][i] = sum_array(n, W[i]);
    }
    free_matrix(n,W);
    return D;
}

/*
 * creates D^(-0.5)
 * */
double** create_ddg_inverse(int n, double** X){
    double** D = create_ddg(n, X);
    double** D_inverse = create_identity_matrix(n);
    for(int i=0; i<n; i++){
        D_inverse[i][i] = 1 / math.sqrt(D[i][i]);
    }
    free_matrix(n,D);
    return D_inverse;
}

/*
 * creates Normalized Graph Laplacian
 * */
double** create_Lnorm(int n, double** X){
    double** D = create_ddg(n,X);
    double** D_inverse = create_ddg_inverse(n, D);
    double** W = create_wam(n,X);
    double** temp = matrix_mult(W,D_inverse);
    double** L_norm = matrix_mult(D_inverse, temp);
    for(int i=0; i<n; i++){
        L_norm[i][i] = 1-L_norm[i][i];
    }
    free_matrix(D);
    free_matrix(D_inverse);
    free_matrix(W);
    free_matrix(temp);
    return L_norm;
}

double* extract_diagonal(int n, double** A){
    int i;
    double* eigenvalues = calloc(n, sizeof(double));
    for(i=0; i<n; i++){
        eigenvalues[i] = A[i][i];
    }
    return eigenvalues;
}


/*
 * finds eigengap heuristic
 * */
int find_eigengap(int n, double** A){
    int limit = floor(n/2);
    double max_val = 0;
    int k = 0;
    double* eigenvals = extract_diagonal(n, A);
    qsort(eigenvals,n, sizeof(double ), compare_int_reversed_order); /*inplace */
    for(int i=0; i<limit-1; i++){
        if(abs(eigenvals[i]-eigenvals(i+1))>max_val){
            max_val = abs(eigenvals[i]-eigenvals(i+1));
            k = i;
        }
    }
    free(eigenvals);
    return k;
}


int compare_int_reversed_order(const void *p1, const void *p2){
    /* returns positive if a<b, 0 if a==b, negative if a>b*/
    const int *q1 = p1, *q2 = p2;
    return ((*q2)-(*q1));
}


/*
 * finds the indices i,j
 * under assumption that K>1
 * */
int* find_ij (int n, double** A){
    int a, b;
    int[2] indices = {0,1};
    double cur_max = math.abs(A[0][1]);
    for(a=0; a<n; a++){
        for(b=0; b<n; b++){
            if(a!=b){
                if(math.abs(A[a][b]) > cur_max){
                    cur_max = math.abs(A[a][b]);
                    indices[0] = a;
                    indices[1] = b;
                }
            }
        }
    }
    return indices;
}


/*
 * uses find_ij and create P_m
1. compute t, s, c, theta (calls helpers)
2. reset temp (I) (reset_matrix)
3. update P_prev and return it
 * */
double** create_P (int n, int i, int j, double** A, double** P_prev){
    double theta = compute_theta(A[i][j], A[i][i], A[j][j]);
    double t = compute_t(theta);
    double c = compute_c(t);
    double s = t * c;
    reset_matrix(n, P_prev);
    P_prev[i][i] = c;
    P_prev[j][j] = c;
    P_prev[i][j] = s;
    P_prev[j][i] = s * -1;
    return P_prev;
}


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
    1. compute i,j
    2. P_m = create_P (updated P_zona)
    3. compute off(A)
    4. set A=A' (call update_A_to_A_tag)
    6. V =V*P_m (matrix_mult)
    7. if(off(A)^2-off(A')^2<=eps){ break;}
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

void print_error_and_exit(){
    printf("An Error Has Occurred");
    exit(1);
}

/*
build new matrix 
*/
double** buildMatrix(int rows, int cols){
    /*
     * Creates empty matrix of size rows x cols
     */
    int i;
    double **a = calloc(rows, sizeof(int*));
    if (a==NULL){
        print_error_and_exit();
    }
    for (i=0;i<rows;i++){
        a[i] = calloc(cols, sizeof(double));
        if (a[i]==NULL){
            print_error_and_exit();
        }
    }
    return a;
}



/*
 * sum of array
 * */
double sum_array(int n, double* arr){
    double sum = 0;
    for(int i=0; i<n; i++){
        sum += arr[i];
    }
    return sum;
}

/*
 * Matrix multiplication
 * */
double** matrix_mult(int n, const double **A, const double **B){
    double sum;
    int a, b, c;
    double** new_mat = buildMatrix(n,n);
    for(a=0; a<n; a++){
        for(b=0; b<n; b++){
            sum = 0.0;
            for(c=0; c<n; c++){
                sum += A[a][c] * B[c][b];
            }
            new_mat[a][b] = sum;
        }
    }
    return new_mat;
}

/*
 * free from memory
 */
int free_matrix( int rows, double** pointer,){
    int i;
    for (i = 0; i<rows; i++){
        if (pointer[i]!=NULL){
            free(pointer[i]);
        } else {
            return 1;
        }
    }
    free(pointer);
    return 0;
}

/*
 * computes the sum of all the off-diagonal cells in A
 * */
double compute_off_diag(int n, const double **A){
    int i,j;
    double sum =0;
    for(i=0; i<n; i++){
        for(j=0; j<n; j++){
            if(i!=j){
                sum += A[i][j] * A[i][j];
            }
        }
    }
    return sum;
}

/*
 * computes theta
 * */
double compute_theta(double A_ij, double A_ii, double A_jj){
    return (A_jj - A_ii) / (2 * A_ij);
}

/*
 * computes t
 * */
double compute_t(double theta){
    double sign = theta>=0? 1.0: -1.0;
    double denom = math.abs(theta) + math.sqrt((theta * theta) + 1 );
    return sign / denom;
}

/*
 * computes c
 * */
double compute_c(double t){
    return 1 / (math.sqrt((t*t)+1));
}

/*
 * recieves a matrix and returns it as identity
 * */
void reset_matrix(int n, double**A){
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            if(i==j){
                A[i][j]=1.0;
            }
            else{
                A[i][j]=0.0;
            }
        }
    }
}

/*
 * updates A by formula (inplace)
  for testing - A' is always symmetric
 * */
void update_A_to_A_tag(int n, int i, int j, double**A){
    double theta = compute_theta(A[i][j], A[i][i], A[j][j]);
    double t = compute_t(theta);
    double c = compute_c(t);
    double s = t * c;
    int l=0;
    /* run i'th row */
    for(l=0; l<n; l++){
        A[i][l] = (c * A[i][l]) - (s * A[j][l]);
    }

    /* run j'th row */
    for(l=0; l<n; l++){
        A[j][l] = (s * A[i][l]) + (c * A[j][l]);
    }

    /* run i'th column */
    for(l=0; l<n; l++){
        A[l][i] = (c * A[i][l]) - (s * A[j][l]);
    }

    /* run j'th row */
    for(l=0; l<n; l++){
        A[l][j] = (s * A[i][l]) + (c * A[j][l]);
    }

    /* update 4 special cases*/
    A[i][i] = (c * c * A[i][i]) - (2 * s * c * A[i][j]) + (s * s * A[j][j]);
    A[j][j] = (s * s * A[i][i]) + (2 * s * c * A[i][j]) + (c * c *A[j][j]);
    A[i][j] = (((c * c) - (s * s)) * A[i][j]) + (s * c * (A[i][i] - A[j][j]));
    A[j][i] = (((c * c) - (s * s)) * A[i][j]) + (s * c * (A[i][i] - A[j][j]));

    return A;
}

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


/*
 * reads data from txt file into a matrix
 * */
double** read_data(int rows, int cols, char* filePath);
int count_cols(char* filePath);
int count_rows(char* filePath);
