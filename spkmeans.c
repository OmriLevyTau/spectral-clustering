#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>


double l2_norm_dist(int n, double *A, double *B);
double weighted_distance(int n,  double *A, double *B);
double** create_wam(int n, int d, double** X);
double** create_identity_matrix(int n);
double** create_ddg(int n, int d, double** X);
double** create_ddg_inverse(int n, int d, double** X);
double** create_Lnorm(int n, int d, double** X);
double* extract_diagonal(int n, double** A);
int find_eigengap(int n, double** A);
int compare_reversed_order(const void *a, const void *b);
int* find_ij (int n, double** A);
void update_P (int n, int i, int j, double** A, double** P_prev);
double*** create_jacobi_matrix (int n, double** L_norm);
double** jacobi_algorithm (int k, int n, int d, double** X);
double** create_T(int rows, int cols, double** U);
void print_error_and_exit();
double** buildMatrix(int rows, int cols);
double sum_array(int n, double* arr);
double** matrix_mult(int n, double **A, double **B);
int free_matrix( int rows, double** pointer);
double compute_off_diag(int n, double **A);
double compute_theta(double A_ij, double A_ii, double A_jj);
double compute_t(double theta);
double compute_c(double t);
void reset_matrix(int n, double**A);
double** create_copy(int n, double** A);
void update_A_to_A_tag(int n, int i, int j, double**A_temp);
int* find_k_max_indices(int n, int k, double** A);
double** create_U (int n, int k, int* indices, double** V);
void spk_helper(int k, int n, int d, double** X, char* input_file);
void spk(int k, char* input_file);
double** read_data_from_file(int rows, int cols, char* filePath);
int count_cols(char* filePath);
int count_rows(char* filePath);
double sign(double x);
FILE* write_output(char* output_filename, int rows, int cols,double** Matrix);



/* build new matrix */
double** buildMatrix(int rows, int cols){
    /* Creates empty matrix of size rows x cols */
    int i;
    double **a = calloc(rows, sizeof(double*));
    if (a==NULL){
        print_error_and_exit();
    }
    for (i=0;i<rows;i++){
        a[i] = calloc(cols, sizeof(double)+8);
        if (a[i]==NULL){
            print_error_and_exit();
        }
    }
    return a;
}


/* L2 Norm distance of the vector A - B */
double l2_norm_dist(int n, double *A, double *B){
    int i;
    double sum = 0;
    for(i=0; i<n; i++){

        sum = sum + ((A[i]-B[i])*(A[i]-B[i]));
    }
    return sqrt(sum);
}

/* calculates the weighted distance between points A and B */
double weighted_distance(int n,  double *A,  double *B){
    double diff = l2_norm_dist(n,A,B);
    return exp(-0.5 * diff);
}

/*
 * creates the Weighted Adjacency Matrix
 1. build matrix nxn
 2. nested for loop - for every cell in A - compute lw_norm_dist
 * */
double** create_wam(int n, int d, double** X){
    int i, j;
    double** W = buildMatrix(n,n);
    for(i=0; i<n; i++){
        for(j=0; j<n; j++){
            if(i==j){
                W[i][j] = 0.0;
            }
            else{
                W[i][j] = weighted_distance(d, X[i], X[j]);
            }
        }
    }
    return W;
}

/* creates identity matrix */
double** create_identity_matrix(int n){
    int i;
    double** eye = buildMatrix(n,n);
    for(i=0; i<n; i++){
        eye[i][i] = 1.0;
    }
    return eye;
}

/* creates Diagonal Degree Matrix, uses sum_array */
double** create_ddg(int n, int d, double** X){
    int i;
    double** W = create_wam(n,d,X);
    double** D = create_identity_matrix(n);
    for(i=0; i<n; i++){
        D[i][i] = sum_array(n, W[i]);
    }
    free_matrix(n,W);
    return D;
}

/* creates D^(-0.5) */
double** create_ddg_inverse(int n, int d, double** X){
    int i;
    double** D = create_ddg(n,d, X);
    double** D_inverse = create_identity_matrix(n);
    for(i=0; i<n; i++){
        D_inverse[i][i] = 1.0 / sqrt(D[i][i]);
    }
    free_matrix(n,D);
    return D_inverse;
}

/* creates Normalized Graph Laplacian */
double** create_Lnorm(int n,int d, double** X){
    int i,j;
    double** D = create_ddg(n,d,X);
    double** D_inverse = create_ddg_inverse(n, d, X);
    double** W = create_wam(n,d,X);
    double** temp = matrix_mult(n, W,D_inverse);
    double** L_norm = matrix_mult(n, D_inverse, temp);
    for(i=0; i<n; i++){
        for (j=0; j<n; j++){
            if (i==j){
                L_norm[i][j] = 1-L_norm[i][j];
            } else {
                L_norm[i][j] = (-1)*L_norm[i][j];
            }
        }
    }
    free_matrix(n,D);
    free_matrix(n, D_inverse);
    free_matrix(n, W);
    free_matrix(n, temp);
    return L_norm;
}

/* gets matrix with eigenvalues on its diagonal, and returns an array with those eigenvalues*/
double* extract_diagonal(int n, double** A){
    int i;
    double* eigenvalues = calloc(n, sizeof(double));
    for(i=0; i<n; i++){
        eigenvalues[i] = A[i][i];
    }
    return eigenvalues;
}


/* finds eigengap heuristic */
int find_eigengap(int n, double** A){
    int i;
    int k = 0;
    double max_val = 0;
    double temp;
    int limit = (int) floor(n/2);
    double* eigenvals = extract_diagonal(n, A);
    qsort(eigenvals,n, sizeof(double), compare_reversed_order); /*inplace */
    for(i=0; i<limit; i++){
        temp = eigenvals[i]-eigenvals[i+1];
        if(sign(temp)*temp>max_val){
            max_val = temp*sign(temp);
            k = i;
        }
    }
    free(eigenvals);
    return (k+1);
}

/* helper for finding eigengap */
int compare_reversed_order(const void *a, const void *b){
    /* returns positive if a<b, 0 if a==b, negative if a>b*/
    double double_a = *((double*)a);
    double double_b = *((double*)b);
    if (double_a==double_b) return 0;
    else if (double_a<double_b) return 1;
    else return -1;
}


/* finds the indices i,j , under assumption that K>1 */
int* find_ij (int n, double** A){
    int a, b;
    int* indices = calloc(2, sizeof(int));
    indices[0] = -1;
    indices[1] = -1;
    double cur_max = -1 * DBL_MAX;
    for(a=0; a<n; a++){
        for(b=0; b<n; b++){
            if(a!=b){
                if(fabs(A[a][b]) > cur_max){
                    cur_max = fabs(A[a][b]);
                    indices[0] = a;
                    indices[1] = b;
                }
            }
        }
    }
    return indices;
}


/*
 * update P for every iteration
1. compute t, s, c, theta (calls helpers)
2. reset temp (I) (reset_matrix)
3. update P_prev and return it
 * */
void update_P (int n, int i, int j, double** A, double** P_prev){
    double theta = compute_theta(A[i][j], A[i][i], A[j][j]);
    double t = compute_t(theta);
    double c = compute_c(t);
    double s = t * c;
    reset_matrix(n, P_prev);
    P_prev[i][i] = c;
    P_prev[j][j] = c;
    P_prev[i][j] = s;
    P_prev[j][i] = -1 * s;
}


/* creates V (eigenvectors as columns) and A' (diagonal, eigenvalues of original A)

 V = I (create_identity);
 A = return copy of L_norm
 initialize temp(create_identity);
 for(int m=0; m<100; m++){
    1. compute i,j
    2. P_m = update_P (updated P_zona)
    3. compute off(A)
    4. set A=A' (call update_A_to_A_tag)
    6. V =V*P_m (matrix_mult)
    7. if(off(A)^2-off(A')^2<=eps){ break;}
 }
 now V holds eigenvectors and A holds eigenvalues
 return V and A
 * */

 double*** create_jacobi_matrix(int n, double** L_norm){
     double** temp_V;
     int* temp_indices;
     double*** VandA = calloc(2, sizeof(double**));
     double** V = create_identity_matrix(n);
     double** A = create_copy(n,L_norm);
     double** P_m = create_identity_matrix(n);
     double off_A, off_A_tag;
     double epsilon = 1.0 / 100000.0;
//     int* indices = calloc(2, sizeof (int));
//     indices[0] = -1;
//     indices[1] = -1;
    int* indices;
     int l, i, j;
     for(l=0; l<100; l++){
         indices = find_ij(n,A);
         temp_indices = indices;
         i = indices[0];
         j = indices[1];
         free(temp_indices);
         off_A = compute_off_diag(n,A);
         update_P(n, i, j, A, P_m);
         update_A_to_A_tag(n,i,j,A);
         off_A_tag = compute_off_diag(n,A);
         temp_V = V;
         V = matrix_mult(n,V,P_m);
         free_matrix(n,temp_V);
         if((off_A-off_A_tag)<=epsilon){ break;}
     }
     VandA[0] = V;
     VandA[1] = A;
     free_matrix(n,P_m);
     return VandA;
 }

/*
 * run jacobi algorithm
 * 1. call create_jacobi_matrix and compute V and A
 * 2. if k=0, calls find_eigengap and compute that required k
 * 3. indices= find_k_max_indices
 * 4. create and return U (call create_U)
 * */
double** jacobi_algorithm (int k, int n, int d, double** X){
    int* indices;
    double** U;
    double** T;
    double** L_norm = create_Lnorm(n,d,X);
    double*** VandA = create_jacobi_matrix(n, L_norm);
    double** V = VandA[0];
    double** A = VandA[1];
    /*
    printf("V: \n");
    printMatrix(V, n, n);
    printf("\n");
    printf("Eigen: \n");
    double* eigen = extract_diagonal(n, A);
    print_double_vector(eigen, n);
    printf("\n");
     */

    if(k==0){
        k = find_eigengap(n,A);
    }
    indices = find_k_max_indices(n,k,A);
    U = create_U(n,k,indices,V);
//    printf("U:");
//    printMatrix(U,n,k);
    free(VandA);
    T = create_T(n,k,U);
    free_matrix(n,V);
    free_matrix(n,A);
    free_matrix(n,U);
    free_matrix(n,L_norm);
    free(indices);

    return T;
}


/* normalizes U and returns it */
double** create_T(int rows, int cols, double** U){
    int i,j;
    double cur_row_sum, sqrt_row_sum;
    double** T = buildMatrix(rows,cols);
    for(i=0; i<rows; i++){
        cur_row_sum = 0.0;
        for (j=0; j<cols; j++){
            cur_row_sum += U[i][j]*U[i][j];
        }
        sqrt_row_sum = sqrt(cur_row_sum);
        for(j=0;j<cols;j++){
            T[i][j] = U[i][j] / sqrt_row_sum;
        }
    }
    return T;
}

void spk_helper(int k, int n, int d, double** X, char* input_file){
    double** T = jacobi_algorithm(k, n, d, X);
    write_output("tmp_T.txt", n, d, T);
    free_matrix(n, T);

}

void spk(int k, char* input_file){
    int n = count_rows(input_file);
    int d = count_cols(input_file);
    double** X = read_data_from_file(n, d, input_file);
    spk_helper(k, n, d, X, input_file);
    free_matrix(n, X);
}




/*
 ***** HELPERS METHODS ***
 * */

void print_error_and_exit(){
    printf("An Error Has Occurred");
    exit(1);
}



/* sum of array */
double sum_array(int n,  double* arr){
    int i;
    double sum = 0.0;
    for(i=0; i<n; i++){
        sum += arr[i];
    }
    return sum;
}

/* returns sum of squared items of an array  */
double sum_squared_array(int k, double* arr){
    int i;
    double sum = 0.0;
    for(i=0; i<k; i++){
        sum = sum +  (arr[i]*arr[i]);
    }
    return sum;
}

/* Matrix multiplication */
double** matrix_mult(int n, double **A, double **B){
    double sum;
    int a, b, c;
    double** new_mat = buildMatrix(n,n);
    for(a=0; a<n; a++){
        for(b=0; b<n; b++){
            sum = 0.0;
            for(c=0; c<n; c++){
                sum = sum+ (A[a][c] * B[c][b]);
            }
            new_mat[a][b] = sum;
        }
    }
    return new_mat;
}

/*
 * free from memory
 */
int free_matrix( int rows, double** pointer){
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
double compute_off_diag(int n, double **A){
    int i,j;
    double sum =0.0;
    for(i=0; i<n; i++){
        for(j=0; j<n; j++){
            if(i!=j){
                sum = sum + (A[i][j]*A[i][j]);
            }
        }
    }
    return sum;
}

/*
 * computes theta
 * */
double compute_theta(double A_ij, double A_ii, double A_jj){
    return (A_jj - A_ii) / (2.0 * A_ij);
}

/*
 * computes t
 * */
double compute_t(double theta){
    double sign_top;
    if (theta>=0){
        sign_top = 1.0;
     } else{
        sign_top = -1.0;
    }
    double denom = fabs(theta) + sqrt(theta * theta + 1 );
    return sign(theta) / denom;
}

double sign(double x){
    if(x>=0.0){
        return 1.0;
    }
    else{
        return -1.0;
    }
}

/*
 * computes c
 * */
double compute_c(double t){
    return (1.0 / (sqrt(t*t+1)));
}

/*
 * recieves a matrix and returns it as identity
 * */
void reset_matrix(int n, double**A){
    int i, j;
    for(i=0; i<n; i++){
        for(j=0; j<n; j++){
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
 * updates A_temp by formula (inplace)
  for testing - A_temp' is always symmetric
 * */
void update_A_to_A_tag(int n, int i, int j, double**A){
    double** A_temp = create_copy(n, A);
    double theta = compute_theta(A_temp[i][j], A_temp[i][i], A_temp[j][j]);
    double t = compute_t(theta);
    double c = compute_c(t);
    double s = t * c;
    int r;
    /* run i'th row */
    for(r=0; r < n; r++){
        A[i][r] =  c * A_temp[i][r] - s * A_temp[j][r];
    }

    /* run j'th row */
    for(r=0; r < n; r++){
        A[j][r] = s * A_temp[i][r] + c * A_temp[j][r];
    }

    /* run i'th column */
    for(r=0; r < n; r++){
        A[r][i] = c * A_temp[i][r] - s * A_temp[j][r];
    }

    /* run j'th column */
    for(r=0; r < n; r++){
        A[r][j] = s * A_temp[i][r] + c * A_temp[j][r];
    }

    /* update 4 special cases*/
    A[i][i] = (c * c * A_temp[i][i]) - (2.0 * s * c * A_temp[i][j]) + (s * s * A_temp[j][j]);
    A[j][j] = (s * s * A_temp[i][i]) + (2.0 * s * c * A_temp[i][j]) + (c * c * A_temp[j][j]);
    A[i][j] = (((c * c) - (s * s)) * A_temp[i][j]) + (s * c * (A_temp[i][i] - A_temp[j][j]));
    A[j][i] = (((c * c) - (s * s)) * A_temp[i][j]) + (s * c * (A_temp[i][i] - A_temp[j][j]));

    free_matrix(n, A_temp);

}

/*
 * find_k_max_indices
 * K times find the next maximum from eigenvalues in A, and returns those indices
 * */
int* find_k_max_indices(int n, int k, double** A){
    int* k_indices= calloc(k, sizeof(int));
    double* all_eigenvals = extract_diagonal(n,A);
    int i,j, cnt=0;
    double max_eigenval;
    int max_ind;
    for(i=0;i<k;i++){
        max_eigenval= DBL_MIN;
        max_ind = -1;
        for(j=0;j<n;j++){
            if(all_eigenvals[j]>max_eigenval) {
                max_eigenval = all_eigenvals[j];
                max_ind = j;
            }
        }
        k_indices[cnt] = max_ind;
        cnt++;
        all_eigenvals[max_ind] = DBL_MIN;
    }
    free(all_eigenvals);
    return k_indices;
}


/*
 * creates U from V by largest K eigenvalues
 * */
double** create_U (int n, int k, int* indices, double** V){
    double** U = buildMatrix(n,k);
    int i, j, cur_ind;
    for(i=0; i<k; i++){
        cur_ind = indices[i];
        for(j=0; j<n; j++){
            U[j][i] = V[j][cur_ind];
        }
    }
    return U;
}

double** create_copy(int n, double** A){
    int i,j;
    double** new_mat = buildMatrix(n,n);
    for(i=0; i<n; i++){
        for(j=0;j<n;j++){
            new_mat[i][j] = A[i][j];
        }
    }
    return new_mat;
}


int count_cols(char* filePath){
    /*
     * input: file Name
     * output: number of columns
     * details: open files, reads first line of file (loops until first '\n').
     *          counts number of ",", return counter+1 if not 0, otherwise 0.
     */

    char c;
    int counter=0;
    FILE *fp =  fopen(filePath,"r");

    if (fp==NULL){
        print_error_and_exit();
    }

    for (c= getc(fp); c!='\n'; c= getc(fp)){
        if (c==','){
            counter+=1;
        }
    }

    if(fclose(fp)!=0){
        print_error_and_exit();
    }

    if (counter==0){
        return 1;
    } else{
        return ++counter;
    }
}

int count_rows(char* filePath){
    /*
     * input: file Name
     * output: number of lines in file
     */
    char c;
    int counter=0;
    FILE *fp =  fopen(filePath,"r");

    if (fp==NULL){
        print_error_and_exit();
    }
    for (c= getc(fp); c!=EOF; c= getc(fp)){
        if (c=='\n'){
            counter+=1;
        }
    }
    if(fclose(fp)!=0){
        print_error_and_exit();
    }
    return counter;
}

double** read_data_from_file(int rows, int cols, char* filePath){
    /*
     * Creates empty matrix and fills it with read values from file
     */
    double** matrix;
    int lineSize = cols*32; /* 17 + 1 */
    char *token; /* String pointer*/
    int i=0,j=0;
    char* line;
    FILE *fp;

    line = calloc(lineSize, sizeof(char ));
    if(line == NULL){
        print_error_and_exit();
    }

    matrix = buildMatrix(rows,cols);

    fp = fopen(filePath,"r");

    if (fp==NULL){
        print_error_and_exit();
    }

    /* Reads each line as a string*/
    while (fgets(line,lineSize,fp)!=NULL){
        token = strtok(line,","); /* token is a string between 2 commas*/
        while (token!=NULL){               /* in end of line token is NULL*/
            matrix[i][j] = atof(token); /* converts the string token to double */
            token = strtok(NULL,","); /* move forward to the next comma. Pointer=NULL so it will continue from the last place */
            j++;
        }
        /* finished line*/
        i++;
        j=0;
    }
    if(fclose(fp)!=0){
        print_error_and_exit();
    }
    free(line);

    return matrix;
}

FILE* write_output(char* output_filename, int rows, int cols,double** Matrix){
    int r,c;
    char tmp_str[180];
    FILE* fp;
    fp = fopen(output_filename, "w");

    if (fp==NULL){
        print_error_and_exit();
    }
    for (r=0;r<rows;r++){
        c = 0;
        for (;c<cols-1;c++){
            sprintf(tmp_str,"%.4f",Matrix[r][c]) ; /* saves centroids[r][c] in tmp_str */
            fputs(tmp_str,fp);
            fputs(",",fp);
        }
        sprintf(tmp_str,"%.4f",Matrix[r][c]) ;
        fputs(tmp_str,fp);
        fputs("\n", fp);
    }
    if(fclose(fp)!=0){
        print_error_and_exit();
    }
    return fp;
}

/*
int main(){
    char* input_file = "jacobi_1.txt";
    int k = 3;
    spk(k, input_file);
    return 0;
}
*/