#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include "spkmeans.h"

double** createMatrix(int rows, int cols, char* filePath);
double* sub_vectors(const double *A, const double *B, int n);
double* add_vectors(const double *A, const double *B, int n);
double squared_dot_product(const double *A, const double *B, int n);
double** K_means(int K, int max_iter, double epsilon, char* tmp_combined_inputs, char* tmp_initial_centroids);
void print_error_and_exit();

void print_error_and_exit(){
    printf("An Error Has Occurred");
    exit(1);
}

double** createMatrix(int rows, int cols, char* filePath){
    /*
     * Creates empty matrix and fills it with read values from file
     */
    double** matrix;
    int lineSize = cols*18; /* 17 + 1 */
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

double* sub_vectors(const double *A, const double *B, int n){
    int i;
    double* res;

    res = (double*)malloc(n*sizeof(double));

    if (res==NULL){
        print_error_and_exit();
    }
    for(i=0; i<n; i++){
        res[i] = A[i] - B[i];
    }
    return res;
}

double* add_vectors(const double *A, const double *B, int n){
    int i;
    double* res;

    res = (double*)malloc(n*sizeof(double));

    if (res==NULL){
        print_error_and_exit();
    }
    for(i=0; i<n; i++){
        res[i] = A[i] + B[i];
    }
    return res;
}

double squared_dot_product(const double *A, const double *B, int n){
    int i;
    double res;
    res = 0;
    for(i=0; i<n; i++){
        res = res + (A[i] * B[i]);
    }
    return res;
}

double** copy(double** data, int K, int cols){
    int i,j;
    double** new_mat;
    new_mat = buildMatrix(K, cols);
    for(i=0; i<K; i++){
        for(j=0; j<cols; j++){
            new_mat[i][j] = data[i][j];
        }
    }
    return new_mat;
}

int free_helper(double** pointer, int rows){
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


double** K_means(int K, int max_iter, double epsilon, char* tmp_combined_inputs, char* tmp_initial_centroids){
    /*
     * recieves input file, K = number of clusters, max_iter = max number of iterations
     * connects every point to the closest cluster
     * returns vector of centroids
     */
    double ** data;
    double** centroids;
    int idx, arg_min, counter,iter,point,cluster_ind, r, k, c, rows, cols, rows_pp,cols_pp, f1;
    double min_dist, dist_point_cluster;
    double** cluster_sum;
    double** old_centroids;
    double* cluster;
    double* tmp_arr;
    double* tmp_pointer;
    int* points_clusters;
    double* cluster_change;
    int* cluster_counter;
    double* tmp_vec;

    /* read data points */
    rows = count_rows(tmp_combined_inputs);
    cols = count_cols(tmp_combined_inputs);

    points_clusters = calloc(rows, sizeof(int));
    if(points_clusters==NULL){
        print_error_and_exit();
    }
    data = createMatrix(rows,cols,tmp_combined_inputs);

    /* Read initial centroids from kmeans_pp */
    rows_pp = count_rows(tmp_initial_centroids);
    cols_pp = count_cols(tmp_initial_centroids);
    /*
    centroids = calloc(rows_pp, sizeof(int));
    if(centroids==NULL){
        print_error_and_exit();
    }
     */
    centroids = createMatrix(rows_pp,cols_pp,tmp_initial_centroids);

    /* Train Model */

    for(iter=0; iter<max_iter; iter++){
        /* iterate through points and assign to the closest cluster */
        for (point=0; point<rows; point++){
            min_dist = INT_MAX;
            arg_min = -1;
            for(cluster_ind=0; cluster_ind<K; cluster_ind++){
                cluster = centroids[cluster_ind];
                tmp_arr = sub_vectors(cluster,data[point], cols);
                dist_point_cluster = squared_dot_product(tmp_arr,tmp_arr,cols);
                if(dist_point_cluster<min_dist){
                    min_dist = dist_point_cluster;
                    arg_min = cluster_ind;
                }
                free(tmp_arr);
            }
            points_clusters[point] = arg_min;
        }
        /* calculate new centroids */
        old_centroids = copy(centroids,K, cols); /* for changes checking */
        cluster_sum = buildMatrix(K, cols); /* zero matrix */
        cluster_change = calloc(K, sizeof(double));
        if(cluster_change==NULL){
            print_error_and_exit();
        }
        cluster_counter = calloc(K, sizeof(int));
        if(cluster_counter==NULL){
            print_error_and_exit();
        }

        memset(cluster_counter, 0, K*sizeof(int)); /* zero array */
        memset(cluster_change, 0, K*sizeof(double));

        /* sum and count */
        for(r=0; r<rows; r++){
            idx = points_clusters[r];
            cluster_counter[idx] += 1;
            tmp_pointer = cluster_sum[idx];
            cluster_sum[idx] = add_vectors(cluster_sum[idx], data[r], cols);
            free(tmp_pointer);
        }

        /* update centroids */
        counter = 0;
        for(k=0; k<K; k++){
            for(c=0; c<cols; c++){
                if (cluster_counter[k]==0){
                    print_error_and_exit();
                }
                centroids[k][c] = cluster_sum[k][c] / cluster_counter[k];
            }
            /* check change vector */
            tmp_vec = sub_vectors(centroids[k],old_centroids[k], cols);
            cluster_change[k] = sqrt(squared_dot_product(tmp_vec, tmp_vec, cols));
            if(cluster_change[k]<epsilon){
                counter += 1;
            }
            free(tmp_vec);
        }
        free(cluster_change);
        free(cluster_counter);
        free_helper(old_centroids, K);
        free_helper(cluster_sum, K);

        /* check if all coordinates changes are less than epsilon*/
        if(counter == K){
            break;
        }
    }

    /* free matrices */
    f1 = free_helper(data,rows);
    free(points_clusters);
    if (f1>0){
        print_error_and_exit();
    }
    return centroids;
}
