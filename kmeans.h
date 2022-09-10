//
// Created by Omri on 9/10/2022.
//

#ifndef SPECTRAL_CLUSTERING_KMEANS_H
#define SPECTRAL_CLUSTERING_KMEANS_H

double** createMatrix(int rows, int cols, char* filePath);
double* sub_vectors(const double *A, const double *B, int n);
double* add_vectors(const double *A, const double *B, int n);
double squared_dot_product(const double *A, const double *B, int n);
double** K_means(int K, int max_iter, double epsilon, char* tmp_combined_inputs, char* tmp_initial_centroids);
void print_error_and_exit();

#endif //SPECTRAL_CLUSTERING_KMEANS_H
