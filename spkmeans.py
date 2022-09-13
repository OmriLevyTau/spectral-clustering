import os
import os.path
import sys
import numpy as np
import pandas as pd
import myspkmeansmodule as spkmeansmodule
from typing import List
np.random.seed(0)

goals_list = ["spk", "wam", "ddg", "lnorm", "jacobi"]

def initialize_centroids(S, k):
    N = len(S)
    i = 1
    first_idx = np.random.randint(N)
    centroids = np.array(S[first_idx]).reshape(-1, len(S[0]))
    indices = [first_idx]
    while i < k:
        D = np.zeros(N)
        for l in range(N):
            D[l] = (np.linalg.norm(S[l] - centroids, axis=1) ** 2).min()
        D_SUM = D.sum()
        Prob = D / D_SUM
        next_mu_index = np.random.choice(np.arange(N), p=Prob)
        centroids = np.append(centroids, [S[next_mu_index]], axis=0)
        indices.append(next_mu_index)
        i += 1
    return centroids, indices


def read_data(name):
    res = pd.read_csv(name, sep=",", header=None)
    return res.to_numpy()


def validate_input_args(argv):
    n = len(argv)

    if (n != 4):
        return True

    k, goal, input_file = argv[1], argv[2], argv[3]

    try:
        with open(input_file, 'r') as file:
            pass
    except:
        return True

    try:
        k = float(k)
    except:
        return True

    if k != int(k) or k < 0:
        return True

    if goal not in goals_list:
        return True

    return False


def write_output(input_text, output_filename):
    with open(output_filename, "w") as file:
        for line in input_text:
            line_data = make_string(line)
            file.writelines(line_data)
            file.write("\n")


def make_string(centroid):
    st = ""
    for cell in centroid:
        tmp = "%.4f" % round(cell, 4)
        st = st + tmp + ","
    return st[:len(st) - 1]

def clear_tmp_files(files_list):
    for file in files_list:
        if os.path.exists(file):
            os.remove(file)

def print_output(centroids,initial_centroids_indices):
    st = ""
    for i in range(len(centroids)-1):
        st += make_string(centroids[i]) + "\n";
    st += make_string(centroids[-1]);

    result = ','.join(str(ind) for ind in initial_centroids_indices)
    print(result + "\n" + st);


def print_matrix(matrix):
    st = ""
    for i in range(len(matrix)-1):
        st += make_string(matrix[i]) + "\n";
    st += make_string(matrix[-1])
    print(st);


def main():
    argv = sys.argv
    if validate_input_args(argv):
        print("Invalid Input!")
        return

    try:
        k, goal, input_file = argv[1], argv[2], argv[3]
        X = read_data(input_file)
        n,d = X.shape[0], X.shape[1]
        k = int(k)

        if goal=="spk":
            if k>=n:
                print("Invalid Input!")
                return

            spkmeansmodule.spk(k, input_file) # calculates T and writes to temp file

            T_path = os.path.join(os.getcwd(), "tmp_T" + "." + "txt")
            X = read_data(T_path);
            k = X.shape[1] # if k==0 it is recalculated in jacobi_algorithm.
            initial_centroids, initial_centroids_indices = initialize_centroids(X.tolist(), k) # here k!=0
            write_output(initial_centroids.tolist(), "tmp_initial_centroids.txt") # writes initial centroids to a temp file
            initial_centroids_path = os.path.join(os.getcwd(), "tmp_initial_centroids" + "." + "txt")
            centroids = spkmeansmodule.fit(k, 300, 0, T_path, initial_centroids_path) # execute kmeans
            clear_tmp_files(["tmp_T.txt", "tmp_initial_centroids.txt"])
            print_output(centroids, initial_centroids_indices)
        elif goal=="wam":
            W = spkmeansmodule.wam(n, d, input_file)
            print_matrix(W)
        elif goal=="ddg":
            D = spkmeansmodule.ddg(n, d, input_file)
            print_matrix(D)
            pass
        elif goal=="lnorm":
            lnorm = spkmeansmodule.lnorm(n, d, input_file)
            print_matrix(lnorm)
        elif goal=="jacobi":
            J = spkmeansmodule.jacobi(n, d, input_file) # returns the result ready to print
            print_matrix(J)
        else:
            print("Invalid Input!") #should not happen

    except:
        print("An Error Has Occurred")



if __name__ == "__main__":
    main()