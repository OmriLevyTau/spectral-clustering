import os
import os.path
import sys
import numpy as np
import pandas as pd
# import spkmeansmodule
# import mykmeanssp
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


def read_data(name: str):
    res = pd.read_csv(name, sep=",", header=None)
    res.sort_values(by=0, inplace=True)
    return res.drop(0, axis=1).to_numpy()


def validate_input_args(argv: List[str]) -> bool:
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

    if k != int(k) or k <= 0:
        return True

    if goal not in goals_list:
        return True

    return False


def write_output(input_text, output_filename: str):
    with open(output_filename, "w") as file:
        for line in input_text:
            line_data = make_string(line)
            file.writelines(line_data)
            file.write("\n")


def make_string(centroid: List['float']):
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
    for centroid in centroids:
        st += make_string(centroid) + "\n";
    result = ','.join(str(ind) for ind in initial_centroids_indices)
    print(result + "\n" + st);

def print_matrix(matrix):
    st = ""
    for row in matrix:
        st += make_string(row) + "\n";
    print(st);

def main():
    argv = sys.argv
    if validate_input_args(argv):
        print("Invalid Input!")
        sys.exit()

    print(argv)

    try:
        k, goal, input_file = argv[1], argv[2], argv[3]
        X = read_data(input_file)
        n,d = X.shape[0], X.shape[1]

        if goal=="spk":
            # spkmeansmodule.jacobi_algorithm (n, d, input_file)
                # calculates T and writes to temp file
            # T_path = os.path.join(os.getcwd(), "tmp_T" + "." + "txt")
            # X = read_data(T_path)
            # k = X.shape[1]
                # if k==0 is re-calculated in jacobi_algorithm, now I got it in my hand again.
            # initial_centroids, initial_centroids_indices = initialize_centroids(X.tolist(), k)
            # write_output(initial_centroids.tolist(), "tmp_initial_centroids.txt")
                # calculates initial centroids and write to a temp file
            # initial_path = os.path.join(os.getcwd(), "tmp_initial_centroids" + "." + "txt")
            # centroids = mykmeanssp.fit(k, max_iter, eps, combined_path, initial_path)
                # execute kmeans
            # clear_tmp_files(["tmp_T.txt", "tmp_initial_centroids.txt"])
            # print_output(centroids, initial_centroids_indices)
            pass

        elif goal=="wam":
            # W = spkmeansmodule.create_wam_api(n, d, input_file)
            # print_matrix(W)
            pass
        elif goal=="ddg":
            # D = spkmeansmodule.create_ddg_api(n, d, input_file)
            # print_matrix(D)
            pass
        elif goal=="lnorm":
            # lnorm = spkmeansmodule.create_lnorm_api(n, d, input_file)
            # print_matrix(lnorm)
            pass
        elif goal=="jacobi":
            # jacobi = spkmeansmodule.create_jacobi_api(n, d, input_file)
            # (returns the result ready tp print!)
            # print_matrix(jacobi)
            pass
        else:
            print("Invalid Input!") #should not happen


    except:
        print("An Error Has Occurred")

    clear_tmp_files()

if __name__ == "__main__":
    main()