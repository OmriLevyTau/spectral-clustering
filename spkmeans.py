import os
import os.path
import sys
import numpy as np
import pandas as pd
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

def clear_tmp_files():
    if os.path.exists("tmp_initial_centroids.txt"):
        os.remove("tmp_initial_centroids.txt")

def print_output(centroids,initial_centroids_indices):
    st = ""
    for centroid in centroids:
        st += make_string(centroid) + "\n";
    result = ','.join(str(ind) for ind in initial_centroids_indices)
    print(result + "\n" + st);


def main():
    argv = sys.argv
    if validate_input_args(argv):
        print("Invalid Input!")
        sys.exit()

    print(argv)

    try:
        k, goal, input_file = argv[1], argv[2], argv[3]
        X = read_data(input_file)

        if goal=="spk":
            # compute wam from X
            # compute lnorm from X
            # Determine k and obtain the largest k eigenvectors of lnorm matrix
            # create U, then create T (// TODO: can use python?)
            # treat T as the input data points, call kmeans++ algorithm:
                # get initial centroids out of T
                # write this output to a temporary file and get its path
                # call kmeans.c algortihm using c interface
            # print:
                # first line are the indices of the initial centroids chosen
                # second line onward are the final centroids

            pass

        elif goal=="wam":
            # compute wam from X using C module
            # print wam
            pass
        elif goal=="ddg":
            # compute ddg from X using C module
            # print ddg
            pass
        elif goal=="lnorm":
            # compute lnorm from X using C module
            # print lnorm
            pass
        elif goal=="jacobi":
            # compute jacobi from X using C module
            # print:
                # first line are the eigenvalues
                # then eigenvectors as columns
            pass
        else:
            print("Invalid Input!") #should not happen


    except:
        print("An Error Has Occurred")

    clear_tmp_files()

if __name__ == "__main__":
    main()