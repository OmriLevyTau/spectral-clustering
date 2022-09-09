import numpy as np;

## ----------- print settings -----------
np.set_printoptions(linewidth=900);
np.set_printoptions(precision=6)
np.set_printoptions(suppress=True);

## ----------- matrices declerations -----------


input_1_sym = np.array([[ 1.      , -0.01021 , -0.001042, -0.999721],
                       [-0.01021 ,  1.      , -0.26966 , -0.011936],
                       [-0.001042, -0.26966 ,  1.      , -0.000928],
                       [-0.999721, -0.011936, -0.000928,  1.      ]])

input_11_sum = np.array([
    [1.0000, -0.0001, -0.0044, -0.0030, -0.0003, -0.2055, -0.3107, -0.0088, -0.0001, -0.1994],
    [-0.0001, 1.0000, -0.0000, -0.0000, -0.4105, -0.0002, -0.0002, -0.0001, -0.5695, -0.0002],
    [-0.0044, -0.0000, 1.0000, -0.6022, -0.0001, -0.0042, -0.0035, -0.4883, -0.0000, -0.0049],
    [-0.0030, -0.0000, -0.6022, 1.0000, -0.0000, -0.0029, -0.0024, -0.3597, -0.0000, -0.0034],
    [-0.0003, -0.4105, -0.0001, -0.0000, 1.0000, -0.0007, -0.0005, -0.0001, -0.5130, -0.0007],
    [-0.2055, -0.0002, -0.0042, -0.0029, -0.0007, 1.0000, -0.3823, -0.0093, -0.0003, -0.4874],
    [-0.3107, -0.0002, -0.0035, -0.0024, -0.0005, -0.3823, 1.0000, -0.0077, -0.0002, -0.3498],
    [-0.0088, -0.0001, -0.4883, -0.3597, -0.0001, -0.0093, -0.0077, 1.0000, -0.0001, -0.0111],
    [-0.0001, -0.5695, -0.0000, -0.0000, -0.5130, -0.0003, -0.0002, -0.0001, 1.0000, -0.0003],
    [-0.1994, -0.0002, -0.0049, -0.0034, -0.0007, -0.4874, -0.3498, -0.0111, -0.0003, 1.0000]])



## ----------- Helpers -----------
def sign(x):
    if x>=0.0:
        return 1.0
    return -1.0


def find_ij(n,M):
    indices = [-1, -1]
    cur_max = -1*np.inf
    for i in range(n):
        for j in range(n):
            if (i!=j):
                if np.abs(M[i,j])>cur_max:
                    cur_max=np.abs(M[i,j])
                    indices[0]=i
                    indices[1]=j
    return indices


def next_P(n, i, j, A):
    theta = (A[j,j] - A[i,i]) / (2 * A[i,j])
    t = sign(theta) / (np.abs(theta) + ((theta ** 2) + 1) ** 0.5)
    c = 1 / (((t ** 2) + 1) ** 0.5)
    s = t * c

    new_P = np.eye(n)

    new_P[i,i] = c
    new_P[j,j] = c
    new_P[i,j] = s
    new_P[j,i] = -1 * s
    return new_P


def new_A(n, i, j, A):
    theta = (A[j,j] - A[i,i]) / (2 * A[i,j])  # Replace with sin
    t = sign(theta) / (np.abs(theta) + ((theta ** 2) + 1) ** 0.5)
    c = 1 / (((t ** 2) + 1) ** 0.5)
    s = t * c
    new_A = np.copy(A)

    for l in range(n):
        new_A[i,l] = c*A[i,l] - s*A[j,l]

    for l in range(n):
        new_A[l,i] = c*A[i,l] - s*A[j,l]

    for l in range(n):
        new_A[j, l] = s * A[i, l] + c * A[j, l]

    for l in range(n):
        new_A[l, j] = s * A[i, l] + c * A[j, l]

    new_A[i,i] = (c * c * A[i,i]) - (2 * s * c * A[i,j]) + (s * s * A[j,j])
    new_A[j,j] = (s * s * A[i,i]) + (2 * s * c * A[i,j]) + (c * c * A[j,j])
    new_A[i,j] = (((c * c) - (s * s)) * A[i,j]) + (s * c * (A[i,i] - A[j,j]))
    new_A[j,i] = (((c * c) - (s * s)) * A[i,j]) + (s * c * (A[i,i] - A[j,j]))

    return new_A


def create_jacobi_matrix(n, L_norm):
    V = np.eye(n)
    A = np.copy(L_norm)
    P_m = np.eye(n)
    indices = (-1, -1)
    for l in range(3000):
        i, j = find_ij(n, A)
        P_m = next_P(n, i, j, A)
        A = new_A(n, i, j, A)  # TODO
        V = np.matmul(V, P_m)
    return (V, A)



## --------- methods from jupyter ----------
def norm_dist(a, b):
    return np.linalg.norm(a-b);


def weighted_distance(a,b):
    norm =  norm_dist(a,b);
    return np.exp(-0.5 * norm);


def create_wam(n, X):
    Y = np.eye(n)
    for i in range(n):
        for j in range(n):
            if(i==j):
                Y[i][j] = 0.0;
            else:
                Y[i][j] = weighted_distance(X[i], X[j])
    return Y


def create_ddg(n, X):
    D = np.eye(n)
    W = create_wam(n, X)
    for i in range(n):
        D[i][i] = W[i].sum()
    return D


def create_d_inverse(n, X):
    D = create_ddg(n, X)
    D_inverse = np.eye(n);
    for i in range(n):
        D_inverse[i][i] = 1 / np.sqrt(D[i][i])
    return D_inverse


def create_Lnorm(n, X):
    D_inv = create_d_inverse(n, X);
    W = create_wam(n, X);
    I = np.eye(n);
    return I - (D_inv@(W@D_inv))

def compare_matrices_sign_agnostic(n, expected, actual):
    # print("\nComparing:\n")
    expected = expected[:,np.abs(expected[0]).argsort()]
    # print(expected)
    # print()
    actual = actual[:,np.abs(actual[0]).argsort()]
    # print(actual)
    for i in range(n):
        opt1 = np.abs((expected[:,i]-actual[:,i])).sum()
        opt2 = np.abs((-1*expected[:,i]-actual[:,i])).sum()
        if (opt1>0.1 and opt2>0.1):
            print(f"opt1: {opt1}, opt2: {opt2}")
            # print("\nColumn not equal: {}".format(i))
            # print(f"\nopt1: {opt1}, opt2: {opt2}")
            # print(expected[:,i].T)
            # print(-1*expected[:, i].T)
            # print(actual[:,i].T)
            return False
    return True

## ----------- run test -----------
def load_text_to_numpy(file_name):
    data = []
    with open(file_name, "r") as f:
        for line in f:
            str_list = line.split(",")
            num_list = [float(x) for x in str_list]
            data.append(num_list)
    return np.array(data)


def run_test_on_file(n, file_name):
    V,A = (create_jacobi_matrix(n, file_name))
    print("### Our Result ###\n")
    print("V:")
    print(V)
    print("\nEigen vals:")
    print(np.sort(A.diagonal()))
    print()
    print("### Numpy Result ###\n")
    print("Eigen vals:")
    eig_np, V_np = np.linalg.eig(file_name)
    print(np.sort(eig_np))
    print("\nV:\n")
    print(V_np)
    print("\n### Final Verdict ###")
    print("\n Matrices span same space? {}\n".format(compare_matrices_sign_agnostic(n,V, V_np)))

def run_tests():
    base = "C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\inputs\\sym_matrix\\sym_matrix_input_"
    for i in range(11,20):
        print(f"Test {i}")
        file_name = base + f"{i}"+".txt"
        arr = load_text_to_numpy(file_name)
        n = arr.shape[0]
        run_test_on_file(n, arr)

if __name__ == '__main__':
    base = "C:\\Users\\Omri\\Desktop\\spectral-clustering\\test_files\\inputs\\sym_matrix\\"
    arr = load_text_to_numpy(base+"sym_matrix_input_17.txt")
    n = arr.shape[0]
    print(arr.shape)
    run_test_on_file(n, arr)
