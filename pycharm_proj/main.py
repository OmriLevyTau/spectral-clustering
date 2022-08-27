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

input_9 = np.array([[1.0000,-0.6822,-0.0693],
                    [-0.6822,1.0000,-0.6822],
                    [-0.0693,-0.6822,1.0000]])

jacobi_0 = np.array([[0.7482679812896987,0.9962678716348444,0.705752719530148,0.3327360839555057,0.556446876169447],
[0.9962678716348444,0.4881966580554369,0.0015561979296321304,0.4511027753265111,0.9266007970596744],
[0.705752719530148,0.0015561979296321304,0.7712266730402363,0.004980905673753422,0.1562223832155636],
[0.3327360839555057,0.4511027753265111,0.004980905673753422,0.8243926884444718,0.8673296129309216],
[0.556446876169447,0.9266007970596744,0.1562223832155636,0.8673296129309216,0.5002389308425548]])

jacobi_1_T = np.array([[0.88759,-0.3808,0.25912],
                       [0.36449,-0.7417,0.56309],
                       [0.70286,0.70657,0.08207],
                       [0.60753,0.01596,-0.7941],
                       [0.27527,0.74068,0.61287]])

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

def compute_off_diag(mat, n):
    res = 0.0;
    for i in range(n):
        for j in range(n):
            if(i!=j):
                res = res + mat[i][j]*mat[i][j];
    return res;

def create_jacobi_matrix(n, L_norm):
    V = np.eye(n)
    A = np.copy(L_norm)
    P_m = np.eye(n)
    indices = (-1, -1)
    # epsilon = 1.0 / 100000
    for l in range(100):
        off_A = compute_off_diag(A, n);
        i, j = find_ij(n, A)
        P_m = next_P(n, i, j, A)
        A = new_A(n, i, j, A)  # TODO
        off_A_tag = compute_off_diag(A, n);
        V = np.matmul(V, P_m)
        # if(off_A -off_A_tag<=epsilon):
        #     break;
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





## ----------- run test -----------
def run_test():
    # L_norm = create_Lnorm(5,jacobi_0);
    # print("Lnorm")
    # # print(L_norm);
    # V,A = (create_jacobi_matrix(5,jacobi_0));
    # print(V)
    # print()
    # print(np.sort(A.diagonal()))
    # print()
    # print("Numpy Calc: ")
    # eig_np, V_np = np.linalg.eig(jacobi_0);
    # print(np.sort(eig_np))
    # print()
    # print(V_np)


    # print((jacobi_0_T**2).sum(axis=1)**0.5)

if __name__ == '__main__':
    run_test()