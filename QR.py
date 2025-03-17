# Dotan Sadka 318474657
import numpy as np

def generate_random_matrix(N,kappa):
    '''
    :param N: Generate a random NxN matrix.
    :param kappa: Condition number of the random matrix. kappa>1.
    :return:
    '''
    B = np.random.randn(N, N)
    U, D, Vt = np.linalg.svd(B)
    D = np.linspace(kappa, 1, N)  # generate matrix with large condition number
    A =  U.dot(np.diag(D).dot(Vt))
    return A

def QR_GS(A,method='CGS'):
    '''
    Compute QR decomposition via Graham-Schmidt
    :param A: The matrix to decompose, can assume square and full rank
    :param method: 'CGS' is classic GS, 'MGS' is modified GS
    :return: Q,R. Q NxN orthogonal matrix and R NxN upper-diagonal matrix with non-zero diagonal elements.
    '''
    N = A.shape[0]
    Q = np.zeros((N, N))
    R = np.zeros((N, N))

    if method == 'CGS':
        for i in range(N):
            #creating U - vector
            u = A[:, i]
            for j in range(i):
                #computes the dot product 
                R[j, i] = np.dot(Q[:, j], A[:, i])
                u = u - R[j, i] * Q[:, j]
            #Normalize the Vector u
            R[i, i] = np.linalg.norm(u)
            Q[:, i] = u / R[i, i]
    elif method == 'MGS':
        #in modified GS we have a small difference
        for i in range(N):
            u = A[:, i]
            R[i, i] = np.linalg.norm(u)
            Q[:, i] = u / R[i, i]
            for j in range(i + 1, N):
                R[i, j] = np.dot(Q[:, i], A[:, j])
                A[:, j] = A[:, j] - R[i, j] * Q[:, i]

    return Q, R   
    # By the end of this process for each i, Q will contain
    # orthonormal columns and R will be an upper triangular matrix with
    # A = QR

def QR_householder(A):
    '''
    Compute QR decomposition via Householder transforms
    :param A: The matrix to decompose, can assume square and full rank
    :return: Q,R. Q is a list of v vectors representing the Householder transforms and R NxN upper-diagonal matrix with non-zero diagonal elements.
    '''
    #Each Householder transformation can be represented as H=I−2vvT,
    # where v is the Householder vector.
    N = A.shape[0]
    R = A.copy()
    Q_list = []

    for k in range(N - 1):
        x = R[k:, k]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        v = x + np.sign(x[0]) * e
        v = v / np.linalg.norm(v)
        Q_list.append(v)
        H = np.eye(N)
        H[k:, k:] -= 2.0 * np.outer(v, v)
        R = H @ R

    return Q_list, R

def apply_Q_householder(x,Q):
    '''
    :param x: a length N vector
    :param Q: Q is a list of v vectors representing the Householder transforms
    :return: Qx
    '''
   
    N = len(x)
    for i, v in enumerate(Q):
        k = i
        x[k:] -= 2 * v * np.dot(v, x[k:])
    return x

def apply_Q_transpose_householder(x,Q):
    '''
    :param x: a length N vector
    :param Q: Q is a list of v vectors representing the Householder transforms
    :return: Q^Tx
    '''
    N = len(x)
    for i, v in enumerate(reversed(Q)):
        k = N - len(v)
        x[k:] -= 2 * v * np.dot(v, x[k:])
    return x

def solve_backsubstitution(x,R):
    '''
    :param x: length N vector
    :param R: Upper diagonal matrix with non-zero diagonal elements
    :return: z such that Rz=x
    '''
    #Back substitution solves the upper triangular system Rz=x
    #working from the bottom row to the top row. For each row,
    #it computes the corresponding element of the solution vector z
    #subtracting the contributions of the already known variables
    #(from previously computed elements of z) and then dividing by
    #the diagonal element.

    N = len(x)
    z = np.zeros_like(x)
    for i in range(N - 1, -1, -1):
        z[i] = (x[i] - np.dot(R[i, i+1:], z[i+1:])) / R[i, i]
    return z


def solve_QR(A,b,method):
    '''

    :param A: NxN invertible matrix
    :param b: length N vector
    :param method: 'CGS','MGS' or 'householder'
    :return: x such that Ax=b
    '''
    if method=='householder':
        Q,R = QR_householder(A)
        z = apply_Q_transpose_householder(b,Q)
    else:
        Q,R = QR_GS(A, method=method)
        z = Q.transpose().dot(b)
    return solve_backsubstitution(z,R)

def test_GS(A,method):
    ''' Test GS decomposition
    :param A: NxN invertible matrix
    :param method: 'CGS' or 'MGS'
    :return: errors
    '''
    # These errors help in assessing the numerical stability
    # and accuracy of the QR decomposition method used
    Q,R = QR_GS(A, method=method)
    N = Q.shape[0]
    eye = np.eye(N)
    I = Q.dot(Q.transpose())
    A_ = Q.dot(R)
    orth_err = np.linalg.norm(I-eye,ord=2) #operator 2 norm
    recon_err = np.linalg.norm(A-A_,ord=2)
    b = np.random.randn(N)
    x = solve_QR(A,b,method)
    b_ = A.dot(x)
    sol_err = np.linalg.norm(b-b_)
    return orth_err,recon_err,sol_err

def test_House(A):
    ''' Test GS decomposition
    :param A: NxN invertible matrix
    :return: errors
    '''
    Q,R = QR_householder(A)
    N = R.shape[0] # Corrected to use R.shape instead of Q.shape
    b = np.random.randn(N)
    z = apply_Q_transpose_householder(b,Q)
    recon = apply_Q_householder(z,Q)
    orth_err = np.linalg.norm(b-recon)
    x = solve_QR(A,b,'householder')
    b_ = A.dot(x)
    sol_err = np.linalg.norm(b-b_)
    return orth_err,sol_err

def main():
    N = 20
    kappas = [1, 10, 100, 1000]
    methods = ['CGS', 'MGS', 'householder']
    test_number = 20 

    results = {method: {kappa: {'orth_err': [], 'recon_err': [], 'sol_err': []} for kappa in kappas} for method in methods}

    for kappa in kappas:
        for _ in range(test_number):
            A = generate_random_matrix(N, kappa)
            for method in methods:
                if method == 'householder':
                    orth_err, sol_err = test_House(A)
                    results[method][kappa]['orth_err'].append(orth_err)
                    results[method][kappa]['sol_err'].append(sol_err)
                else:
                    orth_err, recon_err, sol_err = test_GS(A, method)
                    results[method][kappa]['orth_err'].append(orth_err)
                    results[method][kappa]['recon_err'].append(recon_err)
                    results[method][kappa]['sol_err'].append(sol_err)

    for method in methods:
        for kappa in kappas:
            orth_errs = results[method][kappa]['orth_err']
            recon_errs = results[method][kappa]['recon_err']
            sol_errs = results[method][kappa]['sol_err']
            print(f"Method: {method}, Kappa: {kappa}")
            print(f"  Orthogonalization Error: Mean = {np.mean(orth_errs)}, STD = {np.std(orth_errs)}")
            if method != 'householder':
                print(f"  Reconstruction Error: Mean = {np.mean(recon_errs)}, STD = {np.std(recon_errs)}")
            print(f"  Solution Error: Mean = {np.mean(sol_errs)}, STD = {np.std(sol_errs)}")

if __name__ == "__main__":
    main()
