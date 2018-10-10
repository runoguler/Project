import numpy as np
from scipy.linalg import cholesky
from scipy.stats import pearsonr

def main():
    corr_mat = np.array([[1.0, -0.6, 0.3],
                         [-0.6, 1.0, 0.5],
                         [0.3, 0.5, 1.0]])

    upper_chol = cholesky(corr_mat)

    rnd = np.random.normal(0.0, 1.0, size=(10 ** 7, 3))

    ans = rnd @ upper_chol

    print(ans)

    corr_0_1, _ = pearsonr(ans[:, 0], ans[:, 1])

    print(corr_0_1)

if __name__ == '__main__':
    main()
