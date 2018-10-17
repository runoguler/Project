import numpy as np
from scipy.linalg import cholesky
from scipy.stats import pearsonr

def main():
    np.set_printoptions(linewidth=320)
    preference_table = np.random.choice(2, (10, 10000000), p=[0.7, 0.3])
    print('Correlations: \n', np.corrcoef(preference_table))
    corr_0_1, _ = pearsonr(preference_table[0, :], preference_table[1, :])
    corr_0_2, _ = pearsonr(preference_table[1, :], preference_table[2, :])
    corr_0_3, _ = pearsonr(preference_table[2, :], preference_table[3, :])
    corr_0_4, _ = pearsonr(preference_table[3, :], preference_table[4, :])
    corr_0_5, _ = pearsonr(preference_table[4, :], preference_table[5, :])

    print(corr_0_1)
    print(corr_0_2)
    print(corr_0_3)
    print(corr_0_4)
    print(corr_0_5)

    # corr_mat = np.array([[1.0, 0.6, 0.3, 0.1],
    #                      [0.6, 1.0, 0.5, 0.4],
    #                      [0.3, 0.5, 1.0, 0.8],
    #                      [0.1, 0.4, 0.8, 1.0]])
    #
    # upper_chol = cholesky(corr_mat)
    # print(upper_chol)
    #
    # rnd = np.random.normal(0.5, 1.0, size=(10 ** 7, 4))
    # # rnd = np.random.choice(2, (10 ** 7, 4))
    #
    # # rnd[rnd >= 0] = 1
    # # rnd[rnd < 0] = 0
    # ans = rnd @ upper_chol
    # ans[ans > 0.5] = 1
    # ans[ans < 0.5] = 0
    # print(ans)
    #
    # corr_0_1, _ = pearsonr(ans[:, 0], ans[:, 1])
    #
    # print(corr_0_1)

if __name__ == '__main__':
    main()
