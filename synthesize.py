import numpy as np
import random as rd
from sklearn.cluster import AgglomerativeClustering as AggCl
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import pdist

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def random_pref_table(no_classes, no_people):
    pref_table = np.random.randint(0, 2, (no_people, no_classes))
    return pref_table


def random_cooccurence_matrix(no_classes):
    cooccurrence_matrix = np.random.rand(no_classes, no_classes)
    for i in range(no_classes):
        cooccurrence_matrix[i][i] = 1
    return cooccurrence_matrix


def data_to_cooccurence_matrix(pref_table):
    cooccurrence_matrix = np.dot(pref_table.transpose(), pref_table)

    cooccurrence_matrix_diagonal = np.diagonal(cooccurrence_matrix)
    with np.errstate(divide='ignore', invalid='ignore'):
        cooccurrence_matrix_percentage = np.nan_to_num(
            np.true_divide(cooccurrence_matrix, cooccurrence_matrix_diagonal[:, None]))

    return cooccurrence_matrix_percentage


def cooccurence_matrix_to_data(no_classes, no_people, cooccurrence_matrix):
    random_array = np.random.randint(no_classes, size=no_people)
    pref_table = np.zeros((no_people, no_classes), dtype=int)

    for i in range(no_people):
        init_num = random_array[i]
        pref_table[i][init_num] = 1
        for j in range(no_classes):
            if j == init_num: continue
            prob = cooccurrence_matrix[init_num][j]
            rndm = rd.random()
            if rndm < prob:
                pref_table[i][j] = 1
    return pref_table


def cooccurence_matrix_to_data_recursive(no_classes, no_people, cooccurrence_matrix):
    random_array = np.random.randint(no_classes, size=no_people)
    pref_table = np.zeros((no_people, no_classes), dtype=int)

    rem_arr = np.empty(0, dtype=int)
    for i in range(no_people):
        init_num = random_array[i]
        # print(init_num)
        rem_arr = np.append(rem_arr, init_num)
        while rem_arr.size:
            # print(rem_arr)
            elm = rem_arr[-1]
            rem_arr = rem_arr[:-1]
            pref_table[i][elm] = 1
            for j in range(no_classes):
                if pref_table[i][j] == 1: continue
                prob = cooccurrence_matrix[elm][j]
                random = rd.random()
                if random < prob:
                    pref_table[i][j] = 1
                    rem_arr = np.append(rem_arr, j)
    return pref_table



def main():
    no_classes = 5
    no_people = 10


    rand_cooccurrence_matrix = random_cooccurence_matrix(no_classes)
    #print(rand_cooccurrence_matrix)

    pref_table = cooccurence_matrix_to_data(no_classes, no_people, rand_cooccurrence_matrix)
    np.save('prefs', pref_table)


    # pref_table = np.load('prefs.npy')

    # rand_pref_table = random_pref_table(no_classes, no_people)
    # cooccurrence_matrix = data_to_cooccurence_matrix(pref_table)

    print('Priority Table: \n', pref_table.T)
    print('Correlations: \n', np.corrcoef(pref_table.T))
    # print('Coocc: \n', cooccurrence_matrix)

    # Scipy Hierarchical Clustering
    Z = linkage(pref_table.T, 'ward')
    c, coph_dists = cophenet(Z, pdist(pref_table.T))
    print('Linkage: \n', Z)
    print('Cophenetic Correlation Coefficient: ',c)

    # Agglomerative Clustering
    print("\nAgglomerative Clustering")
    model = AggCl(2)
    model.fit_predict(pref_table.T)
    print(model.labels_)
    model = AggCl(3)
    model.fit_predict(pref_table.T)
    print(model.labels_)
    model = AggCl(4)
    model.fit_predict(pref_table.T)
    print(model.labels_)
    model = AggCl(5)
    model.fit_predict(pref_table.T)
    print(model.labels_)

    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(Z)
    plt.show()

    # np_occ = np.array(cooccurrence_matrix)
    # np_avg = (np_occ + np.transpose(np_occ)) / 2

    # print(np_avg)


if __name__ == '__main__':
    main()
