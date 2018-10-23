import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import pdist

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


class TreeNode():
    def __init__(self, value, left, right, left_depth, right_depth):
        super(TreeNode, self).__init__()
        self.value = value
        self.left = left
        self.right = right
        self.left_depth = left_depth
        self.right_depth = right_depth
        self.depth = min(left_depth, right_depth)
        self.count = len(value)

    def __str__(self, level = 0):
        all = False
        ret = "\t" * level + str(self.value) + " count: " + str(self.count) + " level: " + str(level) + " depth: " + "(" + str(self.left_depth) + "," + str(self.right_depth)+ ")" + "\n"
        if all:
            if not isinstance(self.left, int):
                ret += self.left.__str__(level + 1)
            else:
                ret += "\t" * (level + 1) + str(self.left) + "\n"
            if not isinstance(self.right, int):
                ret += self.right.__str__(level + 1)
            else:
                ret += "\t" * (level + 1) + str(self.right) + "\n"
        else:
            if not isinstance(self.left, int):
                ret += self.left.__str__(level + 1)
            if not isinstance(self.right, int):
                ret += self.right.__str__(level + 1)
        return ret

    def __repr__(self):
        return str(self.value)


def generate_preference_table(classes, samples):
    return np.random.choice(2, (classes, samples), p=[0.7, 0.3])


def simplify_linkage(Z, classes):
    clusters = []
    last_2 = 0
    for i in range(len(Z)):
        if Z[i][1] < classes:
            clusters.append((int(Z[i][0]), int(Z[i][1])))
            last_2 = (int(Z[i][0]), int(Z[i][1]))
        else:
            if Z[i][0] >= classes:
                index_0 = int(Z[i][0] - classes)
                index_1 = int(Z[i][1] - classes)
                clusters.append((clusters[index_0] + clusters[index_1]))
                last_2 = (clusters[index_0],clusters[index_1])
            else:
                index = int(Z[i][1] - classes)
                clusters.append((clusters[index] + (int(Z[i][0]),)))
                last_2 = (clusters[index],(int(Z[i][0])))
    return clusters, last_2


def linkage_to_tree(Z, classes):
    nodes = []
    for i in range(len(Z)):
        if Z[i][1] < classes:
            nodes.append(TreeNode((int(Z[i][0]), int(Z[i][1])), int(Z[i][0]), int(Z[i][1]), 0, 0))
        else:
            if Z[i][0] >= classes:
                index_0 = int(Z[i][0] - classes)
                index_1 = int(Z[i][1] - classes)
                nodes.append(TreeNode((nodes[index_0].value + nodes[index_1].value), nodes[index_0], nodes[index_1], nodes[index_0].depth + 1, nodes[index_1].depth + 1))
            else:
                index = int(Z[i][1] - classes)
                nodes.append(TreeNode((nodes[index].value + (int(Z[i][0]),)), nodes[index], int(Z[i][0]), nodes[index].depth + 1, 0))
    return nodes[-1]

def generate(classes, samples, resume=False):
    if resume:
        preference_table = np.load('preference_table.npy')
    else:
        preference_table = generate_preference_table(classes, samples)
        np.save('preference_table', preference_table)

    Z = linkage(preference_table, 'ward')
    L, last_2 = simplify_linkage(Z, classes)
    RootNode = linkage_to_tree(Z, classes)

    return RootNode

def main():
    np.set_printoptions(linewidth=320)

    classes = 10
    samples = 80

    preference_table = generate_preference_table(classes, samples)
    np.save('preference_table', preference_table)
    print('Preference Table: \n', preference_table)
    print('Correlations: \n', np.corrcoef(preference_table))

    Z = linkage(preference_table, 'ward')
    L, last_2 = simplify_linkage(Z, classes)
    RootNode = linkage_to_tree(Z, classes)
    c, coph_dists = cophenet(Z, pdist(preference_table))
    print('Linkage: \n', Z)
    print('Simplified Linkage: \n', L)
    print('Last 2: \n', last_2)
    print('Root Node: \n', RootNode)
    # print('Root Node contd: \n', RootNode.left, RootNode.right, RootNode.left.left, RootNode.left.right, RootNode.right.left, RootNode.right.right)
    print('Cophenetic Correlation Coefficient: ', c)

    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(Z)
    plt.show()

if __name__ == '__main__':
    main()
