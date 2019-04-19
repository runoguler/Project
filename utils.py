import numpy as np
import os
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
import visdom

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from torch.utils.data.sampler import Sampler
import torch


class TreeNode():
    def __init__(self, value, left, right, left_depth, right_depth):
        super(TreeNode, self).__init__()
        self.value = value
        self.left = left
        self.right = right
        self.left_depth = left_depth
        self.right_depth = right_depth
        self.depth = min(left_depth, right_depth) + 1
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


class Visualizations:
    def __init__(self, env_name='main'):
        self.env_name = env_name
        self.vis = visdom.Visdom(env=self.env_name)
        self.win_loss = None
        self.win_acc = None

    def plot_loss(self, data, step, name='init', xlabel='Epochs', ylabel='Loss', new_plot=False):
        if new_plot:
            self.win_loss = None
        self.win_loss = self.vis.line(
            [data],
            [step],
            win=self.win_loss,
            name=name,
            update='append' if self.win_loss else None,
            opts=dict(
                xlabel=xlabel,
                ylabel=ylabel
            )
        )

    def plot_acc(self, data, step, name='init', xlabel='Epochs', ylabel='Accuracy', new_plot=False):
        if new_plot:
            self.win_acc = None
        self.win_acc = self.vis.line(
            [data],
            [step],
            win=self.win_acc,
            name=name,
            update='append' if self.win_acc else None,
            opts=dict(
                xlabel=xlabel,
                ylabel=ylabel
            )
        )


class IndexSampler(Sampler):
    def __init__(self, indices):
        super().__init__(indices)
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def generate_preference_table(classes, samples, prob=0.3):
    return np.random.choice(2, (classes, samples), p=[(1-prob), prob])


def generate_preference_table_new(classes, samples, n_type=10, load_types=True, load_preferences=False):
    if load_preferences:
        return np.load('preference_table.npy')
    if load_types:
        user_types = np.load('user_types.npy')
    else:
        user_types = np.empty((0, classes))
        for _ in range(n_type):
            rands = np.random.random(classes)
            # rands /= rands.sum()
            user_types = np.vstack((user_types, rands))

        np.save('user_types', user_types)

    selected_types = np.random.randint(0, len(user_types), samples)

    preferences = np.empty((0, classes), dtype=int)
    for type_index in selected_types:
        preferences = np.vstack((preferences, np.random.binomial(1, user_types[type_index], classes)))
    return preferences.T


def binarify_2d(arr):
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            arr[i][j] = 1 if arr[i][j] > 1 else 0
    return arr


def generate_hierarchy_with_cooccurrence(classes, n_type=10, load=False, with_distribution=False, load_gen_users=True, print_types=False, start=2, end_not_inc=4):
    # all_classes_used_check = [False] * classes
    if load and os.path.isfile('user_types.npy'):
        user_types = np.load('user_types.npy')
        print("User Types Load Successful!")
    else:
        num_pref = np.random.randint(start, end_not_inc, n_type)
        user_types = np.empty((0, classes), dtype=int)
        for i in range(n_type):
            preffed = np.random.choice(classes, num_pref[i], replace=False)
            u_type = np.ones(classes, dtype=int)
            for j in preffed:
                u_type[j] = 200
            user_types = np.vstack((user_types, u_type))
        np.save('user_types', user_types)

    if print_types:
        print(user_types)
    '''
    for u in user_types:
        for i in u:
            if not all_classes_used_check[i] and i == 200:
                all_classes_used_check[i] = True
    # print(all_classes_used_check)
    for boo in all_classes_used_check:
        if not boo:
            print('All classes not used')
            break
    '''
    if not with_distribution:
        if load and load_gen_users and os.path.isfile('tree_gen_users.npy'):
            all_users = np.load('tree_gen_users.npy')
            print("Generating Tree Users Load Successful!")
        else:
            no_of_users_for_each_type = 20
            sample_for_each_user = 100
            all_users = []
            for user_type in user_types:
                for _ in range(no_of_users_for_each_type):
                    temp = [0] * classes
                    sample = np.random.choice(len(user_type), sample_for_each_user, p=np.random.dirichlet(user_type, 1)[0])
                    for i in sample:
                        temp[i] += 1
                    all_users.append(temp)
            all_users = np.array(all_users, dtype=int)
            np.save('tree_gen_users', all_users)
        co_mat = all_users.T.dot(all_users)
    else:
        user_types = binarify_2d(user_types)
        co_mat = user_types.T.dot(user_types)
    np.fill_diagonal(co_mat, 0)
    # print(co_mat)

    dist = 1 - co_mat/co_mat.max()
    # print(dist)

    i, j = np.triu_indices(dist.shape[0], k=1)
    p_dist = dist[i, j]
    Q = linkage(p_dist, 'ward', 'precomputed')

    # dendrogram(Q)
    # plt.show()

    '''
    num_users_for_each_type = 10
    length_of_each_user_sample = 100
    all_tests = np.empty((0, length_of_each_user_sample))
    for user_type in user_types:
        for i in range(num_users_for_each_type):
            test_user = np.random.choice(len(user_type), length_of_each_user_sample, p=np.random.dirichlet(user_type, 1)[0])
            all_tests = np.vstack((all_tests, test_user))
    print(user_types)
    '''

    RootNode = linkage_to_tree(Q, classes)
    return RootNode


def generate_hierarchy_from_type_distribution(classes, n_type=10, load=False, print_types=False, start=2, end_not_inc=4):
    if load and os.path.isfile('user_types.npy'):
        user_types = np.load('user_types.npy')
        print("User Types Load Successful!")
    else:
        num_pref = np.random.randint(start, end_not_inc, n_type)
        user_types = np.empty((0, classes), dtype=int)
        for i in range(n_type):
            preffed = np.random.choice(classes, num_pref[i], replace=False)
            u_type = np.ones(classes, dtype=int)
            for j in preffed:
                u_type[j] = 200
            user_types = np.vstack((user_types, u_type))
        print(user_types)
        np.save('user_types', user_types)
    Z = linkage(user_types.T, 'ward')

    if print_types:
        print(user_types)

    '''
    T = linkage(user_types.T, 'single')
    X = linkage(user_types.T, 'average')
    Y = linkage(user_types.T, 'weighted')
    Z = linkage(user_types.T, 'ward')
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    dendrogram(T, ax=axes[0])
    dendrogram(X, ax=axes[1])
    dendrogram(Y, ax=axes[2])
    dendrogram(Z, ax=axes[3])
    plt.show()
    '''

    # dendrogram(Z)
    # plt.show()

    RootNode = linkage_to_tree(Z, classes)
    return RootNode


# if there is a memory problem with this function, use generate_users_memoryless() instead
def generate_users(num_users, num_samples, load=False):
    if load and os.path.isfile('test_scenario_users.npy'):
        print("Test Scenario Users Load Successful!")
        return np.load('test_scenario_users.npy')
    user_types = np.load('user_types.npy')
    test_user_types = np.random.randint(len(user_types), size=num_users)
    #print(test_user_types)
    test_users = np.empty((0, num_samples), dtype=int)
    for user_type in test_user_types:
        test = np.random.choice(len(user_types[user_type]), num_samples, p=np.random.dirichlet(user_types[user_type], 1)[0])
        test_users = np.vstack((test_users, test))
    np.save('test_scenario_users', test_users)
    return test_users


# if there is a memory problem with generate_users(), use this instead
def generate_users_memoryless(num_users, num_samples, seed=123123):
    np.random.seed(seed)
    user_types = np.load('user_types.npy')
    test_user_types = np.random.randint(len(user_types), size=num_users)
    for user_type in test_user_types:
        test_user = np.random.choice(len(user_types[user_type]), num_samples, p=np.random.dirichlet(user_types[user_type], 1)[0])
        yield test_user


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
            nodes.append(TreeNode((int(Z[i][0]), int(Z[i][1])), int(Z[i][0]), int(Z[i][1]), -1, -1))
        else:
            if Z[i][0] >= classes:
                index_0 = int(Z[i][0] - classes)
                index_1 = int(Z[i][1] - classes)
                nodes.append(TreeNode((nodes[index_0].value + nodes[index_1].value), nodes[index_0], nodes[index_1], nodes[index_0].depth, nodes[index_1].depth))
            else:
                index = int(Z[i][1] - classes)
                nodes.append(TreeNode((nodes[index].value + (int(Z[i][0]),)), nodes[index], int(Z[i][0]), nodes[index].depth, -1))
    return nodes[-1]


def generate(classes, samples, load=False, prob=0.3):
    if load:
        preference_table = np.load('preference_table.npy')
    else:
        preference_table = generate_preference_table(classes, samples, prob)
        np.save('preference_table', preference_table)

    Z = linkage(preference_table, 'ward')
    RootNode = linkage_to_tree(Z, classes)

    return RootNode


def main():
    np.set_printoptions(linewidth=320)

    classes = 365
    samples = 200

    root = generate_hierarchy_with_cooccurrence(classes, n_type=1000, load=False, load_gen_users=False, with_distribution=True, print_types=False, start=2, end_not_inc=365)
    print(root.left.value)
    print(root.right.value)
    #root = generate_hierarchy_from_type_distribution(classes, n_type=10, load=True)
    # print(root)
    # print(generate_users(10, 20, load=False))
    exit()

    # preference_table = generate_preference_table(classes, samples, prob=0.3)
    preference_table = generate_preference_table_new(classes, samples, load_types=True, load_preferences=True)

    np.save('preference_table', preference_table)
    print('Preference Table: \n', preference_table)
    print('Correlations: \n', np.corrcoef(preference_table))

    Z = linkage(preference_table, 'ward')
    RootNode = linkage_to_tree(Z, classes)
    c, coph_dists = cophenet(Z, pdist(preference_table))
    print('Linkage: \n', Z)
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
