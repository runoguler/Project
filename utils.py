import numpy as np
import os
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import pdist

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from torch.utils.data.sampler import Sampler


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


class IndexSampler(Sampler):
    def __init__(self, indices):
        super().__init__(indices)
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def binarify_2d(arr):
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            arr[i][j] = 1 if arr[i][j] > 1 else 0
    return arr


def generate_hierarchy_with_cooccurrence(classes, n_type=10, load=False, with_distribution=False, load_gen_users=True, print_types=False, start=2, end_not_inc=4, draw=False):
    all_classes_used_check = [False] * classes
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

    for type in user_types:
        for i in range(len(type)):
            if not all_classes_used_check[i] and type[i] == 200:
                all_classes_used_check[i] = True
    count = 0
    for boo in all_classes_used_check:
        if not boo:
            count += 1
    if count != 0:
        print('{} out of {} are not used'.format(count, len(all_classes_used_check)))

    if not with_distribution:
        if load and load_gen_users and os.path.isfile('tree_gen_users.npy'):
            all_users = np.load('tree_gen_users.npy')
            print("Generating Tree Users Load Successful!")
        else:
            no_of_users_for_each_type = 50
            sample_for_each_user = 250
            all_users = []
            for user_type in user_types:
                for _ in range(no_of_users_for_each_type):
                    temp = [0] * classes
                    sample = np.random.choice(len(user_type), sample_for_each_user, p=np.random.dirichlet(user_type, 1)[0])
                    for i in sample:
                        temp[i] = 1
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

    if draw:
        dendrogram(Q)
        plt.show()

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
    '''
    for i in test_user_types:
        temp = []
        for j in range(len(user_types[i])):
            if user_types[i][j] == 200:
                temp.append(j)
        print(temp)
    '''
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


def main():
    np.set_printoptions(linewidth=320)

    classes = 10
    num_user_types = 10
    start, end = 2, 4
    from_distribution = True

    root = generate_hierarchy_with_cooccurrence(classes, n_type=num_user_types, load=False, load_gen_users=False, with_distribution=from_distribution, print_types=False, start=start, end_not_inc=end, draw=True)
    print(len(root.left.value))
    print(len(root.right.value))
    print(root.left.value)
    print(root.right.value)
    #root = generate_hierarchy_from_type_distribution(classes, n_type=10, load=True)
    # print(root)
    test_users = generate_users(10, 20, load=False)
    print(test_users)


if __name__ == '__main__':
    main()
