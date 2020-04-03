# Tree to be

import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# copied example, will soon be completely useless

#  A      B   C    D    E   F    G
train_data = np.array([
  [5, 133.5, 27, 284, 638, 31, 220],
  [5, 111.9, 27, 285, 702, 36, 230],
  [5,  99.3, 25, 310, 713, 39, 227],
  [5, 102.5, 25, 311, 670, 34, 218],
  [5, 114.8, 25, 312, 685, 34, 222],
])
# These I just made up
test_data_x = np.array([
  [5, 100.0],
  [5, 105.2],
  [5, 102.7],
  [5, 103.5],
  [5, 120.3],
  [5, 132.5],
  [5, 152.5],
])

x = train_data[:, :2]
y = train_data[:, 2:]
# forest = RandomForestClassifier(n_estimators=20, random_state=1)
# forest.fit(x, y)
# print(forest.predict(test_data_x))
# print(dir(forest))

# ============================== #
#           MAH STUFF            #
# ============================== #

# print(dir(estimator))
# print(dir(estimator.tree_))


def max_coverage(tree_model, test_set, n_leaves=1):
    """ Returns a list of pairs with coverage percentages and indices of
     leaves which covers the most samples from test_set. The parameters are:
    | tree_model: A trained decision tree model.
    | test_set: The set where we want to analyze samples from.
    | n_leaves: The number of best leaves that the function returns.
              Set to 1 by default. """

    # prep for leaf analysis
    leaves_classified = tree_model.apply(test_set)
    n_samples = len(leaves_classified)
    n_nodes = tree_model.tree_.node_count
    leaf_sample_count = [0 for _ in range(n_nodes)]

    # we calculate how many samples are in each leaf
    for sample_id in range(n_samples):
        leaf = leaves_classified[sample_id]
        leaf_sample_count[leaf] += 1

    # TODO: decide: Should a list is_leaf be a parameter of the function???
    # check which leaves have the largest coverage
    first_leaf = leaves_classified[0]
    best_leaves = [(leaf_sample_count[first_leaf], first_leaf)]

    for node in range(n_nodes):
        if len(best_leaves) < n_leaves:
            # we only count leaves that cover at least 1 sample
            if leaf_sample_count[node] > 0:
                best_leaves.append((leaf_sample_count[node], node))
                best_leaves.sort()  # could be moved to optimize a little
        else:
            # compare node with the worst example currently included
            if leaf_sample_count[node] >= best_leaves[-1][0]:
                # switch the last element and sort
                best_leaves[-1] = (leaf_sample_count[node], node)
                best_leaves.sort()

    # returns an ordered list of best leaves containing:
    # coverage percentage and id of leaf
    best_leaves = map(lambda pair: (pair[0]/n_samples, pair[1]), best_leaves)
    best_leaves = list(best_leaves)

    return best_leaves


def path_to(tree_model, goal_node):
    """ Returns the path to a given node in the given tree model.
    | tree_model: the tree that we want to search through.
    | goal_node: id of the node from the given tree that we want a path to. """
    # The returned path is formatted as follows:
    # [(feature, value, bool), ...]
    #
    # for example:
    # [(2, 4.0, 1), (4, -2.5, 0)]
    # represents the conditions:
    # 2nd feature is > 4.0
    # 4th feature is <= -2.5

    # we parse the tree structure as shown in documentation
    children_left = tree_model.tree_.children_left
    children_right = tree_model.tree_.children_right

    # We traverse the tree structure to identify the leaves
    stack = [(0, [0])]  # seed is the root node id
    while len(stack) > 0:
        node_id, path_ids = stack.pop()

        # If we have a test node
        if node_id == goal_node:
            stack = []
        else:
            # if the node is a split node, not a leaf
            if (children_left[node_id] != children_right[node_id]):
                # adding left and right sub-branches
                left_child = children_left[node_id]
                left_path = [id for id in path_ids]
                left_path.append(left_child)
                stack.append((left_child, left_path))

                right_child = children_right[node_id]
                right_path = [id for id in path_ids]
                right_path.append(right_child)
                stack.append((right_child, right_path))

    # we write down the conditions of path nodes
    features = tree_model.tree_.feature
    thresholds = tree_model.tree_.threshold
    len_path = len(path_ids)
    path = []

    # we set up the first entry for the root
    feature = features[0]
    threshold = thresholds[0]
    node_id = 0

    for i in range(1, len_path-1):

        # if the path went to the left
        if children_left[node_id] == path_ids[i]:
            # left turn, i.e. <= than threshold
            path_direction = 0
        else:
            # right turn
            path_direction = 1
        # we add entry for the node before
        path.append((feature, threshold, path_direction))

        # update values to current node
        node_id = path_ids[i]
        feature = features[node_id]
        threshold = thresholds[node_id]

    # add element for last node to path
    if children_left[node_id] == path_ids[-1]:
        # left turn
        path_direction = 0
    else:
        # right turn
        path_direction = 1
    path.append((feature, threshold, path_direction))

    return path


def print_path(path):
    """ Function print_path prints out the given path in a neat form. """
    line_count = 0

    for node in path[:-1]:
        feature, threshold, path_dir = node
        # print out node
        if path_dir == 0:
            print("%s test node: feature %s <= %s"
                  % (line_count * "  ",
                     feature,
                     threshold
                     ))
        else:
            print("%s test node: feature %s > %s"
                  % (line_count * "  ",
                     feature,
                     threshold
                     ))
        line_count += 1
    return None


def encoding_naive(forest, code_size, X_set):
    """ Finds a very naive encoding """
    n_trees = forest.n_estimators

    # For a start we shall find candidates very naively.
    # We search through the trees and look at the leaf with
    # largest coverage in each tree. We take code_size best
    # leaves found this way.
    #
    # Any tree can only have one leaf in the encoding.
    #
    # We save candidates using a list of triples that is
    # code_size long. We keep only the best candidates in it.
    #
    # A candidate is described by:
    # (coverage, tree_id, leaf_id)

    candidates = []

    for i in range(n_trees):
        # find the best leaf in current tree
        tree = forest.estimators_[i]
        cover, candidate_leaf = max_coverage(tree, X_set)[0]

        # if the new candidate is good enough we save it. That is:
        # not enough candidates or it's better than one of the candidates
        if len(candidates) < code_size:
            # we add the new candidate and sort the list
            candidates.append((cover, i, candidate_leaf))
            candidates.sort()
        else:
            if cover > candidates[-1][0]:
                # we replace the last element and sort
                candidates[-1] = (cover, i, candidate_leaf)
                candidates.sort()

    # we return the encoding presented in a readable manner
    # the entries are (coverage, path)
    encoding_paths = []
    for candidate in candidates:
        tree = forest.estimators_[candidate[1]]
        leaf = candidate[2]
        cover = candidate[0]
        path = path_to(tree, leaf)
        encoding_paths.append((cover, path))

    return encoding_paths


def encoding(forest, code_size, X_set):
    """ Finds a naive encoding """
    n_trees = forest.n_estimators

    # We search through the trees and look at the leaf with
    # largest coverage in each tree. We take code_size best
    # leaves found this way.
    #
    # Any tree can only have one leaf in the encoding.
    #
    # We save candidates using a list of triples that is
    # code_size long. We keep only the best candidates in it.
    #
    # A candidate is described by:
    # (coverage, tree_id, leaf_id)

    candidates = []

    for i in range(n_trees):
        # find the best leaf in current tree
        tree = forest.estimators_[i]
        cover, candidate_leaf = max_coverage(tree, X_set)[0]

        # if the new candidate is good enough we save it. That is:
        # not enough candidates or it's better than one of the candidates
        if len(candidates) < code_size:
            # we add the new candidate and sort the list
            candidates.append((cover, i, candidate_leaf))
            candidates.sort()
        else:
            if cover > candidates[-1][0]:
                # we replace the last element and sort
                candidates[-1] = (cover, i, candidate_leaf)
                candidates.sort()
                # TODO: insert in right spot instead of sorting?

    # We have the best leaf from each tree. Now we check the best trees in case
    # they have other leaves which would improve our encoding.
    # This might be unlikely?

    candidate_trees = [cand[1] for cand in candidates]
    for i in range(2, code_size+1):
        tree_id = candidate_trees[-i]
        tree = forest.estimators_[tree_id]  # we go backwards
        # we obtain new candidates from tree
        new_leaves = max_coverage(tree, X_set, n_leaves=i)

        # we skip the best leaf as it's already included
        # TODO: would work better replaced with a while loop
        # a lot of overhead at the moment
        for j in range(1, i):
            coverage, leaf = new_leaves[j]
            if coverage > candidates[-1][0]:
                # we replace the last element and sort
                candidates[-1] = (coverage, i, leaf)
                candidates.sort()

    # we return the encoding presented in a readable manner
    encoding_paths = []
    for candidate in candidates:
        tree = forest.estimators_[candidate[1]]
        leaf = candidate[2]
        cover = candidate[0]
        path = path_to(tree, leaf)
        encoding_paths.append((cover, path))

    return encoding_paths


# ============================== #
#         METRIC & OTHER         #
# ============================== #


def d(v1, v2):
    """ Calculates d_2 distance between v1 and v2 """
    n = len(v1)
    m = len(v2)
    if m == n:
        square_differences = [0 for i in range(n)]
        for i in range(n):
            square_differences[i] = (v1[i] - v2[i])**2
        d_2 = np.sqrt(sum(square_differences))
        return(d_2)
    else:
        print("Vector dimension mismatch")
        return None


def estimate_on_set(X):
    """ Trains a multivariable RF on set X to predict X
     and returns the distance between data and prediction """
    return None


# ============================== #
#            TESTING             #
# ============================== #

def read_set(file_name):
    """ Function read_set reads data from file and saves it into a table.
    | file_name: name of file containing the generated data. """

    with open(file_name, mode='r') as csv_file:
        data_reader = csv.reader(csv_file, delimiter=' ')
        data = []
        for row in data_reader:
            entry = list(map(int, row))
            data.append(entry)

    return data


def test_encoding(file):
    """ Function test_encoding reads data from a file and uses that set
    to train a RF model. Then it finds an encoding of the set from the
    RF model and prints it out. """
    # TODO: add model parameters and encoding type as parameters

    X = read_set(file)
    # train model on set X
    forest = RandomForestClassifier(n_estimators=20, random_state=1, n_jobs=2)
    forest.fit(X, X)

    # currently we arbitrarily choose a code size and use a naive encoding
    code_size = 5
    code = encoding_naive(forest, code_size, X)
    # code = encoding(forest, code_size, X)

    # print out the encoding
    for i in range(len(code)):
        print(" ====== %s. path: ====== " % (i+1))
        cover = code[i][0]
        print(" The leaf of this path covers %s of all examples." % cover)
        new_variable_path = code[i][1]
        print_path(new_variable_path)

    return code

trial_code = test_encoding("generated_set.csv")
