
import numpy as np
import csv
import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error

from keras.layers import Input, Dense
from keras.models import Model


# ============================== #
#       PARAMETER SETTING        #
# ============================== #

# we init the parameters to a standard value

# the threshold that the sum of all coverages in the encoding should exceed
coverage_sum_threshold = 1.8
# default measure used to exclude leaves that are too similar
default_measure_of_diff = 0.1

# random forest parameters
default_n_estimators = 50

# TODO: move this into one of the testing functions for parameter tuning


# ============================== #
#      PRINT OUT & COSMETIC      #
# ============================== #


def print_path(path):
    """ Function print_path prints out the given path in a neat form. """
    line_count = 0

    for node in path:
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


def print_encoding(code, forest):
    """ Prints out the encoding in a neatly arranged form. """

    for i in range(len(code)):
        print(" ====== %s. path: ====== " % (i+1))
        # print out the coverage
        cover = code[i][0]
        print(" The leaf of this path covers %s of all examples." % cover)
        # print out the path
        new_variable_path = code[i][1]
        print_path(new_variable_path)

        # print prediction
        prediction = code[i][2]
        class_keys = list(map(list, forest.classes_))
        prediction_values = convert_labels(prediction, class_keys)

        print(" This path predicts the element is: %s" % prediction_values)
        # print(predictions)
        print()


# ============================== #
#      AUXILLIARY FUNCTIONS      #
# ============================== #


# ===== PATH TRANSFORMATION ===== #


def find_path(children_left, children_right, goal_node):
    """ Returns the path to a goal_node in the given tree model. The connections
        in the tree are given with left_children and right_children lists.
        The path is returned as a list of node ids. """

    # travel through the tree until we hit the goal node
    stack = [(0, [0])]  # seed is the root node id
    while len(stack) > 0:
        node_id, path_ids = stack.pop()

        # If we have a test node
        if node_id == goal_node:
            break
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

    return path_ids


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

    # save the way nodes are connected
    children_left = tree_model.tree_.children_left
    children_right = tree_model.tree_.children_right

    # find which nodes are in the path to goal node
    path_ids = find_path(children_left, children_right, goal_node)

    # we write down the parameters of tree nodes
    features = tree_model.tree_.feature
    thresholds = tree_model.tree_.threshold
    len_path = len(path_ids)
    path = []

    # we set up the first path element for the root
    feature = features[0]
    threshold = thresholds[0]
    node_id = 0

    for i in range(1, len_path):

        # if the path went to the left
        if children_left[node_id] == path_ids[i]:
            # left turn, i.e. <= than threshold
            path_direction = 0
        else:
            # right turn, i.e. > than threshold
            path_direction = 1
        # we add entry for the node before
        path.append((feature, threshold, path_direction))

        # update values to current node
        node_id = path_ids[i]
        feature = features[node_id]
        threshold = thresholds[node_id]

    # We leave out the last node in the path as it is a leaf node
    # and the path only ends there.
    return path


# ======== SET ENCODING ======== #


def check_condition_for_sample(condition, sample):
    """ The function checks whether the given condition holds for the given sample.
    Condition is given as a 'path', sample is a vector. """

    # check if the sample fits the steps of the condition
    checklist = list(map(
        lambda x: sample[x[0]] >= x[1] if x[2] else sample[x[0]] < x[1],
        condition))

    # check if all parts of the condition are true
    fits_condition = all(checklist)

    return fits_condition


def encode_sample(code_paths, sample):
    """ Function encode_sample encodes the given sample with
     the given code of code_paths. """

    # for each code_path check if the sample fits
    x_code = map(lambda x: check_condition_for_sample(x, sample), code_paths)
    # transform to 1/0 instead of True/False
    encoded_sample = list(map(lambda x: 1 if x else 0, x_code))

    return encoded_sample


def encode_set(code, X):
    """ Function encode_set encodes the set X with the given code. """

    # extract the code paths from code
    code_paths = list(map(lambda x: x[1], code))
    # map the samples to encoded samples
    encoded_set = list(map(lambda x: encode_sample(code_paths, x), X))

    return encoded_set


# ==== SIMILARITY MEASURE ==== #


# TODO: maybe delete
def path_to_vector(path, number_of_features):
    """ Converts a list that describes a path to a vector.
    The i-th element of the vector is:
    -1; if the i-th feature is set to 0 in the path
    1; if the i-th feature is set to 1
    0; otherwise """
    
    # create vector
    v = [0 for _ in range(number_of_features)]

    # correct vector entriy for each node in path
    for node in path:
        feature, threshold, path_dir = node
        # TODO: check indices, might be an off by one error
        if path_dir >= 1:
            v[feature] = 1
        else:
            v[feature] = -1
    
    return v


def node_to_vector(path, training_set):
    """ Maps the given path to a vector describing which of the examples
    in the training are described by the leaf at the end of the path. """

    # TODO: check if map alters the original object
    v = list(map(lambda x: check_condition_for_sample(path, x), training_set))

    return v


def code_similarity(x, y, training_set):
    """ The function calculates a 'normalized' similarity
    measure for the two given code entries. """

    # we extract vectors to represent the codes
    # x = (cover, path, prediction)

    v1 = node_to_vector(x[1], training_set)
    v2 = node_to_vector(y[1], training_set)

    # scalar product of encodings
    similarity = np.dot(v1, v2)

    # normalization (untested as of yet)
    samples_covered_by_v1 = sum(v1)
    samples_covered_by_v2 = sum(v2)
    norm = max(samples_covered_by_v1, samples_covered_by_v2)

    similarity_normalized = similarity/norm

    return similarity_normalized


# ====== LEAF PREDICTION ====== #
# TODO: debug predictions!!!!


def leaf_label(tree_model, leaf):
    """ Function leaf_label returns the label of the given leaf, which is the class
    that the tree will predict for the elements that belong to the leaf. """

    values = tree_model.tree_.value
    # transform data from numpy array into list
    leaf_values = list(map(list, values[leaf]))

    # for each feature we find which value is dominant among elements of leaf
    prediction = list(map(return_max_index, leaf_values))

    return prediction


def convert_labels(keys, class_legend):
    """ Function convert_labels transforms a predictions of indices given by
    a tree model to the actual values of the variables.
    The argument class_legend is given by forest.classes_ """

    n = len(class_legend)
    class_values = map(lambda x: class_legend[x][keys[x]], list(range(n)))

    return list(class_values)


# ======== DPLL ======== #


def feature_in_list(fixed_value_list, feature):
    """ The function checks if the list already contains a value
     attributed to the given feature to avoid repeated entries. """

    feature_equality_list = list(map(lambda x: x[0] == feature, fixed_value_list))
    feature_present = any(feature_equality_list)

    return feature_present

def prune_formula(formula):
    """ Removes occurences of False from the formula and 
    removes clauses that contain a True literal. """

    pruned_formula = []

    for clause in formula:
        pruned_clause = []

        # go through literals in the clause
        clause_obsolete = False
        for literal in clause:
            if literal == True:
                clause_obsolete = True
            elif literal != False:
                pruned_clause.append(literal)
            
        # add shortened clause to new formula
        if not clause_obsolete:
            pruned_formula.append(pruned_clause)
    
    return pruned_formula


def choose_literal(formula):
    """ Returns a literal contained in the formula. """

    candidate_literal = False

    for clause in formula:
        for literal in clause:
            candidate_literal = literal

    return candidate_literal


def replace_literal(formula, literal):
    """ Replaces all occurences of the selected literal or its negation
    in the formula with True/False as appropriate. """

    n_clauses = len(formula)
    updated_formula = formula # TODO: deep copy !!!!

    feature = literal[0]
    feature_value = literal[1]

    for i in range(n_clauses):
        clause = formula[i]
        n_literals = len(clause)
        for j in range(n_literals):
            if clause[j] != False and clause[j] != True:
                if feature == clause[j][0]:
                    # we assign a value to this literal
                    if clause[j][1] == feature_value:
                        updated_formula[i][j] = True
                    else:
                        updated_formula[i][j] = False

    return updated_formula


def check_for_unit_clause(formula):
    """ Checks if any clause in the formula contains only one literal. """

    unit_clauses = []
    seen_features = []

    for clause in formula:
        if len(clause) == 1:
            if clause[0] != True and clause[0] != False:
                feature = clause[0][0]
                if feature not in seen_features:
                    unit_clauses.append(clause[0]) 
                    seen_features.append(clause[0][0])

    return unit_clauses


def check_for_pure_literal(formula, n_of_features):
    """ Checks if any literal appears only in negated or 
    non-negated form in the formula. """

    # init a vector to keep track of each feature
    feature_vector = [-1 for _ in range(n_of_features)]

    for clause in formula:
        for literal in clause:
            feature = literal[0]
            bool_value = literal[1]

            if feature_vector[feature] == -1:
                # this is the first assignment for this feature
                feature_vector[feature] = bool_value
            elif feature_vector[feature] != bool_value:
                # this assignment doesn't match the ones before
                # the literal isn't pure
                feature_vector[feature] = -2
    
    # go through the list and check for a pure literal
    pure_candidates = []
    for feature in range(n_of_features):
        bool_value = feature_vector[feature]
        if bool_value == 0 or bool_value == 1:
            # this is a pure literal
            pure_candidates.append((feature, bool_value))

    # we return the result
    return pure_candidates


def dpll(formula, n_of_features, fixed_feature_values):
    """ Performs the DPLL algorithm on the given formula in CNF. """
    # TODO: check that that objects are overwritten correctly

    # TODO: deep copy
    new_formula = copy.deepcopy(formula)
    expanded_fixed_values = copy.deepcopy(fixed_feature_values)

    # check if algorithm is finished
    if len(formula) == 0:
        # we went through the whole formula without finding a contradiction
        # therefore the set of values we found is consistent
        return fixed_feature_values
    else:
        for clause in formula:
            if clause == []:
                # we found a contradiction
                return False

    # check for unit clauses and pure literals
    unit_clauses = check_for_unit_clause(formula)
    pure_literals = check_for_pure_literal(formula, n_of_features)

    # TODO: add a check so we don't add repeated literals
    while unit_clauses != []:
        unit_clause = unit_clauses.pop()
        new_formula = replace_literal(new_formula, unit_clause)
        expanded_fixed_values.append(unit_clause)
    while pure_literals != []:
        pure_literal = pure_literals.pop()
        new_formula = replace_literal(new_formula, pure_literal)
        # check so we don't repeat literals
        if not feature_in_list(expanded_fixed_values, pure_literal[0]):
            expanded_fixed_values.append(pure_literal)

    # prune the formula
    pruned_formula = prune_formula(new_formula)

    # choose a literal and fix its value
    trial_literal = choose_literal(pruned_formula)

    if trial_literal == False:
        # there are no more literals, we call another iteration to give results
        return dpll(pruned_formula, n_of_features, expanded_fixed_values)
    else:
        trial_feature = trial_literal[0]
        # we try to solve with literal value fixed as true
        subformula_positive = pruned_formula + [[(trial_feature, 1)]]
        result = dpll(subformula_positive, n_of_features, expanded_fixed_values)

        if result != False:
            return result
        else:
            # we don't find an answer, fix the value as false
            subformula_negative = pruned_formula + [[(trial_feature, 0)]]
            return dpll(subformula_negative, n_of_features, expanded_fixed_values)


# ======== OTHER ======== #


def find_negation_candidate(encoding, n_of_features):
    """ The function looks through the paths of an encoding and 
    finds a prediction for samples that don't match any of the paths. """

    # we have to look through the negations of the path conditions

    # convert encoding into CNF
    code_paths = list(map(lambda x: x[1], encoding))

    # call DPLL
    fixed_values = dpll(code_paths, n_of_features, [])

    # fill out the non-fixed features with random values
    negation_candidate = np.random.randint(2, size=n_of_features)

    if fixed_values != False:
        for fixed_feature in fixed_values:
            feature = fixed_feature[0]
            value = fixed_feature[1]
            negation_candidate[feature] = value
    
    return negation_candidate


def return_max_index(l):
    """ Placeholder. auxilliary function """

    max_element = max(l)
    i = l.index(max_element)

    return i


def d(v1, v2):
    """ Calculates distance between v1 and v2 """
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


# ============================== #
#       ENCODING FRAMEWORK       #
# ============================== #


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


def find_naive_candidates(forest, n_candidates, X_set):
    """ Function find_naive_candidates looks through a given forest and returns
        the leaves with the largest coverage. It is naive and only looks at
        the best leaf of each tree.
    | forest: The forest that the function searches through.
    | n_candidates: how many best candidates are returned by the function.
    | X_set: the set being studied, the forest should be trained on X_set. """

    n_trees = forest.n_estimators
    candidates = []

    # We search through the trees and look at the leaf with largest coverage
    # in each tree. We take n_candidates best leaves found this way.
    #
    # We save candidates using a list of triples that is
    # n_candidates long. We keep only the best candidates in it.
    #
    # A candidate is described by:
    # (coverage, tree_id, leaf_id)

    # TODO: replace lists with a better structure?
    # TODO: insert in correct place instead of sorting every step?
    for i in range(n_trees):
        # find the best leaf in current tree
        tree = forest.estimators_[i]
        cover, candidate_leaf = max_coverage(tree, X_set)[0]

        # if the new candidate is good enough we save it. That is:
        # not enough candidates or it's better than one of the candidates
        if len(candidates) < n_candidates:
            # we add the new candidate and sort the list
            candidates.append((cover, i, candidate_leaf))
            candidates.sort(reverse=True)
        else:
            if cover > candidates[-1][0]:
                # we replace the last element and sort
                candidates[-1] = (cover, i, candidate_leaf)
                candidates.sort(reverse=True)

    return candidates


def find_different_candidates(forest, n_candidates, X_set, measure_of_difference=default_measure_of_diff):
    """ Function find_different_candidates looks through a given forest and returns
        leaves that are suitable candidates as a basis for encoding. It tries to
        ensure a large enough difference between candidates.
    | forest: The forest that the function searches through.
    | n_candidates: how many best candidates are returned by the function.
    | X_set: the set being studied, the forest should be trained on X_set. """

    n_trees = forest.n_estimators
    candidates = []

    # We search through the trees and look at the leaf with largest coverage
    # in each tree. We take n_candidates best leaves found this way.
    #
    # We save candidates using a list of triples that is
    # n_candidates long. We keep only the best candidates in it.
    #
    # A candidate is described by:
    # (coverage, tree_id, leaf_id)

    # TODO: replace lists with a better structure?
    # TODO: insert in correct place instead of sorting every step?
    for i in range(n_trees):
        # find the best leaf in current tree
        tree = forest.estimators_[i]
        cover, candidate_leaf = max_coverage(tree, X_set)[0]
        # init
        difference_check = True
        # check the similarity with other entries
        new_path = path_to(tree, candidate_leaf)

        # if the new candidate is good enough we save it. That is:
        # not enough candidates or it's better than one of the candidates
        if len(candidates) < n_candidates:

            for candidate in candidates:
                # TODO: this should be rewritten
                # convert format to work for code_similarity()
                old_path = path_to(tree, candidate[1])
                sim = code_similarity((cover, new_path, candidate_leaf), (candidate[0], old_path, candidate[2]), X_set)
                
                if sim > measure_of_difference:
                    difference_check = False

            # we add the new candidate if it isn't too similar and sort the list
            if difference_check:
                candidates.append((cover, i, candidate_leaf))
                candidates.sort(reverse=True)

        else:
            for candidate in candidates:
                # TODO: this should be rewritten
                # convert format to work for code_similarity()
                old_path = path_to(tree, candidate[1])
                sim = code_similarity((cover, new_path, candidate_leaf), (candidate[0], old_path, candidate[2]), X_set)

                if sim > measure_of_difference:
                    difference_check = False

            if cover > candidates[-1][0]:
                if difference_check:
                    # we replace the last element and sort
                    candidates[-1] = (cover, i, candidate_leaf)
                    candidates.sort(reverse=True)

    return candidates


def encoding_naive(forest, code_size, X_set):
    """ The function encoding_naive looks through the given random
        forest model and uses it to (very) naively find an econding.
    | forest: The random forest model to be used.
    | code_size: the size of the returned encoding.
    | X_set: the set being studied, the forest should be trained on X_set. """

    # naively find candidates for the encoding
    candidates = find_different_candidates(forest, code_size, X_set)

    # we return the encoding presented in a readable manner
    # the entries are (coverage, path, prediction)
    encoding_paths = []
    for candidate in candidates:
        cover = candidate[0]
        tree = forest.estimators_[candidate[1]]
        leaf = candidate[2]
        path = path_to(tree, leaf)

        # we add the prediction at the end of the path to output
        # TODO: correct comments to include this part
        prediction = leaf_label(tree, leaf)

        encoding_paths.append((cover, path, prediction))

    return encoding_paths


def encoding(forest, code_size, X_set):
    """ The function encoding_naive looks through the given random
        forest model and uses it to naively find an econding.
    | forest: The random forest model to be used.
    | code_size: the size of the returned encoding.
    | X_set: the set being studied, the forest should be trained on X_set. """

    # naively find candidates for the encoding
    candidates = find_naive_candidates(forest, code_size, X_set)

    # We have the best leaf from each tree. Now we check the best trees
    # in case they have other leaves which would improve our encoding.
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
                candidates.sort(reverse=True)

    # we return the encoding presented in a readable manner
    encoding_paths = []
    for candidate in candidates:
        cover = candidate[0]
        tree = forest.estimators_[candidate[1]]
        leaf = candidate[2]
        path = path_to(tree, leaf)

        # we add the prediction at the end of the path to output
        # TODO: correct comments to include this part
        prediction = leaf_label(tree, leaf)

        encoding_paths.append((cover, path, prediction))

    return encoding_paths


def decode_sample(encoding, n_of_features, encoded_sample):
    """ The function decodes an encoded vector. 
    The encoding and the encoded_sample are parameters. """

    # encoding is a list of elements of the format: (coverage, path, prediction)
    # if the encoded sample fit any of the encoded paths, we just take that prediction
    # NOTE: it might be better to just compute find_negation_candidate once in decode_set
    # TODO: FIX THE GODDAMN PREDICTIONS!!!

    n = len(encoded_sample)

    decoded = False
    for i in range(n):

        # if the sample fits into i-th leaf, we take the saved prediction
        if encoded_sample[i] == 1:
            prediction = encoding[i][2]
            decoded_sample = prediction
            decoded = True

    # if the sample doesn't match any path we have to get crafty
    if not decoded:
        decoded_sample = find_negation_candidate(encoding, n_of_features)

    return decoded_sample


def decode_set(encoding, original_set_dim, encoded_set):
    """ The function decodes an encoded set. 
    The encoding and the encoded set are parameters. """

    # encoding is a list of elements of the format: (coverage, path, prediction)
    # we decode the whole set
    decoded_set = list(map(lambda x: decode_sample(encoding, original_set_dim, x), encoded_set))

    return decoded_set


def encode_with_nn(training_set, test_set, encoding_dim):
    """ Trains a neural network autoencoder on the training set and
    prints out its error on the test set. Saves the results. """
    
    original_dim = len(training_set[0])

    input_data = Input(shape=(original_dim,))

    # "encoded_data" is the encoded representation of the input
    encoded_data = Dense(encoding_dim, activation='relu')(input_data)
    # "decoded_data" is the lossy reconstruction of the input
    decoded_data = Dense(original_dim, activation='sigmoid')(encoded_data)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_data, decoded_data)

    # this model maps an input to its encoded representation
    encoder = Model(input_data, encoded_data)

    # create a placeholder for an encoded input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    # ================================= #
    # We have prepared all of the autoencoder parts

    # we use 'adadelta optimize' and a per-pixel binary crossentropy loss
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    # we train the model for 100 epochs
    autoencoder.fit(training_set, training_set,
                    epochs=100,
                    batch_size=128,
                    shuffle=True) # batch_size / epochs changed

    # encode and decode the data
    encoded_set = encoder.predict(test_set)
    decoded_set = decoder.predict(encoded_set)

    encoded_training_set = encoder.predict(training_set)
    decoded_training_set = decoder.predict(encoded_training_set)

    # calculate RMSE
    rmse = mean_squared_error(test_set, decoded_set, squared=False)

    print("The reconstruction error for the neural network autoencoder is:")
    print(rmse)

    # save/show the results
    # print(decoded_data)

    save_results(test_set, decoded_set, encoded_set, "./results/test_set_nn_results.csv")
    save_results(training_set, decoded_training_set, encoded_training_set, "./results/training_set_nn_results.csv")
    
    return None


# ============================== #
#         READ & WRITE           #
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


def save_results(original_set, decoded_set, encoded_set, file_name):
    """ Save the given results including the original set, the encoded set
    and extra info into a txt file. """
    # TODO: add extra info to save, like recon error, etc.

    with open(file_name, mode='w') as csv_file:
        data_writer = csv.writer(csv_file, delimiter=' ')

        # TODO: add header
        n = len(original_set)
        for i in range(n):
            row = list(original_set[i]) + ['|'] + list(decoded_set[i]) + ['|'] + list(encoded_set[i])
            data_writer.writerow(row)


# ============================== #
#            TESTING             #
# ============================== #


def estimate_reconstruction_error(file, metric):
    """ Trains a multivariable RF on set X to predict X
     and returns the distance between data and prediction.
     Possible metrics are: MSE, RMSE, MAE """
    
    # read file
    X = read_set(file)
    original_set_dim = len(X[0])
    set_size = len(X)

    # split set into two
    training_set_samples = np.random.randint(set_size, size=set_size)
    training_set = []

    test_set_samples = [1 for _ in range(set_size)]
    test_set = []

    for sample_id in training_set_samples:
        # maybe deep copy?
        training_set.append(X[sample_id])
        test_set_samples[sample_id] = 0

    for sample_id in range(set_size):
        if test_set_samples[sample_id] == 1:
            test_set.append(X[sample_id])

    # TODO: use train_set to train, test on test_set
    
    # train model on set X
    forest = RandomForestClassifier(n_estimators=default_n_estimators,
                                    random_state=1,
                                    n_jobs=2)
    forest.fit(test_set, test_set)

    # ===== dynamically select encoding size ===== #

    code_size = 1
    covered_samples_sum = 0 # init, chosen to ensure loop
    # THIS IS VERY INEFFICIENT
    while covered_samples_sum < coverage_sum_threshold:
        forest_encoding = encoding_naive(forest, code_size, X)
        # we want it to cover at least 0.5
        covered_samples_sum = sum(map(lambda x: x[0], forest_encoding))
        # expand code by one
        code_size += 1

    # print out the encoding
    # print_encoding(code, forest)

    # ===== train the neural network ===== #

    encode_with_nn(training_set, test_set, code_size)

    # ===== using the encodings on the test set ===== #

    # encode the set as a test:
    forest_encoded_set = encode_set(forest_encoding, test_set)
    forest_encoded_train_set = encode_set(forest_encoding, training_set)

    # reconstruct the sets
    forest_decoded_set = decode_set(forest_encoding, original_set_dim, forest_encoded_set)
    forest_decoded_train_set = decode_set(forest_encoding, original_set_dim, forest_encoded_train_set)

    # calculate reconstruction error
    
    forest_rmse = mean_squared_error(test_set, forest_decoded_set, squared=False)

    # print out results
    print("The reconstruction error for the forest autoencoder is:")
    print(forest_rmse)

    # save results
    save_results(test_set, forest_decoded_set, forest_encoded_set, "./results/test_set_forest_results.csv")
    save_results(training_set, forest_decoded_train_set, forest_encoded_train_set, "./results/training_set_forest_results.csv")

    return None


def test_encoding(file):
    """ Function test_encoding reads data from a file and uses that set
    to train a RF model. Then it finds an encoding of the set from the
    RF model and prints it out. """
    # TODO: add model parameters and encoding type as parameters

    X = read_set(file)
    # train model on set X
    # We can use parameters max_leaf_nodes and min_impurity_decrease
    # to decrease the number of leaves in the tree and therefore increase
    # the number of samples covered by a leaf.
    forest = RandomForestClassifier(n_estimators=default_n_estimators,
                                    # max_leaf_nodes=25,
                                    # min_impurity_decrease=0.003,
                                    random_state=1,
                                    n_jobs=2)
    forest.fit(X, X)

    # currently we arbitrarily choose a code size and use a naive encoding
    # TODO: pca to select code_size?
    # code_size = 5
    # code = encoding_naive(forest, code_size, X)
    # code = encoding(forest, code_size, X)

    # print_encoding(code, forest)

    # ===== 2nd option: dynamically select encoding size ===== #

    code_size = 1
    covered_samples_sum = 0 # init, chosen to ensure loop
    # THIS IS VERY INEFFICIENT
    while covered_samples_sum < coverage_sum_threshold:
        code = encoding_naive(forest, code_size, X)
        # TODO: check that covering is large enough
        # we want it to cover at least 0.5
        covered_samples_sum = sum(map(lambda x: x[0], code))
        # expand code by one
        code_size += 1

    # print out the encoding
    print_encoding(code, forest)

    # encode the set as a test:
    # Y = encode_set(code, X)
    # print(Y)

    # TEST NEURAL NETWORK, TO BE REMOVED
    # encode_with_nn(file, 5)

    return code


# COMPUTE THE ENCODING

# trial_code = test_encoding("generated_set.csv")
# trial_code = test_encoding("latent-space.csv")

# TEST RECON ERROR

estimate_reconstruction_error("latent-space.csv", "rmse")

# TEST THE DPLL IMPLEMENTATION

# CNF = [[(0,1),(1,1)], 
#     [(1,1),(2,0),(3,1)],
#     [(0,0),(1,0)],
#     [(0,0),(2,0),(3,0)],
#     [(0,1)]]

# cnf_solution = dpll(CNF, 4, [])
# print(cnf_solution)
