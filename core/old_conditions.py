#!/usr/bin/env python
# TEXT-FILE WITH THE CONDITIONS FOR ALL THE METHODS
# REFERENCE IS http://scikit-learn.org
# #################################################
import settings
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
# ----------------------------------------------------------------------------------------------------------------------
# ENSEMBLE METHODS BASED ON DECISION TREES
# Two averaging algorithms based on randomized decision trees: the RandomForest algorithm and the Extra-Trees method.
# Both algorithms are perturb-and-combine techniques specifically designed for trees.
# This means a diverse set of classifiers is created by introducing randomness in the classifier construction.
# The prediction of the ensemble is given as the averaged prediction of the individual classifiers.
# ----------------------------------------------------------------------------------------------------------------------
# RANDOM FOREST (RF)
# In random forests, each tree in the ensemble is built from a sample drawn with replacement (i.e., a bootstrap sample)
# from the training set. In addition, when splitting a node during the construction of the tree, the chosen split
# is no longer the best split among all features. Instead, the split that is picked is the best split among a random
# subset of the features. As a result of this randomness, the bias of the forest usually slightly increases (with
# respect to the bias of a single non-random tree) but, due to averaging, its variance also decreases, usually more than
# compensating for the increase in bias, hence yielding an overall better model.
# ----------------------------------------------------------------------------------------------------------------------
# In contrast to the original publication, the scikit-learn implementation combines classifiers by averaging their
# probabilistic prediction, instead of letting each classifier vote for a single class.
# ----------------------------------------------------------------------------------------------------------------------
# EXTRA-TREE (ET)
# In extremely randomized trees, randomness goes one step further in the way splits are computed. As in random forests,
# a random subset of candidate features is used, but instead of looking for the most discriminative thresholds,
# thresholds are drawn at random for each candidate feature and the best of these randomly-generated thresholds is
# picked as the splitting rule. This usually allows to reduce the variance of the model a bit more, at the expense
# of a slightly greater increase in bias.
# ----------------------------------------------------------------------------------------------------------------------
# PARAMETERS (see details for the methods below)
# The main parameters to adjust when using these methods are n_estimators and max_features.
# The former is the number of trees in the forest. The larger the better, but also the longer it will take to compute.
# In addition, note that results will stop getting significantly better beyond a critical number of trees.
# The latter is the size of the random subsets of features to consider when splitting a node.
# The lower the greater the reduction of variance, but also the greater the increase in bias.
# Empirical good default values are max_features=n_features for regression problems, and max_features=sqrt(n_features)
# for classification tasks (where n_features is the number of features in the data).
# Good results are often achieved when setting max_depth=None in combination with min_samples_split=1
# (i.e., when fully developing the trees). Bear in mind though that these values are usually not optimal,
# and might result in models that consume a lot of ram. The best parameter values should always be cross-validated.
# In addition, note that in random forests, bootstrap samples are used by default (bootstrap=True) while the default
# strategy for extra-trees is to use the whole dataset (bootstrap=False).
# When using bootstrap sampling the generalization error can be estimated on the left out or out-of-bag samples.
# This can be enabled by setting oob_score=True.
# ----------------------------------------------------------------------------------------------------------------------


parameters = {
    # DECISION TREE (DT)
    # criterion : string, optional (default="gini")
    #     The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity and
    #     "entropy" for the information gain.
    #
    # splitter : string, optional (default="best")
    #     The strategy used to choose the split at each node. Supported strategies are "best" to choose the best split
    #     and "random" to choose the best random split.
    #
    # max_features : int, float, string or None, optional (default=None)
    #     The number of features to consider when looking for the best split:
    #         If int, then consider max_features features at each split.
    #         If float, then max_features is a percentage and int(max_features * n_features) features are considered
    # at each split.
    #         If "auto", then max_features=sqrt(n_features).
    #         If "sqrt", then max_features=sqrt(n_features).
    #         If "log2", then max_features=log2(n_features).
    #         If None, then max_features=n_features.
    #     Note: the search for a split does not stop until at least one valid partition of the node samples is found,
    #     even if it requires to effectively inspect more than max_features features.
    #
    # max_depth : int or None, optional (default=None)
    #     The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves
    #     contain less than min_samples_split samples. Ignored if max_leaf_nodes is not None.
    #
    # min_samples_split : int, optional (default=2)
    #     The minimum number of samples required to split an internal node.
    #
    # min_samples_leaf : int, optional (default=1)
    #     The minimum number of samples required to be at a leaf node.
    #
    # min_weight_fraction_leaf : float, optional (default=0.)
    #     The minimum weighted fraction of the input samples required to be at a leaf node.
    #
    # max_leaf_nodes : int or None, optional (default=None)
    #     Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity.
    #     If None then unlimited number of leaf nodes. If not None then max_depth will be ignored.
    #
    # class_weight : dict, list of dicts, "balanced" or None, optional (default=None)
    #     Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to
    #     have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.
    #     The "balanced" mode uses the values of y to automatically adjust weights inversely proportional to class
    #     frequencies in the input data as n_samples / (n_classes * np.bincount(y))
    #     For multi-output, the weights of each column of y will be multiplied.
    #     Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight
    #     is specified.
    #
    # random_state : int, RandomState instance or None, optional (default=None)
    #     If int, random_state is the seed used by the random number generator; If RandomState instance, random_state
    #     is the random number generator; If None, the random number generator is the RandomState instance used by np.random
    'DT': {
        'criterion': ['gini', 'entropy'],
        'splitter': ['random'],      # ['best', 'random']
        'max_features': ['auto'],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'min_weight_fraction_leaf': [0.0],
        'max_leaf_nodes': [None],
        'random_state': [999],
    },

    # RANDOM FOREST (RF)
    # n_estimators : integer, optional (default=10)
    #     The number of trees in the forest.
    #
    # criterion : string, optional (default="gini")
    #     The function to measure the quality of a split. Supported criteria are
    #     "gini" for the Gini impurity and "entropy" for the information gain.
    #     Note: this parameter is tree-specific.
    #
    # max_features : int, float, string or None, optional (default="auto")
    #     The number of features to consider when looking for the best split:
    #         If int, then consider max_features features at each split.
    #         If float, then max_features is a percentage and int(max_features * n_features) features are considered
    #         at each split.
    #         If "auto", then max_features=sqrt(n_features).
    #         If "sqrt", then max_features=sqrt(n_features) (same as "auto").
    #         If "log2", then max_features=log2(n_features).
    #         If None, then max_features=n_features.
    #     Note: the search for a split does not stop until at least one valid partition
    #     of the node samples is found, even if it requires to effectively inspect
    #     more than max_features features.
    #     Note: this parameter is tree-specific.
    #
    # max_depth : integer or None, optional (default=None)
    #     The maximum depth of the tree. If None, then nodes are expanded until all
    #     leaves are pure or until all leaves contain less than min_samples_split
    #     samples. Ignored if max_leaf_nodes is not None.
    #     Note: this parameter is tree-specific.
    #
    # min_samples_split : integer, optional (default=2)
    #     The minimum number of samples required to split an internal node.
    #     Note: this parameter is tree-specific.
    #
    # min_samples_leaf : integer, optional (default=1)
    #     The minimum number of samples in newly created leaves.
    #     A split is discarded if after the split, one of the leaves would contain
    #     less then min_samples_leaf samples.
    #     Note: this parameter is tree-specific.
    #
    # min_weight_fraction_leaf : float, optional (default=0.)
    #     The minimum weighted fraction of the input samples required to be at a
    #     leaf node.
    #     Note: this parameter is tree-specific.
    #
    # max_leaf_nodes : int or None, optional (default=None)
    #     Grow trees with max_leaf_nodes in best-first fashion.
    #     Best nodes are defined as relative reduction in impurity. If None then
    #     unlimited number of leaf nodes. If not None then max_depth will be ignored.
    #     Note: this parameter is tree-specific.
    #
    # bootstrap : boolean, optional (default=True)
    #     Whether bootstrap samples are used when building trees.
    #
    # oob_score : bool
    #     Whether to use out-of-bag samples to estimate the generalization error.
    #
    # n_jobs : integer, optional (default=1)
    #     The number of jobs to run in parallel for both fit and predict.
    #     If -1, then the number of jobs is set to the number of cores.
    #
    # random_state : int, RandomState instance or None, optional (default=None)
    #     If int, random_state is the seed used by the random number generator;
    #     If RandomState instance, random_state is the random number generator;
    #     If None, the random number generator is the RandomState instance used by np.random.
    #
    # verbose : int, optional (default=0)
    #     Controls the verbosity of the tree building process.
    #
    # class_weight : dict, list of dicts, "balanced", "balanced_subsample" or None, optional
    #     Weights associated with classes in the form {class_label: weight}.
    #     If not given, all classes are supposed to have weight one.
    #     For multi-output problems, a list of dicts can be provided in the same order as the columns of y.
    #     The "balanced" mode uses the values of y to automatically adjust weights inversely proportional to class
    #     frequencies in the input data as n_samples / (n_classes * np.bincount(y))
    #     The "balanced_subsample" mode is the same as "balanced" except that weights are computed based on the
    #     bootstrap sample for every tree grown.
    #     For multi-output, the weights of each column of y will be multiplied.
    #     Note that these weights will be multiplied with sample_weight
    #     (passed through the fit method) if sample_weight is specified.
    'RF': {
        'n_estimators': [2, 15, 100],
        'criterion': ['gini', 'entropy'],
        'max_features': ['auto'],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'min_weight_fraction_leaf': [0.0],
        'max_leaf_nodes': [None],
        'bootstrap': [True],
        'oob_score': [False],
        'n_jobs': [1],
        'random_state': [999],
    },

    # EXTRA-TREE (ET)
    #     n_estimators : integer, optional (default=10)
    #     The number of trees in the forest.
    #
    # criterion : string, optional (default="gini")
    #     The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity and
    #     "entropy" for the information gain. Note: this parameter is tree-specific.
    #
    # max_features : int, float, string or None, optional (default="auto")
    #     The number of features to consider when looking for the best split:
    #         If int, then consider max_features features at each split.
    #         If float, then max_features is a percentage and int(max_features * n_features) features are considered at
    #         each split.
    #         If "auto", then max_features=sqrt(n_features).
    #         If "sqrt", then max_features=sqrt(n_features).
    #         If "log2", then max_features=log2(n_features).
    #         If None, then max_features=n_features.
    #     Note: the search for a split does not stop until at least one valid partition of the node samples is found,
    #     even if it requires to effectively inspect more than max_features features.
    #     Note: this parameter is tree-specific.
    #
    # max_depth : integer or None, optional (default=None)
    #     The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves
    #     contain less than min_samples_split samples. Ignored if max_leaf_nodes is not None.
    #     Note: this parameter is tree-specific.
    #
    # min_samples_split : integer, optional (default=2)
    #     The minimum number of samples required to split an internal node. Note: this parameter is tree-specific.
    #
    # min_samples_leaf : integer, optional (default=1)
    #     The minimum number of samples in newly created leaves. A split is discarded if after the split, one of the
    #     leaves would contain less then min_samples_leaf samples. Note: this parameter is tree-specific.
    #
    # min_weight_fraction_leaf : float, optional (default=0.)
    #     The minimum weighted fraction of the input samples required to be at a leaf node.
    #     Note: this parameter is tree-specific.
    #
    # max_leaf_nodes : int or None, optional (default=None)
    #     Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity.
    #     If None then unlimited number of leaf nodes. If not None then max_depth will be ignored.
    #     Note: this parameter is tree-specific.
    #
    # bootstrap : boolean, optional (default=False)
    #     Whether bootstrap samples are used when building trees.
    #
    # oob_score : bool
    #     Whether to use out-of-bag samples to estimate the generalization error.
    #
    # n_jobs : integer, optional (default=1)
    #     The number of jobs to run in parallel for both fit and predict. If -1, then the number of jobs is set to the
    #     number of cores.
    #
    # random_state : int, RandomState instance or None, optional (default=None)
    #     If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is
    #     the random number generator; If None, the random number generator is the RandomState instance used by np.random.
    #
    # class_weight : dict, list of dicts, "balanced", "balanced_subsample" or None, optional
    #     Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to
    #     have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.
    #     The "balanced" mode uses the values of y to automatically adjust weights inversely proportional to class
    #     frequencies in the input data as n_samples / (n_classes * np.bincount(y))
    #     The "balanced_subsample" mode is the same as "balanced" except that weights are computed based on the
    #     bootstrap sample for every tree grown.
    #     For multi-output, the weights of each column of y will be multiplied.
    #     Note that these weights will be multiplied with sample_weight (passed through the fit method) if
    #     sample_weight is specified.
     'ET': {
        'n_estimators': [10],
        'criterion': ['gini', 'entropy'],
        'max_features': ['auto'],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'min_weight_fraction_leaf': [0.0],
        'max_leaf_nodes': [None],
        'bootstrap': [False],
        'oob_score': [False],
        'n_jobs': [1],
        'random_state': [999],
        'class_weight': [None],
    },


    # ADA BOOSTING (AB)
    # base_estimator : object, optional (default=DecisionTreeClassifier)
    #     The base estimator from which the boosted ensemble is built. Support for sample weighting is required,
    #     as well as proper classes_ and n_classes_ attributes.
    #
    # n_estimators : integer, optional (default=50)
    #     The maximum number of estimators at which boosting is terminated. In case of perfect fit,
    #     the learning procedure is stopped early.
    #
    # learning_rate : float, optional (default=1.)
    #     Learning rate shrinks the contribution of each classifier by learning_rate.
    #     There is a trade-off between learning_rate and n_estimators.
    #
    # algorithm : {"SAMME", "SAMME.R"}, optional (default="SAMME.R")
    #     If SAMME.R then use the SAMME.R real boosting algorithm. base_estimator must support calculation of
    #     class probabilities. If SAMME then use the SAMME discrete boosting algorithm. The SAMME.R algorithm
    #     typically converges faster than SAMME, achieving a lower test error with fewer boosting iterations.
    #
    # random_state : int, RandomState instance or None, optional (default=None)
    #     If int, random_state is the seed used by the random number generator; If RandomState instance, random_state
    #     is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
    'AB': {
        'base_estimator': [tree.DecisionTreeClassifier()],
        'n_estimators': [10, 50],
        'learning_rate': [1.0],
        'algorithm': ['SAMME.R', 'SAMME'],
        'random_state': [999],
    },


    # GRADIENT BOOSTING (GB)
    # loss : {"deviance", "exponential"}, optional (default="deviance")
    #     loss function to be optimized. "deviance" refers to deviance (= logistic regression) for classification
    #     with probabilistic outputs. For loss "exponential" gradient boosting recovers the AdaBoost algorithm.
    #
    # learning_rate : float, optional (default=0.1)
    #     learning rate shrinks the contribution of each tree by learning_rate.
    #     There is a trade-off between learning_rate and n_estimators.
    #
    # n_estimators : int (default=100)
    #     The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number
    #     usually results in better performance.
    #
    # max_depth : integer, optional (default=3)
    #     maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree.
    #     Tune this parameter for best performance; the best value depends on the interaction of the input variables.
    #     Ignored if max_leaf_nodes is not None.
    #
    # min_samples_split : integer, optional (default=2)
    #     The minimum number of samples required to split an internal node.
    #
    # min_samples_leaf : integer, optional (default=1)
    #     The minimum number of samples required to be at a leaf node.
    #
    # min_weight_fraction_leaf : float, optional (default=0.)
    #     The minimum weighted fraction of the input samples required to be at a leaf node.
    #
    # subsample : float, optional (default=1.0)
    #     The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in
    #     Stochastic Gradient Boosting. subsample interacts with the parameter n_estimators.
    #     Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.
    #
    # max_features : int, float, string or None, optional (default=None)
    #     The number of features to consider when looking for the best split:
    #         If int, then consider max_features features at each split.
    #         If float, then max_features is a percentage and int(max_features * n_features) features are considered at
    #         each split.
    #         If "auto", then max_features=sqrt(n_features).
    #         If "sqrt", then max_features=sqrt(n_features).
    #         If "log2", then max_features=log2(n_features).
    #         If None, then max_features=n_features.
    #
    #     Choosing max_features < n_features leads to a reduction of variance and an increase in bias.
    #     Note: the search for a split does not stop until at least one valid partition of the node samples is found,
    #     even if it requires to effectively inspect more than max_features features.
    #
    # max_leaf_nodes : int or None, optional (default=None)
    #     Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity.
    #     If None then unlimited number of leaf nodes. If not None then max_depth will be ignored.
    #
    # init : BaseEstimator, None, optional (default=None)
    #     An estimator object that is used to compute the initial predictions. init has to provide fit and predict.
    #     If None it uses loss.init_estimator.
    #
    # random_state : int, RandomState instance or None, optional (default=None)
    #     If int, random_state is the seed used by the random number generator; If RandomState instance, random_state
    #     is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
    #
    'GB': {
        'loss': ['deviance'],
        'learning_rate': [0.1],
        'n_estimators': [100],
        'subsample': [1.0],
        'max_depth': [3],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'min_weight_fraction_leaf': [0.0],
        'init': [None],
        'random_state': [999],
        'max_features': [None],
        'max_leaf_nodes': [None],
    },

    # RandomForestRegressor
    'RFR': {
        'n_estimators': [2, 15, 100],
        'criterion': ['mse'],
        'max_features': ['auto'],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'min_weight_fraction_leaf': [0.0],
        'max_leaf_nodes': [None],
        'bootstrap': [True],
        'oob_score': [False],
        'n_jobs': [1],
        'random_state': [999],
    },

}

