
import numpy as np

param_grids = {
    'rNN': {
        'radius': np.arange(1.0,10.1,0.1).tolist(),
        #'weights': ['uniform', 'distance'],
        #'leaf_size': np.arange(1,51,1).tolist(),
        #'algorithm_knn': ['ball_tree', 'kd_tree', 'brute', 'auto'],
        #'p': [1,2],
    },
    'kNN': {
        'n_neighbors': np.arange(3,21,1).tolist(),
        'weights': ['uniform', 'distance'],
        #'leaf_size': np.arange(1,51,1).tolist(),
        #'algorithm_knn': ['ball_tree', 'kd_tree', 'brute', 'auto'],
        'p': [1,2],
    },
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
    #
    # oob_score : bool
    #     Whether to use out-of-bag samples to estimate the generalization error.
    #
    'RF': {
        'n_estimators': np.arange(10,105,10).tolist(),
        'criterion_rf': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [None] + np.arange(3,11,1).tolist(),
        'max_leaf_nodes': [None] + np.arange(5,25,5).tolist(),
        'class_weight': [None, 'balanced', 'balanced_subsample'],
    },
     'ETC': {
        'n_estimators': np.arange(10,505,10).tolist(),
        'criterion_rf': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [None] + np.arange(3,11,1).tolist(),
        'max_leaf_nodes': [None] + np.arange(5,55,5).tolist(),
        'class_weight': [None, 'balanced', 'balanced_subsample'],
    },
    'AB': {
        'n_estimators': np.arange(10, 505, 5).tolist(),
        'algorithm_ab': ['SAMME.R', 'SAMME'],
    },
    # GRADIENT BOOSTING (GB)
    #
    # learning_rate : float, optional (default=0.1)
    #     learning rate shrinks the contribution of each tree by learning_rate.
    #     There is a trade-off between learning_rate and n_estimators.
    #
    # subsample : float, optional (default=1.0)
    #     The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in
    #     Stochastic Gradient Boosting. subsample interacts with the parameter n_estimators.
    #     Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.
    #
    # init : BaseEstimator, None, optional (default=None)
    #     An estimator object that is used to compute the initial predictions. init has to provide fit and predict.
    #     If None it uses loss.init_estimator.
    #
    'GB': {
        'n_estimators': np.arange(10,505,10).tolist(),
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [None] + np.arange(3,11,1).tolist(),
        'max_leaf_nodes': [None] + np.arange(5,55,5).tolist(),
        'loss': ['deviance', 'exponential'],
    },
    'LDA': {
        'shrinkage': np.arange(0.01,1.01,0.01).tolist() + [None, 'auto'],
        'solver': ['svd', 'lsqr', 'eigen'],
    },
    'SVM': {
        'C': np.arange(1,11,1).tolist(),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['auto', 'scale'] + np.arange(0.001,0.016,0.001).tolist(),
        'degree': [1,2,3],
        'class_weight': [None, 'balanced'],
    },
    #'MLP': {
        #'hidden_layer_sizes': np.arange(1,11,1).tolist(),
        #'max_iter': np.arange(1,11,1).tolist(),
        #'max_iter': ['linear', 'poly', 'rbf', 'sigmoid'],
    #},
}
