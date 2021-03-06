
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
        'leaf_size': np.arange(10,51,5).tolist(),
        'algorithm_knn': ['ball_tree', 'kd_tree', 'brute'],
        'p': [1,2,3],
    },
    'RF': {
        'n_estimators': np.arange(10,105,10).tolist(),
        'criterion_rf': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None, 'auto'],
        'max_depth': [None] + np.arange(3,11,1).tolist(),
        'max_leaf_nodes': [None] + np.arange(5,25,5).tolist(),
        #'class_weight': [None, 'balanced', 'balanced_subsample'],
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
    'GB': {
        'n_estimators': np.arange(10,505,10).tolist(),
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [None] + np.arange(3,11,1).tolist(),
        'max_leaf_nodes': [None] + np.arange(5,55,5).tolist(),
        'loss': ['deviance', 'exponential'],
    },
    'LDA': {
        'shrinkage': np.arange(0.01,1.01,0.01).tolist() + [None, 'auto'],
        'solver_lda': ['svd', 'lsqr', 'eigen'],
    },
    'SVM': {
        'C': np.arange(10,110,10).tolist(),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['auto', 'scale'] + np.arange(0.001,0.016,0.001).tolist(),
        'degree': [1,2,3],
        #'class_weight': [None, 'balanced'],
    },
    'MLP': {
        #'solver_mlp': ['lbfgs', 'sgd', 'adam'],
        'solver_mlp': ['adam'],
        #'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'activation': ['logistic'],
        #'learning_rate_init': [0.001, 0.01, 0.1, 1.0],
        'learning_rate_init': [0.001],
        #'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'learning_rate': ['constant'],
        #'hidden_layer_sizes': [(s,) for s in np.arange(10,110,10)],
        'hidden_layer_sizes': [ tuple(10*s for _ in range(l)) for s in range(1, 10+1) for l in range(1, 5+1) ],
        #'max_iter': np.arange(1,11,1).tolist(),
    },
}
