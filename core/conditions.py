
import numpy as np

#print([tuple(s-10*x for x in range(l)) for s in np.flip(np.arange(10,110,10)) for l in range(1,int(100/10)+1) if s-10*(l-1)>0])
#print([tuple(s+10*x for x in range(l)) for s in np.arange(10,110,10) for l in range(1,int(100/10)+1) if s+10*(l-1)<100])
# layer(s) dimensions
increment=10
min_layer_dim=10
max_layer_dim=100
#=======================================
LAYERS = []
# Each successive layer has a lower number of neurons.
for d in [tuple(s-increment*x for x in range(l)) for s in np.flip(np.arange(min_layer_dim,max_layer_dim+increment,increment)) for l in range(1,int(max_layer_dim/increment)+1) if s-increment*(l-1)>0]:
    if d not in LAYERS: LAYERS.append(d)
# Each successive layer has a higher number of neurons.
for a in [tuple(s+increment*x for x in range(l)) for s in np.arange(min_layer_dim,max_layer_dim+increment,increment) for l in range(1,int(max_layer_dim/increment)+1) if s+increment*(l-1)<max_layer_dim]:
    if a not in LAYERS: LAYERS.append(a)
# Each successive layer has an equal number of neurons.
for e in [tuple(increment*s for _ in range(l)) for s in range(1,increment+1) for l in range(1,5+1)]:
    if e not in LAYERS: LAYERS.append(e)
#=======================================

param_grids = {
    'MLP': {
        #'hidden_layer_sizes': LAYERS,
        #'max_iter': np.arange(100,450,50).tolist() + np.arange(50,100,10).tolist(),
        #'solver_mlp': ['lbfgs', 'sgd', 'adam'],
        #'activation': ['identity', 'logistic', 'tanh', 'relu'],
        #'learning_rate_init': [0.001, 0.01, 0.1, 1.0],
        #'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'alpha': [0.0001*10**x for x in range(8)],
        'beta_1': np.arange(0.1,1,0.1).tolist(),
        'beta_2': np.arange(0.111,1,0.111).tolist(),
        'power_t': [0.0005*10**x for x in range(4)],
        'momentum': np.arange(0.1,1,0.1).tolist(),
        #'nesterovs_momentum': [True, False],
        'epsilon': [1e-8*10**x for x in range(9)],
    },
    'RF': {
        'n_estimators': np.arange(10,210,10).tolist(),
        'max_features': ['sqrt', 'log2', None, 'auto'],
        'max_depth': [None] + np.arange(3,11,1).tolist(),
        #'min_samples_split': np.arange(2,11,1).tolist(),
        #'min_samples_leaf': np.arange(1,11,1).tolist(),
        ### regression only params.
        #'criterion_rf': ['mse', 'mae'],
        ### classification only params.
        'criterion_rf': ['gini', 'entropy'],
        'class_weight': [None, 'balanced', 'balanced_subsample'],
    },
     'ET': {
        'n_estimators': np.arange(10,210,10).tolist(),
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [None] + np.arange(3,11,1).tolist(),
        #'min_samples_split': np.arange(2,11,1).tolist(),
        #'min_samples_leaf': np.arange(1,11,1).tolist(),
        ### regression only params.
        #'criterion_rf': ['mse', 'mae'],
        ### classification only params.
        'criterion_rf': ['gini', 'entropy'],
        'class_weight': [None, 'balanced', 'balanced_subsample'],
    },
    'AB': {
        'n_estimators': np.arange(10, 210, 10).tolist(),
        'algorithm_ab': ['SAMME.R', 'SAMME'],
    },
    'GB': {
        #'n_estimators': np.arange(10,210,10).tolist(),
        #'max_features': ['sqrt', 'log2', None],
        'max_depth': [None] + np.arange(3,11,1).tolist(),
        'min_samples_split': np.arange(2,11,1).tolist(),
        'min_samples_leaf': np.arange(1,11,1).tolist(),
        #'loss': ['deviance', 'exponential'],
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
        ### classification only params.
        'class_weight': [None, 'balanced'],
    },
    'kNN': {
        'n_neighbors': np.arange(3,21,1).tolist(),
        'weights': ['uniform', 'distance'],
        'leaf_size': np.arange(10,51,5).tolist(),
        'algorithm_knn': ['ball_tree', 'kd_tree', 'brute'],
        'p': [1,2,3],
    },
    #'rNN': {
        #'radius': np.arange(1.0,10.1,0.1).tolist(),
        #'weights': ['uniform', 'distance'],
        #'leaf_size': np.arange(1,51,1).tolist(),
        #'algorithm_knn': ['ball_tree', 'kd_tree', 'brute', 'auto'],
        #'p': [1,2],
    #},
}
