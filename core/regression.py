
from core import variables, settings
from core.metrics import calc_regr_metrics
from core.validate import leave_one_out

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

def run_model_training(nondef_params=False):
    Xe, Ye = variables.X_tra, variables.Y_tra
    ##### LINEAR ALGORITHMS
    
    # Partial Least Squares (PLS)
    if settings.MODEL=="PLS":
        print("LV\tR2\tQ2\tSDEC\tSDEP")
        stop, best_q2, best_lv = False, 0, 0
        for lv in range(1, 11):
            model = PLSRegression(n_components=lv, scale=False)
            model.fit(Xe, Ye)
            Yr = [ y[0] for y in model.predict(Xe).tolist() ]
            r2, sdec = calc_regr_metrics(Y_exp=Ye, Y_pred=Yr)
            q2, sdep, variables.Y_pred = leave_one_out(X=np.array(Xe), Y=np.array(Ye), M=model)
            scores = { 'R2': r2, 'Q2': q2, 'SDEC': sdec,'SDEP': sdep }
            if scores['Q2'] < best_q2:
                stop=True
            else:
                best_q2, best_lv, variables.model = scores['Q2'], lv, model
                print('\t'.join([str(lv)] + [ str(round(scores[k],3)) for k in list(scores.keys()) ]))
            if stop:
                break
    
    ##### TREE ENSEMBLE ALGORITHMS
    
    # Random Forest (RF)
    elif settings.MODEL=="RF":
        model = RandomForestRegressor(random_state=settings.SEED, n_jobs=-1, verbose=settings.VERBOSE
                                      , n_estimators=100
                                      , criterion='mse'
                                      , max_depth=None
                                      , max_features='auto'
                                      , max_leaf_nodes=None
                                      , min_samples_split=2
                                      , min_samples_leaf=1
                                      , min_weight_fraction_leaf=0.0
                                      , min_impurity_decrease=0.0
                                      , min_impurity_split=None
                                      , bootstrap=True
                                      , oob_score=False
                                      , warm_start=False
                                      )
    ##### NEIGHBORS ALGORITHMS
    
    # k-Nearest Neighbors (kNN)
    elif settings.MODEL=="kNN":
        model = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=-1)
    ##### SUPPORT VECTOR MACHINES
    
    # Support Vector Regression (SVR)
    elif settings.MODEL=="SVM":
        model = SVR(kernel='linear', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    ##### ARTIFICIAL NEURAL NETWORKS
    
    # Multi-layer Perceptron Regressor (MLP)
    elif settings.MODEL=="MLP":
        model = MLPRegressor(hidden_layer_sizes=100, activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=settings.SEED, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
    else:
        print("\nERROR: algorithm not supported\n")
    
    if settings.MODEL!="PLS":
        model.fit(Xe, Ye)
        q2, sdep, variables.Y_pred = leave_one_out(X=np.array(Xe), Y=np.array(Ye), M=model)
        variables.model = model
        print("Q2\tSDEP\n%s\t%s" % (round(q2,3), round(sdep,3)))
        
    return model

