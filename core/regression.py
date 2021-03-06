
from core import variables, settings, parameters
from core.metrics import calc_regr_metrics
from core.validate import leave_one_out, cross_validation

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


def run_pls(X, Y, LV):
    model = PLSRegression(n_components=LV, scale=False)
    model.fit(X, Y)
    Yr = [ y[0] for y in model.predict(X).tolist() ]
    r2, sdec = calc_regr_metrics(Y_exp=Y, Y_pred=Yr)
    q2, sdep, variables.Y_pred = leave_one_out(X=np.array(X), Y=np.array(Y), M=model)
    scores = { 'R2': r2, 'Q2': q2, 'SDEC': sdec,'SDEP': sdep }
    return scores, model


def run_model_training():
    Xe, Ye = variables.X_tra, variables.Y_tra
    
    
    if settings.NPARA==False: ##### ALGORITHMS WITH DEFAULT PARAMETERS
        
        if settings.MODEL=="PLS":
            print("LV\tR2\tQ2\tSDEC\tSDEP")
            stop, best_q2, best_lv = False, 0, 0
            for lv in range(1, 11):
                scores, model = run_pls(X=Xe, Y=Ye, LV=lv)
                if scores['Q2'] < best_q2:
                    stop=True
                else:
                    best_q2, best_lv, variables.model = scores['Q2'], lv, model
                    print('\t'.join([str(lv)] + [ str(round(scores[k],3)) for k in list(scores.keys()) ]))
                if stop:
                    break
        elif settings.MODEL=="RF": model=RandomForestRegressor(random_state=settings.SEED, n_jobs=-1, verbose=False, n_estimators=100, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, warm_start=False)
        elif settings.MODEL=="kNN": model=KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=-1)
        elif settings.MODEL=="SVM": model=SVR(kernel='linear', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
        elif settings.MODEL=="MLP": model=MLPRegressor(hidden_layer_sizes=100, activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=settings.SEED, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
        else: print("\nERROR: algorithm not supported\n")
    
    else:##### ALGORITHMS WITH NON-DEFAULT PARAMETERS
        
        if settings.MODEL=="PLS":
            scores, model = run_pls(X=Xe, Y=Ye, LV=parameters.lv)
            print("LV\tR2\tQ2\tSDEC\tSDEP\n" + "\t".join([str(parameters.lv)] + [ str(round(scores[k],3)) for k in list(scores.keys()) ]))
        elif settings.MODEL=="kNN": model=KNeighborsRegressor(n_neighbors=parameters.n_neighbors, p=parameters.p, weights=parameters.weights, algorithm=parameters.algorithm_knn, leaf_size=parameters.leaf_size)
        elif settings.MODEL=="SVM": model=SVR(C=parameters.C, kernel=parameters.kernel, gamma=parameters.gamma, degree=parameters.degree)
        elif settings.MODEL=="MLP": model=MLPRegressor(random_state=settings.SEED, solver=parameters.solver_mlp, activation=parameters.activation, learning_rate_init=parameters.learning_rate_init, learning_rate=parameters.learning_rate, hidden_layer_sizes=parameters.hidden_layer_sizes,  max_iter=250)
        else: print("\nERROR: algorithm not supported\n")
    
    
    if settings.MODEL!="PLS":
        variables.model = model
        if variables.DMODY:
            q2, sdep, variables.Y_pred = leave_one_out(X=np.array(Xe), Y=np.array(Ye), M=model)
            print("Q2\tSDEP\n%s\t%s" % (round(q2,3), round(sdep,3)))
        if settings.OPTIMIZE==False:
            model.fit(Xe, Ye)
            scores = cross_validation(X=Xe, Y=Ye, M=model)
            print("cross-validation results:\nCV1\tCV2\tCV3\tCV4\tCV5\tAVER\n%s\t%s\t%s\t%s\t%s\t%s" % tuple([round(s,3) for s in scores] + [round(np.mean(scores),3)]))
    
