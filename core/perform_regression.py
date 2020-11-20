from core import settings
from core.metrics import calc_regr_metrics

import numpy as np
from sklearn.model_selection import LeaveOneOut as loo
from sklearn.cross_decomposition import PLSRegression


def leave_one_out(X, Y, M):
    Yp=[]
    for train_index, test_index in loo().split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        M.fit(X_train, Y_train)
        Yp.append(M.predict(X_test).tolist()[0])
    Y_pred = [ y[0] for y in Yp ]
    q2, sdep = calc_regr_metrics(Ye=Y, Yp=Y_pred)
    return q2, sdep


def get_pls_predictions(latent_variables, tra_X, tra_Y, tes_X=None):
    
    model = PLSRegression(n_components=latent_variables, scale=False)
    
    model.fit(tra_X, tra_Y)
    
    tra_Yp = [ y[0] for y in model.predict(tra_X).tolist() ]
    if tes_X != None: tes_Yp = [ y[0] for y in model.predict(tes_X).tolist() ]
    
    r2, sdec = q2, sdep = calc_regr_metrics(Ye=tra_Y, Yp=tra_Yp)
    q2, sdep = leave_one_out(X=np.array(tra_X), Y=np.array(tra_Y), M=model)
    
    scores = {
        'R2': r2,
        'Q2': q2,
        'SDEC': sdec,
        'SDEP': sdep,
    }
    
    if tes_X != None: return tra_Yp, tes_Yp, scores
    else: return tra_Yp, scores
    
