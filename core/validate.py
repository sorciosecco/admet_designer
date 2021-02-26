
from core import variables, settings
from core.metrics import calc_regr_metrics

from sklearn.model_selection import LeaveOneOut as loo
from sklearn.model_selection import cross_val_score

def leave_one_out(X, Y, M):
    Yp=[]
    i=1
    for train_index, test_index in loo().split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        M.fit(X_train, Y_train)
        Yp.append(M.predict(X_test).tolist()[0])
        if settings.MODEL!="PLS": print("%s/%s" % (i, len(variables.Y_tra)))
        i+=1
    if settings.MODEL=="PLS": Y_pred = [ y[0] for y in Yp ]
    else: Y_pred=Yp
    q2, sdep = calc_regr_metrics(Y_exp=Y, Y_pred=Y_pred)
    return q2, sdep, Y_pred


def cross_validation(X, Y, M):
    scores = cross_val_score(M, X, Y, cv=settings.CROSSVAL, verbose=settings.VERBOSE)
    return scores

