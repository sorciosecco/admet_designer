
import numpy as np

from core import variables, settings
from core.metrics import calc_regr_metrics

from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut, cross_val_score, KFold


def multi_class_cv(X, Y):
    X, Y, M, n = np.array(X), np.array(Y), variables.model, 0
    mean_ba, mean_f1 = 0, np.array([0 for x in variables.classes])
    print("\ncross validation results:\nCV\tF1 (%s)\tF1 (%s)\tF1 (%s)\tBA" % tuple(variables.classes))
    for train_index, test_index in KFold(n_splits=5, shuffle=True, random_state=settings.SEED).split(X):
        n+=1
        X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
        M.fit(X_train, Y_train)
        Yp_test = M.predict(X_test)
        f1, ba = f1_score(Y_test, Yp_test, average=None), balanced_accuracy_score(Y_test, Yp_test)
        mean_ba += ba
        mean_f1 = mean_f1 + f1
        print("%s\t%s\t%s\t%s\t%s" % tuple([n] + [round(f, 2) for f in f1] + [round(ba, 2)]))
    print("AVER\t%s\t%s\t%s\t%s" % tuple([round(f/n, 2) for f in mean_f1] + [round(mean_ba/n, 2)]))


def multi_class_loo(X, Y, M):
    pass


def leave_one_out(X, Y, M):
    Yp=[]
    i=1
    for train_index, test_index in LeaveOneOut().split(X):
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

