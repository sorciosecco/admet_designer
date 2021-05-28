
import numpy as np

from core import variables, settings
from core.metrics import calc_regr_metrics

from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import LeaveOneOut, cross_val_score, KFold


def calc_accuracy_only(X, Y, M):
    S, X, Y = [], np.array(X), np.array(Y)
    for train_index, test_index in KFold(n_splits=5, shuffle=True, random_state=settings.SEED).split(X):
        X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
        M.fit(X_train, Y_train)
        Yp_test = M.predict(X_test)
        if variables.N_classes > 2: S.append(balanced_accuracy_score(Y_test, Yp_test))
        else: S.append(accuracy_score(Y_test, Yp_test))
    return S


def class_cv(X, Y):
    X, Y, model, n = np.array(X), np.array(Y), variables.model, 0
    mean_a, mean_f1 = 0, np.array([0 for x in variables.classes])
    if variables.N_classes > 2: print("\ncross validation results:\nCV\t"+"\t".join(["F1 (%s)" % c for c in variables.classes]) +"\tBA")
    else: print("\ncross validation results:\nCV\t"+"\t".join(["F1 (%s)" % c for c in variables.classes]) +"\tAC")
    for train_index, test_index in KFold(n_splits=5, shuffle=True, random_state=settings.SEED).split(X):
        n+=1
        X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
        model.fit(X_train, Y_train)
        Yp_test = model.predict(X_test)
        f1 = f1_score(Y_test, Yp_test, average=None)
        if variables.N_classes > 2: a = balanced_accuracy_score(Y_test, Yp_test)
        else: a = accuracy_score(Y_test, Yp_test)
        mean_a += a
        mean_f1 = mean_f1 + f1
        print("\t".join([ str(l) for l in [n]+[ round(f, 2) for f in f1 ]+[round(a, 2)] ]))
    print("\t".join([ str(l) for l in ["AVER"]+[ round(f/n, 2) for f in mean_f1 ]+[round(mean_a/n, 2)] ]))


def class_loo(X, Y):
    Yp=[]
    X, Y, M, n = np.array(X), np.array(Y), variables.model, 0
    for train_index, test_index in LeaveOneOut().split(X):
        n+=1
        X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
        M.fit(X_train, Y_train)
        yp = M.predict(X_test).tolist()[0]
        Yp.append(yp)
        print("%s/%s\t%s\t%s" % (n, len(Y), Y_test, yp))
    f1 = f1_score(Y, Yp, average=None)
    if variables.N_classes > 2:
        a = balanced_accuracy_score(Y, Yp)
        print("\nleave one out results:\n"+"\t".join(["F1 (%s)" % c for c in variables.classes])+"\tBA\n"+"\t".join([ str(l) for l in [ round(f, 2) for f in f1 ]+[round(a, 2)] ]))
    else:
        a = accuracy_score(Y, Yp)
        print("\nleave one out results:\n"+"\t".join(["F1 (%s)" % c for c in variables.classes])+"\tAC\n"+"\t".join([ str(l) for l in [ round(f, 2) for f in f1 ]+[round(a, 2)] ]))
    return Yp


def regr_loo(X, Y, M):
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

