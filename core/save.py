from core import settings

import os, pickle
import numpy as np


def save_vars_importance(V, I):
    print("\nSaving feature importances...\n")
    csv=open(os.path.join(os.getcwd(), "feature_importances.csv"), "w")
    csv.write("Variable\tImportance\n")
    for x in range(len(V)):
        csv.write("%s\t%s\n" % (V[x], I[x]))
    csv.close()


def save_pls_predictions(y_train_exp, y_test_exp, y_train_pred, y_test_pred, ob_train, ob_test, lt, ht):
    print("\nSaving the predictions...")
    ocsv=open("PLS_predictions.csv", 'w')
    ocsv.write("Object;Set;Y_exp;Y_pred;Y_pred_class;Classification (%s/%s)\n" % (lt, ht))
    for i, Set in enumerate([(y_train_exp, y_train_pred, ob_train), (y_test_exp, y_test_pred, ob_test)]):
        for o in range(len(Set[2])):
            if Set[1][o] > ht:
                if Set[0][o]==1:
                    C, E, P = 'TP', Set[0][o], 1
                else:
                    C, E, P = 'FP', Set[0][o], 1
            elif Set[1][o] < lt:
                if Set[0][o]==0:
                    C, E, P = 'TN', Set[0][o], -1
                else:
                    C, E, P = 'FN', Set[0][o], -1
            else:
                C, E, P = 'NOTPRED', Set[0][o], 0
            if i==0:
                ocsv.write("%s;TRAINING;%s;%s;%s;%s\n" % (Set[2][o], E, Set[1][o], P, C))
            else:
                ocsv.write("%s;TEST;%s;%s;%s;%s\n" % (Set[2][o], E, Set[1][o], P, C))
    ocsv.close()


def save_predictions(name, Y1_exp, Y1_pred, Y1_prob, Y2_exp, Y2_pred, Y2_prob, O1, O2, pc):
    print("\nSaving the predictions...")
    ocsv=open(name+"_predictions.csv", 'w')
    ocsv.write("Object;Set;Y_exp;Y_pred;Classification;Y_prob\n")
    for i, Set in enumerate([(Y1_exp, Y1_pred, Y1_prob, O1), (Y2_exp, Y2_pred, Y2_prob, O2)]):
        for o in range(len(Set[3])):
            if max(Set[2][o]) > pc:
                if Set[0][o]==1 and Set[1][o]==1:
                    C, E, P = 'TP', 1, 1
                elif Set[0][o]==1 and Set[1][o]==0:
                    C, E, P = 'FN', 1, -1
                elif Set[0][o]==0 and Set[1][o]==0:
                    C, E, P = 'TN', 0, -1
                elif Set[0][o]==0 and Set[1][o]==1:
                    C, E, P = 'FP', 0, 1
            else:
                C, E, P = 'NOTPRED', Set[0][o], 0
            if i==0:
                ocsv.write("%s;TRAINING;%s;%s;%s;%s\n" % (Set[3][o], E, P, C, max(Set[2][o])))
            else:
                ocsv.write("%s;TEST;%s;%s;%s;%s\n" % (Set[3][o], E, P, C, max(Set[2][o])))
    ocsv.close()

    
def save_multi_predictions(name, Y1_exp, Y1_pred, Y1_prob, Y2_exp, Y2_pred, Y2_prob, O1, O2):
    print("\nSaving the predictions...")
    ocsv=open(name+"_predictions.csv", 'w')
    ocsv.write("Object;Set;Y_exp;Y_pred;Y_prob;Is pred?\n")
    for i, Set in enumerate([(Y1_exp, Y1_pred, Y1_prob, O1), (Y2_exp, Y2_pred, Y2_prob, O2)]):
        classes, occurrences = np.unique(Set[0], return_counts=True)[0].tolist(), np.unique(Set[0], return_counts=True)[1].tolist()
        cutoff=1/float(len(classes))
        for o in range(len(Set[3])):
            if max(Set[2][o]) > cutoff:
                line = (Set[3][o], Set[0][o], Set[1][o], max(Set[2][o]), 'PRED')
            else:
                line = (Set[3][o], Set[0][o], Set[1][o], max(Set[2][o]), 'NOTPRED')
            if i==0:
                ocsv.write("%s;TRAINING;%s;%s;%s;%s\n" % line)
            else:
                ocsv.write("%s;TEST;%s;%s;%s;%s\n" % line)
    ocsv.close()


def save_model(m, mt):
    model_filename = os.path.join(os.getcwd(), mt+'.model')
    picklefile = open(model_filename, 'wb')
    pickle.dump(m, picklefile)
    picklefile.close()
    
