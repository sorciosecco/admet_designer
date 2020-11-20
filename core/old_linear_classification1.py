from core import settings

import os, math
import pickle
import numpy as np

from sklearn.preprocessing import scale
from sklearn.model_selection import LeaveOneOut as loo
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.cross_decomposition import PLSRegression

class Dataset:
    def __init__(self, y):
        self.y=y
        self.X_labels=[]
        self.Objects=[]
        self.X=[]
        self.Y=[]
        self.Objects2=[]
        self.X2 = []
        self.Y2 = []
    
    def __str__(self):
        classes, occurrences = np.unique(self.Y, return_counts=True)[0].tolist(), np.unique(self.Y, return_counts=True)[1].tolist()
        output = '\nTRAINING SET:\nN objects = %s\nN independent vars = %s\nN dependent vars: 1\n' % (len(self.Objects), len(self.X_labels)) + 'Y (%s) = %s (class %s) + %s (class %s)' % (self.y, occurrences[0], classes[0], occurrences[1], classes[1])
        if settings.PREDICT:
            classes, occurrences = np.unique(self.Y2, return_counts=True)[0].tolist(), np.unique(self.Y2, return_counts=True)[1].tolist()
            output += '\n\nTEST SET:\nN objects = %s\n' % (len(self.Objects2)) + 'Y (%s) = %s (class %s) + %s (class %s)' % (self.y, occurrences[0], classes[0], occurrences[1], classes[1])
        return output
    
    def load_validation_data(self, valset_file):
        l=0
        for line in open(valset_file, 'r'):
            if l==0:
                line = str.split(line.strip(), ';')[1:]
                Yind = line.index(self.y)
            else:
                self.Objects2.append(line.split(';')[0])
                line = str.split(line.strip(), ';')[1:]
                self.X2.append([float(line[x]) for x in range(len(line)) if x != Yind])
                self.Y2.append(int(line[Yind]))
            l+=1
    
    def autoscale(self):
        self.X = scale(self.X, axis=0, with_mean=True, with_std=True, copy=True).tolist()
        if settings.PREDICT != None:
            self.X2 = scale(self.X2, axis=0, with_mean=True, with_std=True, copy=True).tolist()
    
    def load_training_data(self, training_set_file):
        l=0
        for line in open(training_set_file, 'r'):
            if l == 0:
                line = str.split(line.strip(), ';')[1:]
                Yind = line.index(self.y)
                self.X_labels = [line[x] for x in range(len(line)) if x != Yind]
            else:
                self.Objects.append(line.split(';')[0])
                line = str.split(line.strip(), ';')[1:]
                self.X.append([float(line[x]) for x in range(len(line)) if x != Yind])
                self.Y.append(int(line[Yind]))
            l+=1

class Model:
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train=X_train
        self.X_test=X_test
        self.Y_train=Y_train
        self.Y_test=Y_test
        self.model=None
        self.LV=0
        self.M={}
        self.R={}
        self.P={}
    
    def save_model(self):
        model_file_name = os.path.join(os.getcwd(), settings.MODEL+'.model')
        picklefile = open(model_file_name, 'wb')
        pickle.dump(self.model, picklefile)
        picklefile.close()
    
    # Partial Least Squares (PLS)
    def partial_least_squares(self):
        
        def leave_one_out():
            X, Y = np.array(self.X_train), np.array(self.Y_train)
            Yp=[]
            for train_index, test_index in loo().split(X):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                self.model.fit(X_train, Y_train)
                Yp.append(self.model.predict(X_test).tolist()[0])
            Y_pred = [ y[0] for y in Yp ]
            Q2 = r2_score(self.Y_train, Y_pred)
            SDEP = math.sqrt(mean_squared_error(self.Y_train, Y_pred))
            return SDEP, Q2
        
        def calc_train_metrics(Y_exp, Y_recal, LV):
            thresholds = [[x,y] for x in np.arange(0, 1.01, 0.01).tolist() for y in np.arange(0, 1.01, 0.01).tolist()]
            ocsv=open('Yrecal_LV%s.csv' % (LV), 'w')
            ocsv.write('low threshold\thigh threshold\tTP\tFN\tTN\tFP\tUNC\tACC\tSE\tSP\tpPREC\tnPREC\tF1\tMCC\tTotal coverage\tP coverage\tN coverage\n')
            t_list=[]
            for t in thresholds:
                if sorted(t) not in t_list:
                    t_list.append(t)
                    Yr, Ye = [], []
                    UNC=0
                    for y in range(len(Y_recal)):
                        if Y_recal[y] > t[1]:
                            Ye.append(Y_exp[y])
                            Yr.append(1)
                        elif Y_recal[y] < t[0]:
                            Ye.append(Y_exp[y])
                            Yr.append(0)
                        else:
                            UNC+=1
                    TN, FP, FN, TP = confusion_matrix(Ye, Yr).ravel()
                    ACC, F1, MCC = round(accuracy_score(Ye, Yr),2), round(f1_score(Ye, Yr),2), round(matthews_corrcoef(Ye, Yr),2)
                    pPREC, nPREC = round(precision_score(Ye, Yr),2), round(TN/(TN+FN),2)
                    SE, SP = round(recall_score(Ye, Yr),2), round(TN/(TN + FP),2)
                    COV, Pcov, Ncov = round((len(Y_exp)-UNC)*100/len(Y_exp),1), round(Ye.count(1)*100/Y_exp.count(1),1), round(Ye.count(0)*100/Y_exp.count(0),1)
                    ocsv.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (t[0],t[1],TP,FN,TN,FP,UNC,ACC,SE,SP,pPREC,nPREC,F1,MCC,COV,Pcov,Ncov))
            ocsv.close()
        
        if settings.LOO:
            print('\nValidating PLS with LOO...\nLV\tR2\tQ2\tSDEC\tSDEP')
        o, top_Q2 = 0, 0
        for lv in range(1, 11):
            self.model = PLSRegression(n_components=lv, scale=False, max_iter=500, tol=1e-06, copy=True)
            self.model.fit(self.X_train, self.Y_train)
            Y_recal = [y[0] for y in self.model.predict(self.X_train).tolist()]
            R2 = r2_score(self.Y_train, Y_recal)
            SDEC = math.sqrt(mean_squared_error(self.Y_train, Y_recal))
            if settings.LOO:
                SDEP, Q2 = leave_one_out()
                if Q2 < top_Q2:
                    o+=1
                else:
                    top_Q2=Q2
                    self.LV=lv
                    print('%s\t%s\t%s\t%s\t%s' % (self.LV, round(R2,3), round(Q2,3), round(SDEC,3), round(SDEP,3)))
                    self.M[self.LV] = (self.LV, round(R2,3), round(Q2,3), round(SDEC,3), round(SDEP,3))
                    self.R[self.LV] = Y_recal
                    if settings.PREDICT:
                        self.P[self.LV] = [y[0] for y in self.model.predict(self.X_test).tolist()]
            if o==1 or lv==settings.LATENT:
                break
        
        if self.LV>0:
            self.model = PLSRegression(n_components=self.LV, scale=False, max_iter=500, tol=1e-06, copy=True)
            self.model.fit(self.X_train, self.Y_train)
        
        if settings.PREDICT:
            print('\nComputing activity cutoffs and scores...')
            calc_train_metrics(Y_exp=self.Y_train, Y_recal=self.R[self.LV], LV=self.LV)

def effetti_principali(X_train, Y_train, X_test, Y_test, model):
    
    def leave_one_out(X, Y, model):
        X, Y = np.array(X), np.array(Y)
        Yp=[]
        for train_index, test_index in loo().split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            model.fit(X_train, Y_train)
            Yp.append(model.predict(X_test).tolist()[0][0])
        Q2 = r2_score(Y.tolist(), Yp)
        return Q2
    
    def generate_signs_matrix(vars_numb):
        # definisco le colonne generatrici
        matrix=[]
        gencols_numb=0
        while 7:
            gencols_numb+=1
            exps_numb=2**gencols_numb
            if exps_numb>vars_numb:
                break
        for a in range(gencols_numb):
            matrix.append([])
            e=2**a
            for b in range(e):
                matrix[a].append(-1)
            for b in range(e,exps_numb):
                matrix[a].append(matrix[a][b-e]*-1)
        if settings.VERBOSE:
            print('\nMatrice dei segni.\n\nNumero di variabili = %s\nNumero di esperimenti = %s\nNumero di colonne generatrici = %s' % (vars_numb, exps_numb, gencols_numb))
        # cicli dei doppi prodotti
        for i in range(gencols_numb):
            for j in range(i+1,gencols_numb):
                #print(i+1,j+1)
                if len(matrix)==vars_numb:
                    break
                else:
                    matrix.append([a*b for a,b in zip(matrix[i],matrix[j])])
        print("Numero di colonne dopo i doppi prodotti = %s" % (len(matrix)))
        # cicli dei tripli prodotti
        x=0
        for i in range(gencols_numb):
            for j in range(i+1,gencols_numb-x):
                for k in range(j+1,gencols_numb):
                    #print(i+1,j+1,k+1)
                    if len(matrix)==vars_numb:
                        break
                    else:
                        matrix.append([a*b*c for a,b,c in zip(matrix[i],matrix[j],matrix[k])])
                if len(matrix)==vars_numb:
                    break
            x+=1
            if len(matrix)==vars_numb:
                break
        print("Numero di colonne dopo i tripli prodotti = %s" % (len(matrix)))
        # cicli dei tetra prodotti
        x=0
        for i in range(gencols_numb):
            for j in range(i+1,gencols_numb-x):
                for k in range(j+1,gencols_numb-x+1):
                    for l in range(k+1,gencols_numb):
                        #print(i+1,j+1,k+1,l+1)
                        if len(matrix)==vars_numb:
                            break
                        else:
                            matrix.append([a*b*c*d for a,b,c,d in zip(matrix[i],matrix[j],matrix[k],matrix[l])])
                    if len(matrix)==vars_numb:
                        break
                if len(matrix)==vars_numb:
                    break
            x+=1
            if len(matrix)==vars_numb:
                break
        print("Numero di colonne dopo i tetra prodotti = %s" % (len(matrix)))
        # cicli dei penta prodotti
        x=0
        for i in range(gencols_numb):
            for j in range(i+1,gencols_numb-x):
                for k in range(j+1,gencols_numb-x+1):
                    for l in range(k+1,gencols_numb-x+2):
                        for m in range(l+1,gencols_numb):
                            #print(i+1,j+1,k+1,l+1,m+1)
                            if len(matrix)==vars_numb:
                                break
                            else:
                                matrix.append([a*b*c*d*e for a,b,c,d,e in zip(matrix[i],matrix[j],matrix[k],matrix[l],matrix[m])])
                        if len(matrix)==vars_numb:
                            break
                    if len(matrix)==vars_numb:
                        break
                if len(matrix)==vars_numb:
                    break
            x+=1
            if len(matrix)==vars_numb:
                break
        print("Numero di colonne dopo i penta prodotti = %s\n" % (len(matrix)))
        if settings.VERBOSE:
            for e in range(len(matrix[0])):
                print('\t'.join([str(matrix[x][e]) for x in range(len(matrix))]))
        return matrix, exps_numb
    
    import operator
    print ("\nCalcolo degli Effetti Principali...\n")
    x1 = ['V','S','R','G','POL','MW']
    x2 = ['W1','W2','W3','W4','W5','W6','W7','W8','IW1','IW2','IW3','IW4','CW1','CW2','CW3','CW4','CW5','CW6','CW7','CW8']
    x3 = ['D1','D2','D3','D4','D5','D6','D7','D8','ID1','ID2','ID3','ID4','CD1','CD2','CD3','CD4','CD5','CD6','CD7','CD8']
    x4 = ['WO1','WO2','WO3','WO4','WO5','WO6']
    x5 = ['WN1','WN2','WN3','WN4','WN5','WN6']
    x6 = ['DIFF','LOGP n-Oct','LOGP c-Hex','AUS7.4','SOLY','PB','VD','CACO2','SKIN','LgBB','MetStab','HTSflag']
    x7 = ['LgD5','LgD6','LgD7','LgD7.5','LgD8','LgD9','LgD10']
    x8 = ['LgS3','LgS4','LgS5','LgS6','LgS7','LgS7.5','LgS8','LgS9','LgS10','LgS11','L0LgS','L1LgS','L2LgS','L3LgS','L4LgS']
    x9 = ['%FU4','%FU5','%FU6','%FU7','%FU8','%FU9','%FU10']
    x10 = ['HL1','HL2','A','CP']
    x11 = ['FLEX','FLEX_RB','NCC','PSA','HSA','PSAR','PHSAR']
    
    if settings.EFFETTIPRINCIPALIBLOCK:
        B = x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11
    else:
        B = [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11]
    
    S, exps_numb = generate_signs_matrix(vars_numb=len(B))
    
    if settings.VERBOSE:
        print("\nEsperimenti in corso...\nExp\tObjs\tVars\tQ2")
    
    Scores=[]
    for e in range(exps_numb):
        Xe=[]
        signs_e = [S[s][e] for s in range(len(S))]
        for x in range(len(B)):
            if signs_e[x] == 1:
                if settings.EFFETTIPRINCIPALIBLOCK:
                    Xe.append(B[x])
                else:
                    Xe+=B[x]
        Xe_train = [[X_train[key][i] for key in list(X_train.keys()) if key in Xe] for i in range(len(Y_train))]
        
        score = leave_one_out(X=Xe_train, Y=Y_train, model=model)
        Scores.append(score)
        if settings.VERBOSE:
            print("%s\t%s\t%s\t%s" % (e, len(Xe_train), len(Xe_train[0]), score))
    E={}
    for i, s in enumerate(S):
        ap, an = 0, 0
        for e in range(exps_numb):
            if s[e]==1:
                ap+=Scores[e]
            else:
                an+=Scores[e]
        E[i] = ap/(exps_numb/2)-an/(exps_numb/2)
    print("\nEffetti Principali:\n(block, score)\n"+'\n'.join([str(item) for item in sorted(E.items(), key=operator.itemgetter(1), reverse=True)]))
    
    
    if settings.EFFETTIPRINCIPALILIMIT != None:
        limit = settings.EFFETTIPRINCIPALILIMIT
    else:
        limit = 0
    Xr=[]
    for key in sorted(list(E.keys())):
        if E[key] > limit:
            if settings.EFFETTIPRINCIPALIBLOCK:
                Xr.append(B[key])
            else:
                Xr+=B[key]
    Xr_train = [[X_train[key][i] for key in list(X_train.keys()) if key in Xr] for i in range(len(Y_train))]
    
    print("\nModello finale:\nNumero di variabili = %s\nQ2 = %s\n" % (len(Xr), round(leave_one_out(X=Xr_train, Y=Y_train, model=model),3)))
    return Xr, Xr_train

def calc_test_metrics(LV, Y_exp, Y_pred, filename='Ypred_LV%s.csv'):
    thresholds = [[x,y] for x in np.arange(0, 1.01, 0.01).tolist() for y in np.arange(0, 1.01, 0.01).tolist()]
    ocsv=open(filename % (LV), 'w')
    ocsv.write('low threshold\thigh threshold\tTP\tFN\tTN\tFP\tUNC\tACC\tSE\tSP\tpPREC\tnPREC\tF1\tMCC\tTotal coverage\tP coverage\tN coverage\n')
    t_list=[]
    for t in thresholds:
        if sorted(t) not in t_list:
            t_list.append(t)
            Yr, Ye = [], []
            UNC=0
            for y in range(len(Y_pred)):
                if Y_pred[y] > t[1]:
                    Ye.append(Y_exp[y])
                    Yr.append(1)
                elif Y_pred[y] < t[0]:
                    Ye.append(Y_exp[y])
                    Yr.append(0)
                else:
                    UNC+=1
            try:
                TN, FP, FN, TP = confusion_matrix(Ye, Yr).ravel()
                ACC, F1, MCC = round(accuracy_score(Ye, Yr),2), round(f1_score(Ye, Yr),2), round(matthews_corrcoef(Ye, Yr),2)
                pPREC, nPREC = round(precision_score(Ye, Yr),2), round(TN/(TN+FN),2)
                SE, SP = round(recall_score(Ye, Yr),2), round(TN/(TN + FP),2)
                COV, Pcov, Ncov = round((len(Y_exp)-UNC)*100/len(Y_exp),1), round(Ye.count(1)*100/Y_exp.count(1),1), round(Ye.count(0)*100/Y_exp.count(0),1)
                ocsv.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (t[0],t[1],TP,FN,TN,FP,UNC,ACC,SE,SP,pPREC,nPREC,F1,MCC,COV,Pcov,Ncov))
            except:
                pass
    ocsv.close()

def save_pls_predictions(model, X_train, X_test, Y_train, Y_test, O_train, O_test, filename="predictions.csv"):
    Yp_train, Yp_test = [y[0] for y in model.predict(X_train).tolist()], [y[0] for y in model.predict(X_test).tolist()]
    ocsv=open(filename, 'w')
    ocsv.write("Object;Set;Y_exp;Y_pred;Classification (%s/%s)\n" % (settings.LOWTHRESHOLD, settings.HIGHTHRESHOLD))
    for o in range(len(O_train)):
        if Yp_train[o] > settings.HIGHTHRESHOLD:
            if Y_train[o]==1:
                cla='TP'
            else:
                cla='FP'
        elif Yp_train[o] < settings.LOWTHRESHOLD:
            if Y_train[o]==0:
                cla='TN'
            else:
                cla='FN'
        else:
            cla='NOTPRED'
        ocsv.write("%s;TRAINING;%s;%s;%s\n" % (O_train[o], Y_train[o], Yp_train[o], cla))
    for o in range(len(O_test)):
        if Yp_test[o] > settings.HIGHTHRESHOLD:
            if Y_test[o]==1:
                cla='TP'
            else:
                cla='FP'
        elif Yp_test[o] < settings.LOWTHRESHOLD:
            if Y_test[o]==0:
                cla='TN'
            else:
                cla='FN'
        else:
            cla='NOTPRED'
        ocsv.write("%s;TEST;%s;%s;%s\n" % (O_test[o], Y_test[o], Yp_test[o], cla))
    ocsv.close()

def linear_model_1y(args):
    settings.ACTIVITY=args.activity
    settings.MODEL=args.model
    settings.FIT=args.fit
    settings.PREDICT=args.predict
    settings.SAVEMODEL=args.savemodel
    settings.LOO = args.loo
    settings.EFFETTIPRINCIPALI = args.effettiprincipali
    settings.EFFETTIPRINCIPALILIMIT=args.effettiprincipalilimit
    settings.EFFETTIPRINCIPALIBLOCK=args.effettiprincipaliblock
    settings.LOWTHRESHOLD=args.lowthreshold
    settings.HIGHTHRESHOLD=args.highthreshold
    settings.LATENT=args.latent
    
    D=Dataset(y=settings.ACTIVITY)
    D.load_training_data(training_set_file=settings.FIT)
    if settings.PREDICT != None:
        D.load_validation_data(valset_file=settings.PREDICT)
    D.autoscale()
    if settings.VERBOSE:
        print(D)
    
    X_train, Y_train, X_test, Y_test = D.X, D.Y, D.X2, D.Y2
    M=Model(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
    if settings.MODEL == 'PLS':
        M.partial_least_squares()
        
        if settings.PREDICT != None:
            calc_test_metrics(LV=M.LV, Y_exp=Y_test, Y_pred=M.P[M.LV])
            
        if settings.LOWTHRESHOLD!=None and settings.HIGHTHRESHOLD!=None:
            save_pls_predictions(model=M.model, X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, O_train=D.Objects, O_test=D.Objects2)
            
    if settings.EFFETTIPRINCIPALI:
        X1={x: [X_train[a][i] for a in range(len(X_train))] for i,x in enumerate(D.X_labels)}
        X2={x: [X_test[a][i] for a in range(len(X_test))] for i,x in enumerate(D.X_labels)}
        selected_vars_list, X1_reduced = effetti_principali(X_train=X1, Y_train=Y_train, X_test=X2, Y_test=Y_test, model=M.model)
        M.model.fit(X1_reduced, Y_train)
        print(";".join(selected_vars_list))
        calc_test_metrics(LV=M.LV, Y_exp=Y_train, Y_pred=[y[0] for y in M.model.predict(X1_reduced).tolist()], filename='reduced_Yrecal_LV%s.csv')
        X2_reduced=[[X2[key][i] for key in list(X2.keys()) if key in selected_vars_list] for i in range(len(Y_test))]
        calc_test_metrics(LV=M.LV, Y_exp=Y_test, Y_pred=[y[0] for y in M.model.predict(X2_reduced).tolist()], filename='reduced_Ypred_LV%s.csv')
        if settings.LOWTHRESHOLD!=None and settings.HIGHTHRESHOLD!=None:
            save_pls_predictions(model=M.model, X_train=X1_reduced, X_test=X2_reduced, Y_train=Y_train, Y_test=Y_test, O_train=D.Objects, O_test=D.Objects2, filename="reduced_predictions.csv")
        
    if settings.SAVEMODEL:
        M.save_model()
    
