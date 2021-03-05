
import os, shutil, itertools, tempfile
import numpy as np
import multiprocessing as mp
import pandas as pd
#from sklearn.model_selection import KFold
#from sklearn.metrics import matthews_corrcoef, recall_score, confusion_matrix, f1_score, balanced_accuracy_score

from core import settings, parameters, conditions, variables
from core.validate import cross_validation
from core.classification import define_model_for_optimization
from core.regression import run_model_training

m=mp.Manager()
q=m.Queue()

#def cross_validation(x, y, cv, model):
    #X, Y = np.array(x), np.array(y)
    #kf = KFold(n_splits=cv, shuffle=True, random_state=settings.SEED)
    #scores, SE, SP, MCC = [], [], [], []
    #for train_index, test_index in kf.split(X):
        #X_train, X_test = X[train_index], X[test_index]
        #y_train, y_test = Y[train_index], Y[test_index]
        #model.fit(X_train, y_train)
        #y_pred=model.predict(X_test).tolist()
        #if settings.MULTICLASS:
            #s = sum([1 for x in range(len(y_test)) if y_pred[x]!=y_test[x]]) / len(y_test)
            #scores.append(s)
        #else:
            #TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
            #se, sp = recall_score(y_test, y_pred), TN/(TN + FP)
            #mcc = matthews_corrcoef(y_test, y_pred)
            #SE.append(se)
            #SP.append(sp)
            #MCC.append(mcc)
    #if settings.MULTICLASS:
        #return [round(np.mean(scores),2)]
    #else:
        #return [np.mean(SE), np.mean(SP), np.mean(MCC)]


def compute_model(model):
    try:
        model.fit(variables.X_tra, variables.Y_tra)
    except:
        aver_r2 = -99
    else:
        scores = cross_validation(X=variables.X_tra, Y=variables.Y_tra, M=model)
        aver_r2=np.mean(scores)
        #Yp=m.predict(X2).tolist()
        #scores = [round(f1, 2) for f1 in f1_score(Y2, Yp, average=None)]+[round(balanced_accuracy_score(Y2, Yp), 2)]
    
    q.put(1)
    size=q.qsize()
    advance=int(size*100/variables.n)
    if advance!=variables.m:
        variables.m=advance
        print("completed %s/100 (%s/%s models)" % (variables.m, size, variables.n))
        
    #if settings.MULTICLASS:
        #params = { k.split('__')[-1]: m.get_params()[k] for k in list(m.get_params().keys()) }
    #else:
    params = model.get_params()
    
    ocsv=open(os.path.join(variables.workdir, str(size))+".csv", "w")
    ocsv.write(';'.join([str(size)] + [str(params[k]) for k in sorted(list(params.keys())) if k in variables.N_list] + [str(aver_r2)]) + '\n')
    ocsv.close()


def run_grid_cross_validation():
    settings.NPARA=True
    grid = conditions.param_grids[settings.MODEL]
    names, values = list(grid.keys()), list(itertools.product(*grid.values()))
    
    origdir=os.getcwd()
    workdir=tempfile.mkdtemp(prefix='TMP')
    os.chdir(workdir)
    variables.workdir=workdir
    
    variables.N_list, M_list = [], []
    for v in values:
        combo={names[i]: v[i] for i in range(len(v))}
        for p in list(combo.keys()):
            if p=='n_estimators': parameters.n_estimators = combo[p]
            elif p=='criterion_rf': parameters.criterion_rf = combo[p]
            elif p=='criterion_gb': parameters.criterion_gb = combo[p]
            elif p=='max_features': parameters.max_features = combo[p]
            elif p=='max_depth': parameters.max_depth = combo[p]
            elif p=='max_leaf_nodes': parameters.max_leaf_nodes = combo[p]
            elif p=='class_weight': parameters.class_weight = combo[p]
            elif p=='loss': parameters.loss = combo[p]
            elif p=='algorithm_ab': parameters.algorithm_ab = combo[p]
            elif p=='algorithm_knn': parameters.algorithm_knn = combo[p]
            elif p=='shrinkage': parameters.shrinkage = combo[p]
            elif p=='solver': parameters.solver = combo[p]
            elif p=='C': parameters.C = combo[p]
            elif p=='kernel': parameters.kernel = combo[p]
            elif p=='gamma': parameters.gamma = combo[p]
            elif p=='degree': parameters.degree = combo[p]
            elif p=='n_neighbors': parameters.n_neighbors = combo[p]
            elif p=='weights': parameters.weights = combo[p]
            elif p=='leaf_size': parameters.leaf_size = combo[p]
            elif p=='p': parameters.p = combo[p]
            
            if p.startswith("criterion") or p.startswith("algorithm"): variables.N_list.append(p.split('_')[0])
            else: variables.N_list.append(p)
        
        #if settings.MODEL in ["RF", "ETC", "GB"]:
            #if parameters.max_leaf_nodes != None:
                #if parameters.max_leaf_nodes != counter:
                    #model = define_model_for_optimization(mt=settings.MODEL, ndp=True, mc=settings.MULTICLASS)
                    #models.append((X1, Y1, X2, Y2, model))
                    #couter = parameters.max_leaf_nodes
            #else:
                #model = define_model_for_optimization(mt=settings.MODEL, ndp=True, mc=settings.MULTICLASS)
                #models.append((X1, Y1, X2, Y2, model))
        #elif settings.MODEL == "LDA":
            #if parameters.solver == "svd":
                #if parameters.shrinkage == None:
                    #model = define_model_for_optimization(mt=settings.MODEL, ndp=True, mc=settings.MULTICLASS)
                    #models.append((X1, Y1, X2, Y2, model))
            #else:
                #model = define_model_for_optimization(mt=settings.MODEL, ndp=True, mc=settings.MULTICLASS)
                #models.append((X1, Y1, X2, Y2, model))
        
        if settings.MODEL=="kNN":
            if parameters.algorithm_knn not in ['ball_tree', 'kd_tree']:
                if parameters.leaf_size==30: counter=0
                else: counter=1
            else:
                counter=0
        
        elif settings.MODEL=="SVM":
            if parameters.gamma == "scale": counter=1
            else:
                if parameters.kernel == "linear":
                    if parameters.gamma == "auto": counter=0
                    else: counter=1
                elif parameters.kernel != "poly":
                    if parameters.degree==3: counter=0
                    else: counter=1
                else:
                    counter=0
        else:
            counter=0
        
        #ADD IT BEFORE THIS LINE
        if counter==0:
            run_model_training()
            M_list.append(variables.model)
    
    variables.n=len(M_list)
    #for m in M_list:
        #compute_model(m)
    pool=mp.Pool(mp.cpu_count())
    pool.map_async(compute_model, M_list)
    pool.close()
    pool.join()
    
    os.chdir(origdir)
    outfile=open("opt_results_%s.csv" % settings.MODEL, "w")
    #if settings.MULTICLASS: outfile.write(';'.join(['model_id'] + sorted(names) + ['F11', 'F12', 'F13', 'F14', 'BA\n']))
    #else: outfile.write(';'.join(['model_id'] + sorted(names) + ['SE','SP','MCC\n']))
    outfile.write(';'.join(['model_id'] + sorted(names) + ['R2\n']))
    for f in os.listdir(workdir):
        line=open(os.path.join(workdir,f)).readline()
        outfile.write(line)
    outfile.close()
    shutil.rmtree(workdir)
    
    df_results = pd.read_csv("opt_results_%s.csv" % settings.MODEL, sep=";", header=0, index_col=0).sort_values(by='R2', ascending=False)
    print(df_results)
    
