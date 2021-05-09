
import os, shutil, itertools, tempfile
import numpy as np
import multiprocessing as mp
import pandas as pd

from core import settings, parameters, conditions, variables
from core.validate import cross_validation, get_balanced_accuracy_cv
from core.classification import train_the_model
from core.regression import run_model_training

m=mp.Manager()
q=m.Queue()


def compute_model(model):
    try:
        model.fit(variables.X_tra, variables.Y_tra)
    except:
        aver_score = -99
    else:
        if variables.N_classes != None: scores = get_balanced_accuracy_cv(X=variables.X_tra, Y=variables.Y_tra, M=model)
        else: scores = cross_validation(X=variables.X_tra, Y=variables.Y_tra, M=model)
        aver_score=round(np.mean(scores),3)
    
    q.put(1)
    size=q.qsize()
    advance=int(size*100/variables.n)
    if advance!=variables.m:
        variables.m=advance
        print("completed %s/100 (%s/%s models)" % (variables.m, size, variables.n))
        
    if variables.N_classes > 2: params = { k.split('__')[-1]: model.get_params()[k] for k in list(model.get_params().keys()) }
    else: params = model.get_params()
    
    ocsv=open(os.path.join(variables.workdir, str(size))+".csv", "w")
    ocsv.write(';'.join([str(size)] + [str(params[k]) for k in sorted(list(params.keys())) if k in variables.N_list] + [str(aver_score)]) + '\n')
    ocsv.close()


def run_optimization_cv():
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
            elif p=='class_weight': parameters.class_weight = combo[p]
            elif p=='loss': parameters.loss = combo[p]
            elif p=='algorithm_ab': parameters.algorithm_ab = combo[p]
            elif p=='algorithm_knn': parameters.algorithm_knn = combo[p]
            elif p=='shrinkage': parameters.shrinkage = combo[p]
            elif p=='solver_lda': parameters.solver_lda = combo[p]
            elif p=='solver_mlp': parameters.solver_mlp = combo[p]
            elif p=='C': parameters.C = combo[p]
            elif p=='kernel': parameters.kernel = combo[p]
            elif p=='gamma': parameters.gamma = combo[p]
            elif p=='degree': parameters.degree = combo[p]
            elif p=='n_neighbors': parameters.n_neighbors = combo[p]
            elif p=='weights': parameters.weights = combo[p]
            elif p=='leaf_size': parameters.leaf_size = combo[p]
            elif p=='p': parameters.p = combo[p]
            elif p=='activation': parameters.activation = combo[p]
            elif p=='learning_rate': parameters.learning_rate = combo[p]
            elif p=='learning_rate_init': parameters.learning_rate_init = combo[p]
            elif p=='hidden_layer_sizes': parameters.hidden_layer_sizes = combo[p]
            elif p=='alpha': parameters.alpha = combo[p]
            elif p=='power_t': parameters.power_t = combo[p]
            elif p=='momentum': parameters.momentum = combo[p]
            elif p=='nesterovs_momentum': parameters.nesterovs_momentum = combo[p]
            elif p=='beta_1': parameters.beta_1 = combo[p]
            elif p=='beta_2': parameters.beta_2 = combo[p]
            elif p=='epsilon': parameters.epsilon = combo[p]
            elif p=='max_iter': parameters.max_iter = combo[p]
            elif p=='min_samples_split': parameters.min_samples_split = combo[p]
            elif p=='min_samples_leaf': parameters.min_samples_leaf = combo[p]
            
            if p.split("_")[0] in ["criterion", "algorithm", "solver"]: variables.N_list.append(p.split('_')[0])
            else: variables.N_list.append(p)
        
        if settings.MODEL=="LDA":
            if parameters.solver_lda=="svd":
                if parameters.shrinkage==None: counter=0
                else: counter=1
            else: counter=0
        
        elif settings.MODEL=="kNN":
            if parameters.algorithm_knn not in ['ball_tree', 'kd_tree']:
                if parameters.leaf_size==30: counter=0
                else: counter=1
            else: counter=0
        
        elif settings.MODEL=="SVM":
            if parameters.gamma == 'scale': counter=1
            else:
                if parameters.kernel == 'linear':
                    if parameters.gamma == 'auto': counter=0
                    else: counter=1
                elif parameters.kernel != 'poly':
                    if parameters.degree==3: counter=0
                    else: counter=1
                else: counter=0
        
        elif settings.MODEL=="MLP":
            if parameters.solver_mlp == 'lbfgs':
                if parameters.learning_rate == 'constant':
                    if parameters.learning_rate_init == 0.001 and parameters.power_t == 0.5 and parameters.momentum == 0.9 and parameters.nesterovs_momentum and parameters.beta_1 == 0.9 and parameters.beta_2 == 0.999 and parameters.epsilon == 1e-8: counter=0
                    else: counter=1
                else: counter=1
            elif parameters.solver_mlp == 'adam':
                if parameters.learning_rate == 'constant':
                    if parameters.power_t == 0.5 and parameters.momentum == 0.9 and parameters.nesterovs_momentum: counter=0
                    else: counter=1
                else: counter=1
            else:
                if parameters.learning_rate != 'invscaling':
                    if parameters.power_t == 0.5 and parameters.beta_1 == 0.9 and parameters.beta_2 == 0.999 and parameters.epsilon == 1e-8: counter=0
                    else: counter=1
                else:
                    if parameters.beta_1 == 0.9 and parameters.beta_2 == 0.999 and parameters.epsilon == 1e-8: counter=0
                    else: counter=1
        
        else: counter=0
        
        
        if counter==0:
            if variables.N_classes!=None: model, model_name = train_the_model()
            else: run_model_training()
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
    if variables.N_classes != None: outfile.write(';'.join(['model_id'] + sorted(names) + ['BA\n']))
    #else: outfile.write(';'.join(['model_id'] + sorted(names) + ['SE','SP','MCC\n']))
    else: outfile.write(';'.join(['model_id'] + sorted(names) + ['R2\n']))
    for f in os.listdir(workdir):
        line=open(os.path.join(workdir,f)).readline()
        outfile.write(line)
    outfile.close()
    shutil.rmtree(workdir)
    
    if variables.N_classes != None: df_results = pd.read_csv("opt_results_%s.csv" % settings.MODEL, sep=";", header=0, index_col=0).sort_values(by='BA', ascending=False)
    else: df_results = pd.read_csv("opt_results_%s.csv" % settings.MODEL, sep=";", header=0, index_col=0).sort_values(by='R2', ascending=False)
    print(df_results)
    
