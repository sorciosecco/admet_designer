
import os, sys
import time
import numpy as np
import multiprocessing as mp

from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

m=mp.Manager()
q=m.Queue()
Nmods=0

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

class Process:
    def __init__(self, X_train, Y_train, X_test, Y_test, A):
        self.X_train=X_train
        self.X_test=X_test
        self.Y_train=Y_train
        self.Y_test=Y_test
        self.A=A
        self.Parameters=[]
        self.header=''
        
    def define_parameters(self):
        if self.A=="LDA":
            param_list = ['shrinkage', 'solver']
            shrinkage=np.arange(0.01, 1.01, 0.01).tolist()+[None,'auto']
            solver=['svd','lsqr','eigen']
            
            d=0
            for p1 in shrinkage:
                for p2 in solver:
                    p = [p1,p2]
                    D={param_list[k]: p[k] for k in range(len(param_list))}
                    D['model_id'], D['X_train'],  D['X_test'], D['Y_train'], D['Y_test'] = str(d), self.X_train, self.X_test, self.Y_train, self.Y_test
                    self.Parameters.append(D)
                    d+=1
        if self.A=="SVM":
            param_list = ['C', 'kernel', 'gamma', 'degree']
            
            C = np.arange(1, 11, 1).tolist()
            kernel=['linear', 'poly', 'rbf', 'sigmoid']
            gamma=['auto','scale']+ np.arange(0.001, 0.016, 0.001).tolist()
            degree=[1,2,3]
            
            d=0
            for p1 in C:
                for p2 in kernel:
                    for p3 in gamma:
                        for p4 in degree:
                            p = [p1,p2,p3,p4]
                            D={param_list[k]: p[k] for k in range(len(param_list))}
                            D['model_id'], D['X_train'],  D['X_test'], D['Y_train'], D['Y_test'] = str(d), self.X_train, self.X_test, self.Y_train, self.Y_test
                            self.Parameters.append(D)
                            d+=1
        if self.A=="RF":
            param_list = ['n_estimators','criterion','max_features','max_leaf_nodes','class_weight','max_depth']
        
            criterion = ['gini', 'entropy']
            max_features = ['auto', 'log2', None]
            class_weight = [None, 'balanced', 'balanced_subsample']
            n_estimators = np.arange(10, 210, 10).tolist()
            max_leaf_nodes = [None] + np.arange(5, 55, 5).tolist()
            max_depth=[None]+np.arange(3, 11, 1).tolist()
        
            d=0
            for p1 in n_estimators:
                for p2 in criterion:
                    for p4 in max_features:
                        for p5 in max_leaf_nodes:
                            for p6 in class_weight:
                                for p7 in max_depth:
                                    p = [p1,p2,p4,p5,p6,p7]
                                    D={param_list[k]: p[k] for k in range(len(param_list))}
                                    D['model_id'], D['X_train'],  D['X_test'], D['Y_train'], D['Y_test'] = str(d), self.X_train, self.X_test, self.Y_train, self.Y_test
                                    self.Parameters.append(D)
                                    d+=1
        if self.A=="GB":
            param_list = ['n_estimators','max_features','max_leaf_nodes','max_depth']
        
            max_features = ['auto', 'log2', None]
            n_estimators = np.arange(10, 210, 10).tolist()
            max_leaf_nodes = [None] + np.arange(5, 55, 5).tolist()
            max_depth=[None]+np.arange(3, 11, 1).tolist()
        
            d=0
            for p1 in n_estimators:
                for p3 in max_features:
                    for p4 in max_leaf_nodes:
                        for p5 in max_depth:
                            p = [p1,p3,p4,p5]
                            D={param_list[k]: p[k] for k in range(len(param_list))}
                            D['model_id'], D['X_train'],  D['X_test'], D['Y_train'], D['Y_test'] = str(d), self.X_train, self.X_test, self.Y_train, self.Y_test
                            self.Parameters.append(D)
                            d+=1
        if self.A=="ETC":
            param_list=['n_estimators','criterion','max_features','max_leaf_nodes','class_weight','max_depth','bootstrap']
            
            criterion = ['gini', 'entropy']
            max_features = ['auto', 'log2', None]
            class_weight = [None, 'balanced', 'balanced_subsample']
            n_estimators = np.arange(10, 210, 10).tolist()
            max_leaf_nodes = [None] + np.arange(5, 55, 5).tolist()
            max_depth=[None]+np.arange(3, 11, 1).tolist()
            bootstrap=[False,True]
            
            d=0
            for p1 in n_estimators:
                for p2 in criterion:
                    for p4 in max_features:
                        for p5 in max_leaf_nodes:
                            for p6 in class_weight:
                                for p7 in max_depth:
                                    for p8 in bootstrap:
                                        p = [p1,p2,p4,p5,p6,p7,p8]
                                        D={param_list[k]: p[k] for k in range(len(param_list))}
                                        D['model_id'], D['X_train'],  D['X_test'], D['Y_train'], D['Y_test'] = str(d), self.X_train, self.X_test, self.Y_train, self.Y_test
                                        self.Parameters.append(D)
                                        d+=1
        if self.A=="AB":
            param_list = ['n_estimators', 'algorithm']
            
            algorithm=['SAMME', 'SAMME.R']
            n_estimators=np.arange(10, 510, 10).tolist()
            
            d=0
            for p1 in n_estimators:
                for p2 in algorithm:
                    p = [p1,p2]
                    D={param_list[k]: p[k] for k in range(len(param_list))}
                    D['model_id'], D['X_train'],  D['X_test'], D['Y_train'], D['Y_test'] = str(d), self.X_train, self.X_test, self.Y_train, self.Y_test
                    self.Parameters.append(D)
                    d+=1
        self.header='model_id;'+';'.join(sorted(param_list))+';ACC;SE;SP;F1;MCC;time (sec)\n'

def define_model(P, alg=sys.argv[3]):
    if alg=="RF":
        model=RandomForestClassifier(n_estimators=P['n_estimators']
                                     , criterion=P['criterion']
                                     , max_features=P['max_features']
                                     , max_leaf_nodes=P['max_leaf_nodes']
                                     , class_weight=P['class_weight']
                                     , max_depth=P['max_depth']
                                     , random_state=666
                                     )
    if alg=="GB":
        model=GradientBoostingClassifier(n_estimators=P['n_estimators']
                                     , max_features=P['max_features']
                                     , max_leaf_nodes=P['max_leaf_nodes']
                                     , max_depth=P['max_depth']
                                     , random_state=666
                                     )
    if alg=="SVM":
        model=SVC(C=P['C']
                  , kernel=P['kernel']
                  , gamma=P['gamma']
                  , degree=P['degree']
                  , random_state=666
                  )
    if alg=="AB":
        from sklearn.tree import DecisionTreeClassifier
        model=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1)
                                 , n_estimators=P['n_estimators']
                                 , algorithm=P['algorithm']
                                 , random_state=666
                                 )
    if alg=="ETC":
        model=ExtraTreesClassifier(n_estimators=P['n_estimators']
                                   , criterion=P['criterion']
                                   , max_features=P['max_features']
                                   , max_leaf_nodes=P['max_leaf_nodes']
                                   , class_weight=P['class_weight']
                                   , max_depth=P['max_depth']
                                   , random_state=666
                                   , bootstrap=['bootstrap']
                                   )
    if alg=="LDA":
        model=LinearDiscriminantAnalysis(shrinkage=P['shrinkage']
                                         , solver=P['solver']
                                         )
    return model

def run_multithread(params):
    
    def calc_class_metrics(Y_test, X_test, model):
        Y_pred=model.predict(X_test).tolist()
        TN, FP, FN, TP = confusion_matrix(Y_test, Y_pred).ravel()
        ACC = round(accuracy_score(Y_test, Y_pred),2)
        SE = round(recall_score(Y_test, Y_pred),2)
        F1 = round(f1_score(Y_test, Y_pred),2)
        MCC = round(matthews_corrcoef(Y_test, Y_pred),2)
        SP = round(TN/(TN + FP),2)
        return ACC,SE,SP,F1,MCC
    
    t0=time.time()
    
    model = define_model(params)
    
    try:
        model.fit(params['X_train'], params['Y_train'])
    except:
        line = [params['model_id']] + [str(params[key]) for key in sorted(list(params.keys())) if key not in  ['model_id','X_train','Y_train','X_test','Y_test']] + ['fail']*5 + [str(round(time.time()-t0))]
    else:
        ACC2,SE2,SP2,F12,MCC2 = calc_class_metrics(X_test=params['X_test'], Y_test=params['Y_test'], model=model)
        
        line2 = [params['model_id']] + [str(params[key]) for key in sorted(list(params.keys())) if key not in  ['model_id','X_train','Y_train','X_test','Y_test']] + [str(x) for x in [ACC2,SE2,SP2,F12,MCC2]] + [str(round(time.time()-t0))]
        
        ofile = open('model_id'+params['model_id']+'.txt', 'w')
        ofile.write(';'.join(line2)+'\n')
        ofile.close()
    
    q.put(params['model_id'])
    size=q.qsize()
    print('['+str(round(size*100/Nmods,2))+' %] of models completed')
    return 'model_id'+params['model_id']+'.txt'

def save_results(files, header):
    print('\nSaving results...')
    ocsv=open(os.path.join(os.getcwd(), 'opt_results_.csv'), 'w')
    ocsv.write(header)
    for file_i in files:
        try:
            i=open(file_i,'r')
            line=i.readline()
            i.close()
            ocsv.write(line)
        except:
            print("not file")
        else:
            os.remove(file_i)
    ocsv.close()

if __name__=="__main__":
    D=Dataset(y="SOLU")
    D.load_training_data(training_set_file=sys.argv[1])
    D.load_validation_data(valset_file=sys.argv[2])
    D.autoscale()
    
    Proc=Process(X_train=D.X, Y_train=D.Y, X_test=D.X2, Y_test=D.Y2, A=sys.argv[3])
    Proc.define_parameters()
    
    print("\nSetting up the optimization process...\n\nnumber of models to compute = %s" % len(Proc.Parameters))
    Nmods=len(Proc.Parameters)
    pool=mp.Pool(mp.cpu_count())
    results=pool.map_async(run_multithread, Proc.Parameters)
    pool.close()
    pool.join()
    #for param in P.Parameters:
        #run_multithread(param)
    
    save_results(files=results.get(), header=Proc.header)
    
