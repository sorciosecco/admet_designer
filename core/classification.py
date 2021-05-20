##### NB: rNN has no probabilities!!!

from core import settings, parameters, variables
#from core.save import save_model, save_vars_importance
from core.validate import *

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier#, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier


def train_the_model():
    Xe, Ye = variables.X_tra, variables.Y_tra
    
    if settings.NPARA==False: ##### ALGORITHMS WITH DEFAULT PARAMETERS
        if settings.MODEL=="RF": model=RandomForestClassifier(random_state=settings.SEED, n_estimators=100, class_weight=None, criterion='gini', max_depth=None, max_features='auto', max_leaf_nodes=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, warm_start=False)
        elif settings.MODEL=="AB": model=AdaBoostClassifier(random_state=settings.SEED, base_estimator=None, n_estimators=100, algorithm='SAMME.R', learning_rate=1.0)
        elif settings.MODEL=="ET": model=ExtraTreesClassifier(random_state=settings.SEED, bootstrap=True, n_estimators=100, class_weight=None, criterion='gini', max_depth=None, max_features='auto', max_leaf_nodes=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, min_impurity_decrease=0.0, min_impurity_split=None, oob_score=False, warm_start=False)
        elif settings.MODEL=="GB": model=GradientBoostingClassifier(random_state=settings.SEED, n_estimators=100, max_depth=3, max_features=None, max_leaf_nodes=None, learning_rate=0.1, loss='deviance', subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, min_impurity_decrease=0.0, min_impurity_split=None, init=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)
        elif settings.MODEL=="kNN": model=KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None)
        #elif settings.MODEL=="rNN": model=RadiusNeighborsClassifier(radius=1.0, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None)
        elif settings.MODEL=="LDA": model=LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=None, n_components=None, store_covariance=False, tol=0.0001)
        elif settings.MODEL=="MLP": model=MLPClassifier(random_state=settings.SEED, hidden_layer_sizes=(100,), activation='relu', solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, shuffle=True, tol=0.0001, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_iter=500)
        elif settings.MODEL=="SVM": model=SVC(random_state=settings.SEED, probability=True, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, class_weight=None, decision_function_shape='ovr')
        else: print("\nERROR: algorithm not supported\n")
        
    else: ##### ALGORITHMS WITH NON-DEFAULT PARAMETERS
        if settings.MODEL=="RF": model=RandomForestClassifier(random_state=settings.SEED, n_estimators=parameters.n_estimators, class_weight=parameters.class_weight, criterion=parameters.criterion_rf, max_depth=parameters.max_depth, max_features=parameters.max_features, min_samples_split=parameters.min_samples_split, min_samples_leaf=parameters.min_samples_leaf)
        elif settings.MODEL=="AB": model=AdaBoostClassifier(random_state=settings.SEED, n_estimators=parameters.n_estimators, algorithm=parameters.algorithm_ab)
        elif settings.MODEL=="ET": model=ExtraTreesClassifier(random_state=settings.SEED, bootstrap=True, n_estimators=parameters.n_estimators, class_weight=parameters.class_weight, criterion=parameters.criterion_rf, max_depth=parameters.max_depth, max_features=parameters.max_features, min_samples_split=parameters.min_samples_split, min_samples_leaf=parameters.min_samples_leaf)
        elif settings.MODEL=="GB": model=GradientBoostingClassifier(random_state=settings.SEED, n_estimators=parameters.n_estimators, max_depth=parameters.max_depth, max_features=parameters.max_features, loss=parameters.loss, criterion=parameters.criterion_gb, min_samples_split=parameters.min_samples_split, min_samples_leaf=parameters.min_samples_leaf)
        elif settings.MODEL=="kNN": model=KNeighborsClassifier(n_neighbors=parameters.n_neighbors, leaf_size=parameters.leaf_size, algorithm=parameters.algorithm_knn, p=parameters.p, weights=parameters.weights)
        #elif settings.MODEL=="rNN": model=RadiusNeighborsClassifier(radius=parameters.radius, p=parameters.p, weights=parameters.weights)
        elif settings.MODEL=="LDA": model=LinearDiscriminantAnalysis(solver=parameters.solver_lda, shrinkage=parameters.shrinkage)
        elif settings.MODEL=="MLP": model=MLPClassifier(random_state=settings.SEED, solver=parameters.solver_mlp, activation=parameters.activation, learning_rate_init=parameters.learning_rate_init, learning_rate=parameters.learning_rate, hidden_layer_sizes=parameters.hidden_layer_sizes,  max_iter=parameters.max_iter, alpha=parameters.alpha, power_t=parameters.power_t, momentum=parameters.momentum, nesterovs_momentum=parameters.nesterovs_momentum, beta_1=parameters.beta_1, beta_2=parameters.beta_2, epsilon=parameters.epsilon)
        elif settings.MODEL=="SVM": model=SVC(random_state=settings.SEED, C=parameters.C, degree=parameters.degree, gamma=parameters.gamma, kernel=parameters.kernel, class_weight=parameters.class_weight)
        else: print("\nERROR: algorithm not supported\n")
    
    model_name=settings.MODEL
    if settings.MODEL!="PLS":
        #if variables.DMODY:
            #q2, sdep, variables.Y_pred = leave_one_out(X=np.array(Xe), Y=np.array(Ye), M=model)
            #print("Q2\tSDEP\n%s\t%s" % (round(q2,3), round(sdep,3)))
        if variables.N_classes > 2:
            if settings.MULTICLASS:
                model_name+="_1vs1.model"
                model=OneVsOneClassifier(model, n_jobs=1)
            else:
                model_name+="_1vsRest.model"
                model=OneVsRestClassifier(model, n_jobs=1)
        
        variables.model = model
        
        if settings.OPTIMIZE==False:
            if settings.LEAVEONEOUT: variables.Y_pred = multi_class_loo(X=Xe, Y=Ye)
            else: multi_class_cv(X=Xe, Y=Ye)
            model.fit(Xe, Ye)
    return model, model_name


def run_modelling_steps(X_train, X_test, Y_train, Y_test, model_type, nondef_params, sm, mc):
    
    model=algorithm_setup(model_type, nondef_params)# This defines the model
    
    if mc:
        model=OneVsRestClassifier(model, n_jobs=1)
        #model=OneVsOneClassifier(model, n_jobs=1)
        
    model.fit(X_train, Y_train)# Fit the model
    
    Y1_pred, Y2_pred, Y1_prob, Y2_prob = model.predict(X_train).tolist(), model.predict(X_test).tolist(), model.predict_proba(X_train).tolist(), model.predict_proba(X_test).tolist()
    
    if settings.SAVEVARS: save_vars_importance(V=settings.VARS, I=model.feature_importances_)
    if sm: save_model(m=model, mt=model_type)
    
    return Y1_pred, Y2_pred, Y1_prob, Y2_prob
