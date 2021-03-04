
from core import settings, variables
from core.load import load_datasets
from core.classification import run_modelling_steps
from core.regression import run_model_training
from core.metrics import calculate_class_scores
from core.save import save_predictions, save_multi_predictions
from core.conditions import param_grids
from core.optimization import run_grid_cross_validation


def run_procedure(X1, X2, Y1, Y2, O1, O2):
    Y1_pred, Y2_pred, Y1_prob, Y2_prob = run_modelling_steps(X_train=X1, X_test=X2, Y_train=Y1, Y_test=Y2, model_type=settings.MODEL, nondef_params=settings.NPARA, sm=settings.SAVEMODEL, mc=settings.MULTICLASS)
    
    scores = calculate_class_scores(Y1_exp=Y1, Y1_pred=Y1_pred, Y1_prob=Y1_prob, Y2_exp=Y2, Y2_pred=Y2_pred, Y2_prob=Y2_prob, O1=O1, O2=O2, pc=settings.PROBACUTOFF)
    
    if settings.SAVEPRED:
        if settings.MULTICLASS: save_multi_predictions(name=settings.MODEL, Y1_exp=Y1, Y1_pred=Y1_pred, Y1_prob=Y1_prob, Y2_exp=Y2, Y2_pred=Y2_pred, Y2_prob=Y2_prob, O1=O1, O2=O2)
        else: save_predictions(name=settings.MODEL, Y1_exp=Y1, Y1_pred=Y1_pred, Y1_prob=Y1_prob, Y2_exp=Y2, Y2_pred=Y2_pred, Y2_prob=Y2_prob, O1=O1, O2=O2, pc=settings.PROBACUTOFF)
    else:
        if settings.VERBOSE!=0: print("\n*** Predicted values have not been saved.\n")
    

def build_classification_model(args):
    settings.NPARA=args.npara
    settings.MULTICLASS=args.multiclass
    settings.PROBACUTOFF=args.probacutoff
    settings.SAVEMODEL=args.savemodel
    settings.SAVEPRED=args.savepred
    settings.GRIDSEARCH=args.gridsearch
    #settings.BACKFEEL=args.backfeel
    
    X1, X2, Y1, Y2, O1, O2, V = load_datasets(training=settings.FIT, test=settings.PREDICT, response=settings.RESPONSE)
    settings.VAR_NAMES=V[1:-1]
    settings.VARS=V
    
    #if settings.GRIDSEARCH: gridsearchcv(X1=X1, Y1=Y1, X2=X2, Y2=Y2, grid=param_grids[settings.MODEL])
    #else: run_procedure(X1=X1, X2=X2, Y1=Y1, Y2=Y2, O1=O1, O2=O2)
    run_procedure(X1=X1, X2=X2, Y1=Y1, Y2=Y2, O1=O1, O2=O2)
    

def build_regression_model(args):
    settings.NPARA, settings.OPTIMIZE = args.npara, args.optimize
    #settings.PROBACUTOFF=args.probacutoff
    #settings.SAVEPRED=args.savepred
    if settings.PREDICT==None:
        variables.X_tra, variables.Y_tra, variables.O_list, variables.V_list = load_datasets(training=settings.FIT, response=settings.RESPONSE)
        print("\nTRAINING SET:\nN objects = %s\nN independent vars = %s\nN dependent vars: 1 (%s)\n\nCreating %s model..." % (len(variables.O_list), len(variables.V_list), settings.RESPONSE, settings.MODEL))
        
        #import numpy as np
        #print(np.array(variables.X_tra).var())
        
        run_model_training()
        
        if settings.OPTIMIZE and settings.MODEL!="PLS": run_grid_cross_validation()
        
    else:
        X1, X2, Y1, Y2, O1, O2, V = load_datasets(training=settings.FIT, test=settings.PREDICT, response=settings.RESPONSE)
        print("\nTRAINING SET:\nN objects = %s\nN independent vars = %s\nN dependent vars: 1 (%s)\n\nTEST SET:\nN objects = %s\n" % (len(O1), len(X1[0]), settings.RESPONSE, len(O2)))
        settings.VARS=V
        
        if settings.MODEL=="PLS": tra_Yp, tes_Yp = perform_pls(tra_X=X1, tes_X=X2, tra_Y=Y1, tes_Y=Y2)
        else: tra_Yp, tes_Yp = perform_machine_learning(tra_X=X1, tes_X=X2, tra_Y=Y1, tes_Y=Y2)
        
        tra_scores, tes_scores = calc_regr_metrics(Ye=Y1, Yp=tra_Yp), calc_regr_metrics(Ye=Y2, Yp=tes_Yp)
        
        print("SET\tCORR\tERROR\nTRA\t%s\t%s\nTES\t%s\t%s\n" % tuple([round(s, 2) for s in tra_scores+tes_scores]))
    
