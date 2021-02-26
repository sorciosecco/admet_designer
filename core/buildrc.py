
from core import settings
from core.load import load_datasets
#from core.perform_regression import get_pls_predictions
from core.metrics import calculate_pls_metrics
from core.save import save_pls_predictions

def get_pls_predictions():
    pass

def run_pls(X_train, X_test, Y_train, Y_test, O_train, O_test, nlv, lt, ht):
    if nlv!=None:
        Y_train_pred, Y_test_pred, Scores = get_pls_predictions(x_train=X_train, x_test=X_test, y_train=Y_train, latent_variables=nlv)
        #print('\n'+'\t'.join([ k for k in list(Scores.keys()) ])+'\n'+'\t'.join([ str(round(Scores[k],3)) for k in list(Scores.keys()) ])+'\n')
    else:
        print("\nLV\tR2\tQ2\tSDEC\tSDEP")
        stop, best_q2, best_lv = False, 0, 0
        for lv in range(1, 11):
            Y_train_pred, Y_test_pred, Scores = get_pls_predictions(x_train=X_train, x_test=X_test, y_train=Y_train, latent_variables=lv)
            #if Scores['Q2'] < best_q2:
                #stop=True
            #else:
                #best_q2, best_lv = Scores['Q2'], lv
                #print('\t'.join([str(lv)] + [ str(round(Scores[k],3)) for k in list(Scores.keys()) ]))
            #if stop:
                #break
    
    
    calculate_pls_metrics(y_train_exp=Y_train, y_train_pred=Y_train_pred, y_test_exp=Y_test, y_test_pred=Y_test_pred, lco=lt, hco=ht)
    
    if settings.SAVEPRED:
        if settings.LOWTHRESHOLD!=None and settings.HIGHTHRESHOLD!=None:
            save_pls_predictions(y_train_exp=Y_train, y_test_exp=Y_test, y_train_pred=Y_train_pred, y_test_pred=Y_test_pred, ob_train=O_train, ob_test=O_test, lt=settings.LOWTHRESHOLD, ht=settings.HIGHTHRESHOLD)

def build_class_regression_model(args):
    settings.LATENT=args.latent
    settings.HIGHTHRESHOLD=args.highthreshold
    settings.LOWTHRESHOLD=args.lowthreshold
    settings.SAVEPRED=args.savepred
    settings.VERBOSE=-1
    
    X1, X2, Y1, Y2, O1, O2, V = load_datasets(training=settings.FIT, test=settings.PREDICT, y_name=settings.ACTIVITY)
    
    if settings.MODEL=='PLS':
        run_pls(X_train=X1, X_test=X2, Y_train=Y1, Y_test=Y2, O_train=O1, O_test=O2, nlv=settings.LATENT, lt=settings.LOWTHRESHOLD, ht=settings.HIGHTHRESHOLD)
    
