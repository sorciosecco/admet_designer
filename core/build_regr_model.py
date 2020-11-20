from core import settings
from core.load import load_datasets
from core.perform_regression import get_pls_predictions
from core.metrics import calc_regr_metrics, run_cross_valid
from core.save import save_vars_importance

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

def perform_pls(tra_X, tra_Y, tes_X=None, tes_Y=None):
    if tes_X != None:
        tra_Yp, tes_Yp, Scores = get_pls_predictions(tra_X=tra_X, tes_X=tes_X, tra_Y=tra_Y, latent_variables=settings.LATENT)
        print("\nR2 (TRA) = %s\nR2 (TES) = %s\n" % (round(Scores['R2'],3), round(r2_score(tes_Y, tes_Y_pred),3)))
        return tra_Yp, tes_Yp
    else:
        print("\nLV\tR2\tQ2\tSDEC\tSDEP")
        stop, best_q2, best_lv = False, 0, 0
        for lv in range(1, settings.LATENT+1):
            tra_Yp, Scores = get_pls_predictions(tra_X=tra_X, tra_Y=tra_Y, latent_variables=lv)
            if Scores['Q2'] < best_q2:
                stop=True
            else:
                best_q2, best_lv = Scores['Q2'], lv
                print('\t'.join([str(lv)] + [ str(round(Scores[k],3)) for k in list(Scores.keys()) ]))
            if stop:
                break
        return tra_Yp


def perform_machine_learning(tra_X, tra_Y, tes_X=None, tes_Y=None):
    def algorithm_setup(model_type):#, nondef_params):
        if model_type=="RF": model = RandomForestRegressor(random_state=settings.SEED, n_jobs=-1, verbose=settings.VERBOSE
                                                           , n_estimators=100
                                                           , criterion='mse'
                                                           , max_depth=None
                                                           , max_features='auto'
                                                           , max_leaf_nodes=None
                                                           , min_samples_split=2
                                                           , min_samples_leaf=1
                                                           , min_weight_fraction_leaf=0.0
                                                           , min_impurity_decrease=0.0
                                                           , min_impurity_split=None
                                                           , bootstrap=True
                                                           , oob_score=False
                                                           , warm_start=False
                                                           )
        else: print("\nERROR: attaccati al cazzo\n")
        return model
    
    model=algorithm_setup(model_type=settings.MODEL)
    
    model.fit(tra_X, tra_Y)
    
    if settings.SAVEVARS: save_vars_importance(V=settings.VARS, I=model.feature_importances_)
    
    if settings.PREDICT!=None:
        tra_Yp, tes_Yp = model.predict(tra_X).tolist(), model.predict(tes_X).tolist()
        return tra_Yp, tes_Yp
    else:
        scores = run_cross_valid(X=tra_X, Y=tra_Y, M=model)
        return scores


def build_regression_model(args):
    #settings.NPARA=args.npara
    #settings.PROBACUTOFF=args.probacutoff
    #settings.SAVEPRED=args.savepred
    settings.LATENT=args.latent
    settings.CROSSVAL=args.crossval
    
    if settings.PREDICT==None:
        X, Y, O, V = load_datasets(training=settings.FIT, test=settings.PREDICT, response=settings.RESPONSE)
        settings.VARS=V
        print("\nTRAINING SET:\nN objects = %s\nN independent vars = %s\nN dependent vars: 1 (%s)\n" % (len(O), len(X[0]), settings.RESPONSE))
        
        if settings.MODEL=="PLS":
            tra_Yp = perform_pls(tra_X=X, tra_Y=Y)
        else:
            scores = perform_machine_learning(tra_X=X, tra_Y=Y)
            message = "CV%s\t"*settings.CROSSVAL + "AVER\n" + "%s\t"*settings.CROSSVAL + "%s\n"
            print(message % (tuple(list(range(1, settings.CROSSVAL+1)) + [round(s, 2) for s in scores] + [round(sum(scores)/len(scores), 2)])))
        
    else:
        X1, X2, Y1, Y2, O1, O2, V = load_datasets(training=settings.FIT, test=settings.PREDICT, response=settings.RESPONSE)
        print("\nTRAINING SET:\nN objects = %s\nN independent vars = %s\nN dependent vars: 1 (%s)\n\nTEST SET:\nN objects = %s\n" % (len(O1), len(X1[0]), settings.RESPONSE, len(O2)))
        settings.VARS=V
        
        if settings.MODEL=="PLS": tra_Yp, tes_Yp = perform_pls(tra_X=X1, tes_X=X2, tra_Y=Y1, tes_Y=Y2)
        else: tra_Yp, tes_Yp = perform_machine_learning(tra_X=X1, tes_X=X2, tra_Y=Y1, tes_Y=Y2)
        
        tra_scores, tes_scores = calc_regr_metrics(Ye=Y1, Yp=tra_Yp), calc_regr_metrics(Ye=Y2, Yp=tes_Yp)
        
        print("SET\tCORR\tERROR\nTRA\t%s\t%s\nTES\t%s\t%s\n" % tuple([round(s, 2) for s in tra_scores+tes_scores]))
    
    
