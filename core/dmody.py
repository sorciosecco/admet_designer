
import math, scipy
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from core import settings, variables
from core.load import load_datasets
from core.regression import run_model_training
from core.graphic import plot_pls_exp_vs_pred


algorithms_list=['PLS', 'kNN', 'SVM', 'RF', 'MLP']

def calc_dmody():
    if settings.MODEL=="PLS":
        t_scores, u_scores = np.array([ t[-1] for t in variables.model.x_scores_ ]), np.array([ u[-1] for u in variables.model.y_scores_ ])
        slope, intercept, r, p, stderr = scipy.stats.linregress(t_scores, u_scores)
        x, y = np.array(variables.Y_tra), np.array(variables.Y_pred)
        r2, b = r2_score(x, y), -1
        line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.3f}'
        #print(line)
        plot_pls_exp_vs_pred(x, y, slope, intercept, line)
    
    #b = -1
    D_list=[]
    for o in range(len(variables.O_list)):
        if settings.MODEL=="PLS": d=abs(slope * variables.Y_tra[o] + b * variables.Y_pred[o] + intercept) / math.sqrt(slope**2 * b**2)
        else: d=abs(variables.Y_tra[o] - variables.Y_pred[o])
        D_list.append(d)
    return D_list


def run_dmody_regression_operations(args):
    settings.NPARA, variables.DMODY = False, True
    
    variables.X_tra, variables.Y_tra, variables.O_list, variables.V_list = load_datasets(training=settings.FIT, response=settings.RESPONSE)
    print("\nTRAINING SET:\nN objects = %s\nN independent vars = %s\nN dependent vars: 1 (%s)" % (len(variables.O_list), len(variables.X_tra[0]), settings.RESPONSE))
    
    df_results = pd.DataFrame({'Y_exp':variables.Y_tra}, index=variables.O_list)
    
    for a in algorithms_list:
        settings.MODEL=a
        print("\nPerforming %s+LOO..." % a)
        run_model_training()
        
        distances = calc_dmody()
        df_results['Y_pred (%s)' % settings.MODEL] = variables.Y_pred
        df_results['DModY (%s)' % settings.MODEL] = distances
        
    #print(df_results)
    df_results.to_csv("DModY.csv", sep=";")
    
