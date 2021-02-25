
import math, scipy
import numpy as np
from sklearn.metrics import r2_score

from core import settings, variables
from core.load import load_datasets
from core.regression import run_model_training
from core.graphic import plot_pls_exp_vs_pred


#algorithms_list=['PLS', 'RF', 'kNN', 'SVM', 'MLP']
algorithms_list=['PLS']

def calc_dmody():
    t_scores, u_scores = np.array([ t[-1] for t in variables.model.x_scores_ ]), np.array([ u[-1] for u in variables.model.y_scores_ ])
    #t_scores, u_scores = variables.X_tra[variables.model.support_], variables.Y_tra[variables.model.support_]
    slope, intercept, r, p, stderr = scipy.stats.linregress(t_scores, u_scores)
    
    x, y = np.array(variables.Y_tra), np.array(variables.Y_pred)
    r2 = r2_score(x, y)
    line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.3f}'
    #line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r2={r2:.3f}'
    #print(line)
    
    plot_pls_exp_vs_pred(x, y, slope, intercept, line)
    
    b = -1
    ocsv=open("DModY_%s.csv" % settings.MODEL, "w")
    ocsv.write("Object\tY_exp\tY_pred\tDModY\n")
    for o in range(len(variables.O_list)):
        d=abs(slope * variables.Y_tra[o] + b * variables.Y_pred[o] + intercept) / math.sqrt(slope**2 * b**2)
        ocsv.write("%s\t%s\t%s\t%s\n" % (variables.O_list[o], variables.Y_tra[o], variables.Y_pred[o], d))
    ocsv.close()


def run_dmody_regression_operations(args):
    
    variables.X_tra, variables.Y_tra, variables.O_list, variables.V_list = load_datasets(training=settings.FIT, response=settings.RESPONSE)
    print("\nTRAINING SET:\nN objects = %s\nN independent vars = %s\nN dependent vars: 1 (%s)" % (len(variables.O_list), len(variables.X_tra[0]), settings.RESPONSE))
    
    for a in algorithms_list:
        settings.MODEL=a
        print("\nPerforming %s+LOO..." % a)
        model = run_model_training()
        
        calc_dmody()
    
