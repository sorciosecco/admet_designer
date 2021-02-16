
from core import settings
from core.load import load_datasets
from core.classification import run_modelling_steps
from core.metrics import calculate_class_scores


MultiModel=('RF', 'kNN', 'AB', 'ETC', 'GB', 'LDA', 'MLP', 'SVM')
MModel_scores=[]
def run_procedure(X1, X2, Y1, Y2, O1, O2):
    for mods in MultiModel:
        settings.MODEL= mods
        Y1_pred, Y2_pred, Y1_prob, Y2_prob = run_modelling_steps(X_train=X1, X_test=X2, Y_train=Y1, Y_test=Y2, model_type=settings.MODEL, nondef_params=settings.NPARA, sm=settings.SAVEMODEL, mc=settings.MULTICLASS)
        scores = calculate_class_scores(Y1_exp=Y1, Y1_pred=Y1_pred, Y1_prob=Y1_prob, Y2_exp=Y2, Y2_pred=Y2_pred, Y2_prob=Y2_prob, O1=O1, O2=O2, pc=settings.PROBACUTOFF)
        MModel_scores.append(scores)
        print(mods)

def build_auto(args):
    
    X1, X2, Y1, Y2, O1, O2, V = load_datasets(training=settings.FIT, test=settings.PREDICT, response=settings.RESPONSE)
    #settings.VAR_NAMES=V[1:-1]
    #settings.VARS=V
    
    run_procedure(X1=X1, X2=X2, Y1=Y1, Y2=Y2, O1=O1, O2=O2)
    
