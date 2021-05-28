
import os, pickle
from core import settings, variables
from core.load import load_external_set

def run_prediction(args):
    settings.INFILE=args.infile
    
    variables.X_pred, variables.O_list, variables.V_list = load_external_set(filename=settings.INFILE)
    
    M = [[o] for o in variables.O_list]
    
    print('\nEXTERNAL SET\nN objects: %s\nX independent vars: %s\n' % (len(variables.O_list), len(variables.V_list)))
    
    for modelname in sorted(os.listdir(settings.models_dir)):
        print("Predicting with %s..." % modelname.split("_")[0])
        
        mfile = open(os.path.join(settings.models_dir, modelname), 'rb')
        model = pickle.load(mfile)
        mfile.close()
        
        Yp = model.predict(variables.X_pred)
        
        for y in range(len(Yp)):
            M[y].append(Yp[y])
            
    print('\nSaving the predictions...')
    ocsv=open('models_predictions.csv', 'w')
    ocsv.write("\t".join(["Objects"] + ["Y_pred ("+n.split("_")[0]+")" for n in sorted(os.listdir(settings.models_dir))])  + "\n")
    for line in M:
        ocsv.write("\t".join([str(l) for l in line])  + "\n")
    ocsv.close()
