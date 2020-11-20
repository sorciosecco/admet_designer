
import random
import numpy as np

from core import settings

def load_infile(infile, A):
    print("\nLoading infile...")
    X, Y = {}, {}
    i=0
    for line in open(infile, 'r'):
        if i==0:
            header = str.split(line.strip(), ';')
            y=header.index(A)
        else:
            x = str.split(line.strip(), ';')
            X[x[0]], Y[x[0]] = [float(x[d]) for d in range(len(x)) if d != y if d != 0], int(x[y])
        i+=1
    return header, X, Y

def get_class_occurrences(Y):
    classes, occurrences = np.unique(Y, return_counts=True)[0].tolist(), np.unique(Y, return_counts=True)[1].tolist()
    maj_class, min_class = classes[occurrences.index(max(occurrences))], classes[occurrences.index(min(occurrences))]
    class_occurrences={classes[x]: occurrences[x] for x in range(len(classes))}
    print("\nGetting class occurrences...\n\t=> %s" % class_occurrences)
    return maj_class, min_class, class_occurrences

def random_subselection(C, A, B):
    class_choises = {key: list(range(len(C[key]))) for key in list(C.keys())}
    R={}
    for key in list(A.keys()):
        chosen=[]
        for n in range(A[key]):
            x=random.choice(class_choises[key])
            chosen.append(x)
            class_choises[key].remove(x)
        R[key] = chosen
    R2={}
    for key in list(B.keys()):
        chosen=[]
        for n in range(B[key]):
            x=random.choice(class_choises[key])
            chosen.append(x)
            class_choises[key].remove(x)
        R2[key] = chosen
    selected_A, selected_B, not_selected = {key: [C[key][x] for x in R[key]] for key in list(R.keys())}, {key: [C[key][x] for x in R2[key]] for key in list(R2.keys())}, {key: [C[key][x] for x in class_choises[key]] for key in list(class_choises.keys())}
    
    return selected_A, selected_B, not_selected

def save_results(X, Y, selected_A, selected_B, not_selected, header):
    print("\nSaving results...")
    A, B, C = {}, {}, {}
    for key in list(selected_A.keys()):
        for n in selected_A[key]:
            A[n] = ";".join([str(x) for x in X[n]+[Y[n]]]) + "\n"
        for n in selected_B[key]:
            B[n] = ";".join([str(x) for x in X[n]+[Y[n]]]) + "\n"
        for n in not_selected[key]:
            C[n] = ";".join([str(x) for x in X[n]+[Y[n]]]) + "\n"
    ocsv1=open("A_"+settings.FIT, "w")
    ocsv1.write(";".join(header)+'\n')
    for key in sorted(list(A.keys())):
        ocsv1.write(";".join([key] + [A[key]]))
    ocsv1.close()
    ocsv3=open("B_"+settings.FIT, "w")
    ocsv3.write(";".join(header)+'\n')
    for key in sorted(list(B.keys())):
        ocsv3.write(";".join([key] + [B[key]]))
    ocsv3.close()
    ocsv2=open("C_"+settings.FIT, "w")
    ocsv2.write(";".join(header)+'\n')
    for key in sorted(list(C.keys())):
        ocsv2.write(";".join([key] + [C[key]]))
    ocsv2.close()

def balance_sets(args):
    settings.PERCENTAGE=args.percentage
    
    np.random.seed(settings.SEED)
    
    header, X, Y = load_infile(infile=settings.FIT, A=settings.ACTIVITY)
    
    maj_class, min_class, class_occurrences = get_class_occurrences(Y=list(Y.values()))
    
    A = {key: round(class_occurrences[min_class]*settings.PERCENTAGE/100) for key in list(class_occurrences.keys())}
    B = {key: round(class_occurrences[min_class]*(100-settings.PERCENTAGE)/100) for key in list(class_occurrences.keys())}
    print("\nConfiguring subselections...\n\tSet A => %s\n\tSet B => %s\n\tSet C => %s" % (A, B, {key: class_occurrences[key]-B[key]-A[key] for key in list(class_occurrences.keys())}))
    
    classes = {key: [k for k in list(Y.keys()) if Y[k] == key] for key in list(class_occurrences.keys())}
    
    selected_A, selected_B, not_selected = random_subselection(C=classes, A=A, B=B)
    
    save_results(X, Y, selected_A, selected_B, not_selected, header)
    
