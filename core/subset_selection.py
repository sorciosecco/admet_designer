from core import settings

import random
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

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
            X[x[0]] = [float(x[d]) for d in range(len(x)) if d != y if d != 0]
            try:
                int(x[y])
            except:
                Y[x[0]] = float(x[y])
            else:
                Y[x[0]] = int(x[y])
        i+=1
    return header, X, Y

def get_class_occurrences(Y):
    print("\nGetting class occurrences...")
    classes, occurrences = np.unique(Y, return_counts=True)[0].tolist(), np.unique(Y, return_counts=True)[1].tolist()
    maj_class, min_class = classes[occurrences.index(max(occurrences))], classes[occurrences.index(min(occurrences))]
    class_occurrences={classes[x]: occurrences[x] for x in range(len(classes))}
    print("\t=> %s" % class_occurrences)
    return maj_class, min_class, class_occurrences

def get_class_subselections(class_occurrences, min_class, N, P, B):
    print("\nConfiguring subselections...")
    if N != None:
        subselections = {key: N for key in list(class_occurrences.keys())}
    elif P != None:
        if B:
            subselections = {key: round(class_occurrences[min_class]*P/100) for key in list(class_occurrences.keys())}
        else:
            subselections = {key: round(class_occurrences[key]*P/100) for key in list(class_occurrences.keys())}
    else:
        subselections = None
    print("\t=> %s" % subselections)
    return subselections

def save_results0(selected, not_selected, infile, header):
    print("\nSaving results...")
    ocsv1=open("sel_"+infile, "w")
    ocsv1.write(";".join(header)+'\n')
    for line in selected:
        ocsv1.write(";".join(line)+'\n')
    ocsv1.close()
    ocsv2=open("notsel_"+infile, "w")
    ocsv2.write(";".join(header)+'\n')
    for line in not_selected:
        ocsv2.write(";".join(line)+'\n')
    ocsv2.close()

def save_results(X, Y, selected, not_selected, infile, header):
    #def plot_pca_scores(selected, not_selected):
    print("\nSaving results...")
    S, NS = {}, {}
    for key in list(selected.keys()):
        for n in selected[key]:
            S[n] = ";".join([str(x) for x in X[n]+[Y[n]]]) + "\n"
        for n in not_selected[key]:
            NS[n] = ";".join([str(x) for x in X[n]+[Y[n]]]) + "\n"
    ocsv1=open("sel_"+infile, "w")
    ocsv1.write(";".join(header)+'\n')
    for key in sorted(list(S.keys())):
        ocsv1.write(";".join([key] + [S[key]]))
    ocsv1.close()
    ocsv2=open("notsel_"+infile, "w")
    ocsv2.write(";".join(header)+'\n')
    for key in sorted(list(NS.keys())):
        ocsv2.write(";".join([key] + [NS[key]]))
    ocsv2.close()
    
    #plot_pca_scores(selected, not_selected)

def random_subselection0(X, Y, S):
    chosen=[]
    K=sorted(list(Y.keys()))
    choices = list(range(len(K)))
    for c in range(S):
        x = random.choice(choices)
        chosen.append(x)
        choices.remove(x)
    selected, not_selected = [[K[c]] + [str(x) for x in X[K[c]]+[Y[K[c]]]] for c in chosen], [[K[c]] + [str(x) for x in X[K[c]]+[Y[K[c]]]] for c in choices]
    
    return selected, not_selected

def random_subselection(C, S):
    class_choises = {key: list(range(len(C[key]))) for key in list(C.keys())}
    R={}
    for key in list(S.keys()):
        chosen=[]
        for n in range(S[key]):
            x=random.choice(class_choises[key])
            chosen.append(x)
            class_choises[key].remove(x)
        R[key] = chosen
    selected, not_selected = {key: [C[key][x] for x in R[key]] for key in list(R.keys())}, {key: [C[key][x] for x in class_choises[key]] for key in list(class_choises.keys())}
    
    #print(selected)
    return selected, not_selected

def compute_scores1(X):
    #print(X[0], len(X[0]))
    pca = PCA(n_components=3)
    #pca.fit(X)
    scores = pca.fit_transform(X)
    #print(len(billo[0]))
    #print(pca.explained_variance_)
    #print(sum(pca.explained_variance_))
    loadings = pca.components_
    #print(len(pca.components_[0]))
    #plt.scatter(pca.components_[0], pca.components_[1])
    #plt.scatter([o[0] for o in scores], [o[1] for o in scores])
    #plt.show()
    return scores

def compute_clusters(S):
    def loadings_plot_clusters(S, C):
        x, y = np.array([s[0] for s in S]), np.array([s[1] for s in S])
        cdict={c: 'C'+str(i) for i,c in enumerate(np.unique(C))}
        gig, ax = plt.subplots()
        for cluster in np.unique(C):
            ix = np.where(C == cluster)
            ax.scatter(x[ix], y[ix], c=cdict[cluster], label=cluster)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.axis('equal')
        ax.legend(title='clusters')
        plt.show()
    
    
    model=DBSCAN(eps=3, min_samples=10, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=-1)
    C=model.fit_predict(S).tolist()
    print(sorted(C))
    print(np.unique(C, return_counts=True))
    #loadings_plot_clusters(S, C)
    
    s=1
    for cluster in np.unique(C):
        ix = np.where(C == cluster)
        #print(ix[0].tolist())
        if cluster == -1:
            subsel = ix[0].tolist()
        else:
            subsel += np.random.choice(ix[0], size=s, replace=False).tolist()
    print(subsel, len(subsel))
    return subsel

def select_subset(args):
    infile=settings.FIT
    P=settings.PERCENTAGE = args.percentage
    B=settings.BALANCE=args.balance
    A=settings.RESPONSE
    N=settings.NUMBER=args.number
    M=settings.METHOD=args.method
    STR=settings.STRATEGY=args.strategy
    
    np.random.seed(settings.SEED)
    
    header, X, Y = load_infile(infile, A)
    
    if STR:
        maj_class, min_class, class_occurrences = get_class_occurrences(Y=list(Y.values()))
        subselections = get_class_subselections(class_occurrences, min_class, N, P, B)
        classes = {key: [k for k in list(Y.keys()) if Y[k] == key] for key in list(class_occurrences.keys())}
    else:
        if N != None:
            subselections = N
        elif P != None:
            subselections = round(len(X)*P/100)
        else:
            subselections = None
    
    if subselections==None:
        print("\nWARNING: specify the number or the percantage of the subset!")
    else:
        print('\nSubselecting...')
        if M=="R":
            if STR:
                selected, not_selected = random_subselection(C=classes, S=subselections)
            else:
                selected, not_selected = random_subselection0(X, Y, S=subselections)
        elif M=="D":
            if STR:
                pass
            else:
                objects, X = [o for o in sorted(list(X.keys()))], [X[key] for key in sorted(list(X.keys()))]
                scores=compute_scores1(X=scale(X, axis=0, with_mean=True, with_std=True, copy=True).tolist())
                clusters = compute_clusters(S=scores)
        elif M=="L":
            if STR:
                pass
            else:
                pass
        else:
            print("\nWARNING: specify the method for subselection!")
    
    if subselections!=None and M!=None:
        if STR:
            save_results(X, Y, selected, not_selected, infile, header)
        else:
            save_results0(selected, not_selected, infile, header)
    
