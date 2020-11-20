
import operator
from core import settings
from sklearn.metrics import balanced_accuracy_score

def backward_feature_elimination(m, Xtr, Xte, Ytr, Yte):
    #print(settings.VAR_NAMES[:0+2], settings.VAR_NAMES[0+2:])
    m.fit(Xtr, Ytr)
    BA = balanced_accuracy_score(Yte, m.predict(Xte).tolist())
    #print(BA)
    #k=0
    #D=[]
    Xtr_i, Xte_i = Xtr, Xte
    V={}
    for i in range(len(settings.VAR_NAMES)):
        D=[]
        #Xtr_i = [[x[v] for v in range(len(x)) if v!=i] for x in Xtr]
        #Xte_i = [[x[v] for v in range(len(x)) if v!=i] for x in Xte]
        #m.fit(Xtr_i, Ytr)
        #BA_i = balanced_accuracy_score(Yte, m.predict(Xte_i).tolist())
        #print(i, settings.VAR_NAMES[i], BA_i)
        #V[settings.VAR_NAMES[i]] = BA_i
        
    #print(settings.VAR_NAMES[D.index(min(D))])
    #V[settings.VAR_NAMES[D.index(min(D))]] = min(D)
    #sorted_V = sorted(V.items(), key=operator.itemgetter(1), reverse=True)
    
    
        for j in range(len(settings.VAR_NAMES)-i):
            #k+=1
            Xtr_ij, Xte_ij = [[x[v] for v in range(len(x)) if v!=j] for x in Xtr_i], [[x[v] for v in range(len(x)) if v!=j] for x in Xte_i]
            m.fit(Xtr_ij, Ytr)
            BA_ij = balanced_accuracy_score(Yte, m.predict(Xte_ij).tolist())
            print(i, j, settings.VAR_NAMES[j], BA_ij)
            D.append(BA_ij)
        print("\nless usefull descriptor: %s\n" % settings.VAR_NAMES[D.index(min(D))])
        V[settings.VAR_NAMES[D.index(min(D))]] = min(D)
        Xtr_i, Xte_i = [[x[v] for v in range(len(x)) if v!=D.index(min(D))] for x in Xtr_i], [[x[v] for v in range(len(x)) if v!=D.index(min(D))] for x in Xte_i]
    
    #print([settings.VAR_NAMES[v] for v in range(len(list(V.keys()))) if V[list(V.keys())[v]] > BA])
    X2tr, X2te = [[x[v] for v in range(len(x)) if V[list(V.keys())[v]] > BA] for x in Xtr], [[x[v] for v in range(len(x)) if V[list(V.keys())[v]] > BA] for x in Xte]
    m.fit(X2tr, Ytr)
    BA2 = balanced_accuracy_score(Yte, m.predict(X2te).tolist())
    print("\n\nBA 1 = %s\nBA 2 = %s" % (BA, BA2))
