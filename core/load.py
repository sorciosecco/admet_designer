
import numpy as np
from sklearn.preprocessing import scale
from core import settings

def load_datasets(training, response, test=None):
    
    def print_data(X1, X2, Y1, Y2, O1, O2, response):
        classes, occurrences = np.unique(Y1, return_counts=True)[0].tolist(), np.unique(Y1, return_counts=True)[1].tolist()
        output = '\nTRAINING SET:\nN objects = %s\nN independent vars = %s\nN dependent vars: 1\n' % (len(O1), len(X1[0])) + 'Y (%s) = ' % (response) + ' + '.join(['%s (class %s)' % (occurrences[x], classes[x]) for x in range(len(classes))])
        classes, occurrences = np.unique(Y2, return_counts=True)[0].tolist(), np.unique(Y2, return_counts=True)[1].tolist()
        output += '\n\nTEST SET:\nN objects = %s\n' % (len(O2)) + 'Y (%s) = ' % (response) + ' + '.join(['%s (class %s)' % (occurrences[x], classes[x]) for x in range(len(classes))])
        print(output)
    
    def read_file(filename, y):
        l, X, Y, O = 0, [], [], []
        for line in open(filename, 'r'):
            if l==0:
                V, line = str.split(line.strip(), ';')[1:-1], str.split(line.strip(), ';')[1:]
                Yind = line.index(y)
            else:
                O.append(line.split(';')[0])
                line = str.split(line.strip(), ';')[1:]
                X.append([float(line[x]) for x in range(len(line)) if x != Yind])
                try:
                    int(line[Yind])
                except:
                    Y.append(float(line[Yind]))
                else:
                    Y.append(int(line[Yind]))
            l+=1
        return X, Y, O, V
    
    X1, Y1, O1, V = read_file(filename=training, y=response)
    X1 = scale(X1).tolist()
    if settings.PREDICT!=None:
        X2, Y2, O2, V = read_file(filename=test, y=response)
        X2 = scale(X2).tolist()
        #print_data(X1, X2, Y1, Y2, O1, O2, response)
        return X1, X2, Y1, Y2, O1, O2, V
    else:
        return X1, Y1, O1, V
        
