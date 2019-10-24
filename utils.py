import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression

def evaluate(X, y):
    # model = LogisticRegression(C=11, max_iter=10e6, solver='liblinear', multi_class='ovr')
    model = SVC(kernel='poly', degree=1, gamma='scale')
    shuffler = StratifiedShuffleSplit(test_size=0.15)
    acc = np.empty(10)
    M = []
    for i, (train_index, test_index) in enumerate(shuffler.split(X, y)):
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]
            model.fit(X_train, y_train)
            acc[i] = model.score(X_test, y_test)
            M.append(model)
    i = acc.argmax()
    print('Optimal accuracy is', acc[i])
    return M[i]

def getPreprocessor(X, ndims=30):
    #scale data
    scaler1 = StandardScaler()
    X = scaler1.fit_transform(X)
    
    # pca
    pca = PCA(n_components=ndims)
    X = pca.fit_transform(X)

    # scale again
    scaler2 = StandardScaler()
    X = scaler2.fit_transform(X)

    return scaler1, pca, scaler2, X

def getLandmarks(image, detector, predictor):
    try:
        face = detector(image)[0]
        landmarks = predictor(image, face)
        
        x = [landmarks.part(i).x for i in range(68)]
        y = [landmarks.part(i).y for i in range(68)]

        dx = [x[i]-x[j] for i in range(68) for j in range(i+1, 68)]
        dy = [y[i]-y[j] for i in range(68) for j in range(i+1, 68)]

        f = [(dx[i]**2+dy[i]**2)**0.5 for i in range(len(dx))] 

        return True, np.array(f).reshape((1,-1)), zip(x,y)
    except:
        return False, None, None

