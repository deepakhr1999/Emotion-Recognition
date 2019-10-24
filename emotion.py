import utils
import numpy as np
import pickle

class EmotionDetector(object):
    def __init__(self, X, y, classes=None):
        self.prescaler, self.pca, self.postscaler, X = utils.getPreprocessor(X)
        self.model = utils.evaluate(X, y)
        self.classes = classes

    def preprocess(self, X):
        X = self.prescaler.transform(X)
        X = self.pca.transform(X)
        return self.postscaler.transform(X)
    
    def predict(self, X, preprocess=True):
        if preprocess:
            X = self.preprocess(X)
        return self.model.predict(X)

    def predictClass(self, X):
        i = self.predict(X)[0]
        if self.classes == None:
            return i
        else:
            return self.classes[i]

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
            print(f'Saved model in {filename}')

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

if __name__ == '__main__':
    X = np.load('peak.npy')
    print(X.shape)
    y = np.load('targets.npy')
    classes = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sad', 'surprise']
    model = EmotionDetector(X,y, classes)

    i = 230
    print('Original is', classes[y[i]])
    print(model.predictClass(X[[i]]))
    model.save('../models/emotion.pkl')