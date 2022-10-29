import numpy as np


class LinearRegression:
    def __init__(self):
        self._w = None
        
    def fit(self, _X, _y):
        # Data pre-processing
        X1 = np.array([np.concatenate((np.ones((1)), x))
                       for x in _X])  # adding x0 = 1
        
        self._w = np.linalg.inv(X1.T@X1)@X1.T@_y
        
    def predict(self, _x):
        return self.w@np.concatenate((np.ones((1)), _x))

    def classifier(self, _x):
        return self.sign(self.predict(_x))

    def eval(self, x1):  # method that gets the value of x2
        if not self._w[2]:
            return x1 * 0
        
        a = (-1) * self._w[1] / self._w[2]
        b = (-1) * self._w[0] / self._w[2]
        
        return x1 * a + b
    
    def score(self, X, Y):
        success = 0
        
        for x, y in zip(X, Y):
            if self.classifier(x) == y:
                success += 1
            
        return success / len(Y)
     
    @staticmethod    
    def sign(x):
        return 1 if x >= 0 else -1

    @property
    def w(self):
        return self._w