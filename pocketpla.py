import random
import numpy as np
from linregression import LinearRegression


class PocketPLA():
    def __init__(self):
        self._w = None
        
    def fit(self, X, Y, iter_max=1000):      
        X1 = [np.concatenate((np.ones((1)), x)) for x in X]  # adding x0 = 1
        
        # Initial weights and initial list of points classified incorrectly
        lr = LinearRegression()
        lr.fit(X, Y)
        self._w = lr.w
        pci = self.pci_list(X1, Y)  
        
        best_w = self.w.copy()
        best_error = len(pci)  # stores the error in the training set at each best iteration
        
        _iter = 0
        while pci:
            # Weight correction step
            xi, yi = random.choice(pci)

            self._w = self._w + yi*xi
            
            pci = self.pci_list(X1, Y)

            # Saving best error and vector w
            e_in = len(pci)
            if (best_error > e_in):
                best_error = e_in
                best_w = self.w.copy()
            
            _iter += 1
            if (_iter > iter_max):
                break
            
        self._w = best_w
                    
    def predict(self, x):
        return self.sign(self._w@np.concatenate((np.ones((1)), x)))
    
    def eval(self, x1):  # method that gets the value of x2
        if not self._w[2]:
            return x1 * 0
        
        a = (-1) * self._w[1] / self._w[2]
        b = (-1) * self._w[0] / self._w[2]
        
        return x1 * a + b
    
    def score(self, X, Y):
        success = 0
        
        for x, y in zip(X, Y):
            if self.predict(x) == y:
                success += 1
            
        return success / len(Y)
        
    def pci_list(self, X, Y):
        return [(x, y)
                for x, y in zip(X, Y)
                if self.sign(self.w@x) != y]
        
    @staticmethod    
    def sign(x):
        return 1 if x >= 0 else -1

    @property
    def w(self):
        return self._w
    
    @property
    def bias(self):
        return self.w[0] if self.w else None
