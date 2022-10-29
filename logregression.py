import random
import numpy as np


class LogisticRegression:
    def __init__(self, eta=0.1):
        self.eta = eta
    
        self._w = None

    # Infere o vetor w da função hipótese
    # Executa a minimização do erro de entropia cruzada pelo algoritmo gradiente de descida
    def fit(self, _X, _y, batch_size, tmax=1000):
        X = np.array([np.concatenate((np.ones((1)), x))
                      for x in _X])  # adding x0 = 1

        N = X.shape[0]
        d = X.shape[1]
        
        if (not batch_size):
            batch_size = N

        self._w = np.zeros(d)
        for t in range(tmax):
            #Escolhendo o lote de entradas
            if batch_size < N:
                indices = random.sample(range(N), batch_size)
                batchX = [X[index] for index in indices]
                batchY = [_y[index] for index in indices]
            else:
                batchX = X
                batchY = _y
                
            somatorio = 0
            for yn, xn in zip(batchY, batchX):         
                somatorio +=  (yn*xn) / (1 + np.exp((yn*self._w).T@xn))
            
            gt = somatorio / N
            
            if np.linalg.norm(gt) < 0.00001:
                break
            
            self._w = self._w + self.eta*gt
        
    # Função hipótese inferida pela regressão logística  
    def predict_prob(self, x):
        return 1 / (1 + np.exp((-1)*(self.w@np.concatenate((np.ones((1)), x)))))
                
    # Predição por classificação linear
    def predict(self, x):
        return 1 if self.predict_prob(x) >= 0.5 else -1

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
    
    @property
    def w(self):
        return self._w
