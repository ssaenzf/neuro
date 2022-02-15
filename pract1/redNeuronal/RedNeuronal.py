import numpy as np

class RedNeuronal:
    def __init__(self, cons_aprend=1):
        self.capas = np.empty(0, dtype=object)
        self.cons_aprend = cons_aprend
    def liberar(self):
        pass

    def inicializar(self):
        for capa in self.capas:
            capa.inicializar()

    def aniadir(self, capa):
        self.capas = np.append(self.capas, capa)
    
    def disparar(self):
        for capa in self.capas:
            capa.disparar()

    def propagar(self):
        for capa in self.capas:
            capa.propagar()
    
            
