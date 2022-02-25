import numpy as np
from redNeuronal.Tipo import Tipo
from scipy.stats import uniform

class Capa:
    def __init__(self):
        self.neuronas = np.empty(0, dtype=object)

    def liberar(self):
        pass

    def inicializar(self):
        for neurona in self.neuronas:
            neurona.inicializar(0)

    def aniadir(self, neurona):
        self.neuronas = np.append(self.neuronas, neurona)
    
    def conectar(self, capa, peso_min, peso_max):
        for neurona_capa in capa.neuronas:
            self.conectar_neurona(neurona_capa, peso_min, peso_max)

    def conectar_neurona(self, neurona, peso_min, peso_max):
        for mi_neu in self.neuronas:
            if neurona.tipo == Tipo.MCCULLOCH or neurona.tipo == Tipo.SESGOIGUAL:
                peso = 0
            else:
                peso = uniform(peso_min, peso_max - peso_min).rvs()
            mi_neu.conectar(neurona, peso)

    def disparar(self):
        for neurona in self.neuronas:
            neurona.disparar()

    def propagar(self):
        for neurona in self.neuronas:
            neurona.propagar()
    
    def __str__(self):
        text = ''
        for neurona in self.neuronas:
            text += neurona.__str__() + '\n'
        return text