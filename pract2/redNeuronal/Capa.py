import numpy as np
from redNeuronal.Tipo import Tipo
from scipy.stats import uniform

class Capa:
    def __init__(self, name=''):
        self.neuronas = np.empty(0, dtype=object)
        self.name = name

    def liberar(self):
        pass

    def inicializar(self):
        for neurona in self.neuronas:
            neurona.inicializar(0)
    
    def addRed(self, red):
        self.red = red
        for neurona in self.neuronas:
            neurona.addRed(red)

    def aniadir(self, neurona):
        self.neuronas = np.append(self.neuronas, neurona)
    
    def conectar(self, capa, peso_min, peso_max, bias=False):
        if bias:
            start = 1
        else:
            start = 0
        for neurona_capa in capa.neuronas[start:]:
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
        text = f'{self.name}: \n'
        for neurona in self.neuronas:
            text += neurona.__str__(tab='  ') + '\n'
        return text