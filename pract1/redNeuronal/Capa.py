from Neurona import Neurona
import numpy as np

class Capa:
    def __init__(self):
        self.neuronas = np.empty(0, dtype=object)

    def liberar(self):
        pass

    def inicializar(self):
        for neurona in self.neuronas:
            neurona.Inicializar(0)

    def aniadir(self, neurona):
        self.neuronas = np.append(self.neuronas, neurona)
    
    def conectar(self, capa, peso_min, peso_max):
        for neurona_capa in capa.neuronas:
            self.conectar(neurona_capa, peso_min, peso_max)

    def conectar(self, neurona, peso_min, peso_max):
        for mi_neu in self.neuronas:
            # peso = random.uniform(peso_min, peso_max)
            peso = peso_min # Se cambiara con el codigo del arriba para siguientes practicas
            mi_neu.conectar(neurona, peso)

    def disparar(self):
        for neurona in self.neuronas:
            neurona.disparar()

    def propagar(self):
        for neurona in self.neuronas:
            neurona.propagar()