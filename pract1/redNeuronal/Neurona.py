from Conexion import Conexion
from Tipo import Tipo

class Neurona:
    def __init__(self, umbral, tipo):
        self.umbral = umbral
        self.tipo = tipo
        self.conexiones = []

    def Liberar(self):
        pass

    def Inicializar(self, x):    
        self.valor_entrada = x

    def Conectar(self, neurona, peso):
        conexion = Conexion(peso, neurona)
        self.conexiones.append(conexion)

    def Disparar(self):
        if self.tipo == Tipo.DIRECTA:
            self.valor_salida = self.valor_entrada
        elif self.tipo == Tipo.MCCULLOCH:
            if self.valor_entrada >= self.umbral:
                self.valor_salida = 1
            else:
                self.valor_salida = 0
        else:
            pass
    
    def Propagar(self):
        for conexion in self.conexiones:
            conexion.neurona.valor_entrada += conexion.valor * conexion.peso 
