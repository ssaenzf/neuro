from redNeuronal.Conexion import Conexion
from redNeuronal.Tipo import Tipo
import numpy as np

class Neurona:
    def __init__(self, umbral=2.0, tipo=Tipo.DIRECTA, name=''):
        self.umbral = umbral
        self.tipo = tipo
        self.conexiones = np.empty(0, dtype=object)
        self.valor_entrada = 0
        self.valor_salida = 0
        self.name = name

    def liberar(self):
        pass

    def inicializar(self, x):    
        self.valor_entrada = x

    def conectar(self, neurona, peso):
        conexion = Conexion(peso, neurona)
        self.conexiones = np.append(self.conexiones, conexion)

    def disparar(self):
        if self.tipo == Tipo.DIRECTA:
            self.valor_salida = self.valor_entrada
        elif self.tipo == Tipo.SESGOIGUAL or self.tipo == Tipo.SESGO:
            self.valor_salida = 1
        elif self.tipo == Tipo.MCCULLOCH:
            self.valor_salida = 1 if self.valor_entrada >= self.umbral else 0
        elif self.tipo == Tipo.PERCEPTRON :
            if self.valor_entrada >= self.umbral:
                self.valor_salida = 1
            elif self.valor_entrada < -self.umbral:
                self.valor_salida = -1
            else:
                self.valor_salida = 0
        elif self.tipo == Tipo.ADALINE:
            if self.valor_entrada >= self.umbral:
                self.valor_salida = 1
            else:
                self.valor_salida = -1
        else:
            self.valor_salida = 0
        
        for conexion in self.conexiones:
            conexion.valor = self.valor_salida
    
    def propagar(self):
        for conexion in self.conexiones:
            conexion.neurona.valor_entrada += conexion.valor * conexion.peso 

    def __str__(self):
        return "Neurona: " + self.name + " " + str(self.valor_entrada) + " " + str(self.valor_salida)