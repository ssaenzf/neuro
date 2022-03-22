from redNeuronal.Conexion import Conexion
from redNeuronal.Tipo import Tipo
import numpy as np
from scipy.special import expit

# def sigmoid(x):
#     y = np.exp(-x)
#     return 1/(1+y)

def sigmoid(x):
    if x > 20:
        return 1
    elif x < -20:
        return -1
    y = np.exp(-x)
    return (2/(1+y)) - 1

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

    def addRed(self, red):
        self.red = red

    def conectar(self, neurona, peso):
        conexion = Conexion(peso, neurona, name=self.name + "-" + neurona.name)
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
        elif self.tipo == Tipo.SIGMOIDE:
            entrada = round(self.valor_entrada, 5)
            if entrada in self.red.calculo:
                self.valor_salida = self.red.calculo[entrada]
            else:
                self.valor_salida = sigmoid(entrada)
                self.red.calculo[entrada] = self.valor_salida
        else:
            self.valor_salida = 0
        
        for conexion in self.conexiones:
            conexion.valor = self.valor_salida
    
    def propagar(self):
        for conexion in self.conexiones:
            conexion.neurona.valor_entrada += conexion.valor * conexion.peso 

    def __str__(self, tab=''):
        text = ''
        text += str(tab) + "Neurona " + self.name + ": " + str(self.valor_entrada) + " " + str(self.valor_salida) + "\n"
        tab += '  '
        for conexion in self.conexiones:
            text += str(tab) + str(conexion) + '\n'
        
        return text