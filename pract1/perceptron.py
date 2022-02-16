from sympy import arg
from redNeuronal.RedNeuronal import RedNeuronal
from redNeuronal.Capa import Capa
from redNeuronal.Neurona import Neurona
from redNeuronal.Tipo import Tipo
import argparse
from sklearn.model_selection import train_test_split
import numpy as np
from leerFichero import LeerFichero

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Perceptron')
    parser.add_argument('--modo1',
                        nargs=2,
                        metavar=('fichero', 'porcion'),
                        help='Nombre del fichero de entrada y porcion')
    parser.add_argument('--modo2',
                        nargs=1,
                        metavar='fichero',
                        help='Nombre del fichero de entrada')
    parser.add_argument('--modo3',
                        nargs=2,
                        metavar=('train', 'test'),
                        help='Nombre del fichero de entrada')

    args = parser.parse_args()

    if args.modo1:
        X_train, X_test, y_train, y_test = LeerFichero.mode1(args.modo1[0], args.modo1[1])

    elif args.modo2:
        X_train, X_test = LeerFichero.mode2(args.modo2[0])

    elif args.modo3:
        X_train, X_test, y_train, y_test = LeerFichero.mode3(args.modo3[0], args.modo3[1])

class Perceptron():

    def __init__(self, X_train, y_train, epocas=100, umbral=0.0, alpha=1.0):
        # Construccion de la red del perceptron
        self.epocas = epocas
        self.perceptron = RedNeuronal()
        self.alpha = alpha
        self.X_train = X_train
        self.y_train = y_train
        n_atributos = len(X_train[0])
        n_clases = len(y_train[0])

        # Neuronas capa entrada
        neruonas_entrada = []
        for i in range(0, n_atributos):
            # Las neuronas de entrada, siempre directas ya que se limitan a retransmitir su entrada, por ello no tienen umbral
            neruonas_entrada.append(Neurona(umbral = 0.0, tipo=Tipo.DIRECTA))
        # Neurona correspondiente al bias
        neruonas_entrada.append(Neurona(umbral = 0.0, tipo=Tipo.SESGO))

        # Neuronas capa salida
        neruonas_salida = [] 
        for i in range(n_clases):
            neruonas_salida.append(Neurona(umbral = umbral, tipo=Tipo.PERCEPTRON))  # Se aniade el umbral especificado
        
        # Conexiones. Todas las neuronas de la capa de entrada se conectan con todas las neruonas de la capa de salida
        for i in range(n_atributos + 1): # + 1 debido a que hay que conectar el bias a todas las neuronas de la capa de salida tambien
            for j in range(n_clases):
                neruonas_entrada[i].conectar(neruonas_salida[j], 0) # Pesos de las conexiones inicialmente a 0

        # Creacion capas y anidamiento de neuronas en estas, y anidamiento capas dentro de la red neuronal perceptron
        capa_entrada = Capa()
        capa_salida = Capa()
        for i in range(n_atributos + 1): # + 1 debido a que en la capa de entrada se aniade tambien el bias
            capa_entrada.aniadir(neruonas_entrada[i])
        for i in range(n_clases): 
            capa_salida.aniadir(neruonas_salida[i])
        self.perceptron.aniadir(capa_entrada)
        self.perceptron.aniadir(capa_salida)

    def train(self):

        for epoca in range(self.epocas + 1):
            error_cuad_med = 0
            for record_x, record_y in  self.X_train, self.y_train:

                # Se inicializan a 0 las neuronas de la capa de salida
                for i in range(self.perceptron.capas[1].nueronas):
                    self.perceptron.capas[1].nueronas[i].inicializar(0.0)
                
                # Se inicializan con los valores de entrada las neuronas de la capa de entrada, a excepcion del bias 
                for i in range(self.perceptron.capas[0].neuronas - 1):
                    self.perceptron.capas[0].neuronas[i].inicializar(float(record_x[i]))
                    self.perceptron.capas[0].neuronas[i].disparar()
                    self.perceptron.capas[0].neuronas[i].propagar()
                self.perceptron.capas[0].neuronas[i+1].inicializar(0.0) # Neurona del bias, siempre inicializada a 1
                self.perceptron.capas[0].neuronas[i+1].disparar()
                self.perceptron.capas[0].neuronas[i+1].propagar()

                # Obtención de las salidas, de la capa de salida
                for i in range(self.perceptron.capas[1].nueronas):
                    self.perceptron.capas[1].nueronas[i].disparar()
                
                # Obtencion flag sobre si ha habido error o no en la prediccion para saber realizar el posterior ajuste
                error = False   # flag para saber si hay diferencia del valor predecido respecto al real
                for i in range(self.perceptron.capas[1].nueronas):
                    if self.perceptron.capas[1].nueronas[i].valor_salida != float(record_y[i])

