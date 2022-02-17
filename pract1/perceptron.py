from sys import stdout
from redNeuronal.RedNeuronal import RedNeuronal
from redNeuronal.Capa import Capa
from redNeuronal.Neurona import Neurona
from redNeuronal.Tipo import Tipo
import argparse
from leerFichero import LeerFichero

class Perceptron():

    # Funcion para la construccion de la red del perceptron
    def __init__(self, umbral=0.0, alpha=1.0):
        self.perceptron = RedNeuronal()
        self.umbral = umbral
        self.alpha = alpha

    def make_red(self, n_atributos, n_clases):
         # Neuronas capa entrada
        neuronas_entrada = []
        for i in range(0, n_atributos):
            # Las neuronas de entrada, siempre directas ya que se limitan a retransmitir su entrada, por ello no tienen umbral
            neuronas_entrada.append(Neurona(umbral = 0.0, tipo=Tipo.DIRECTA))
        # Neurona correspondiente al bias
        neuronas_entrada.append(Neurona(umbral = 0.0, tipo=Tipo.SESGO))

        # Neuronas capa salida
        neuronas_salida = [] 
        for i in range(n_clases):
            neuronas_salida.append(Neurona(umbral = self.umbral, tipo=Tipo.PERCEPTRON))  # Se aniade el umbral especificado
        
        # Conexiones. Todas las neuronas de la capa de entrada se conectan con todas las neuronas de la capa de salida
        for i in range(n_atributos + 1): # + 1 debido a que hay que conectar el bias a todas las neuronas de la capa de salida tambien
            for j in range(n_clases):
                neuronas_entrada[i].conectar(neuronas_salida[j], 0) # Pesos de las conexiones inicialmente a 0

        # Creacion capas y anidamiento de neuronas en estas, y anidamiento capas dentro de la red neuronal perceptron
        capa_entrada = Capa()
        capa_salida = Capa()
        for i in range(n_atributos + 1): # + 1 debido a que en la capa de entrada se aniade tambien el bias
            capa_entrada.aniadir(neuronas_entrada[i])
        for i in range(n_clases): 
            capa_salida.aniadir(neuronas_salida[i])
        self.perceptron.aniadir(capa_entrada)
        self.perceptron.aniadir(capa_salida)

    # Funcion para la realizacion del entrenamiento por el perceptron
    def train(self, X_train, y_train):
        # Se crea la red del perceptron
        self.make_red(X_train.shape[1], y_train.shape[1])

        # Paso 0, inicial todos los pesos y sesgo
        self.perceptron.inicializar()
        epoca = 0

        # Paso 1, mientras que haya actualizacion de peso, se ejecutra paso 2-6
        parar = False
        while not parar:
            # Se pone parar a True suponiendo que no va a actualizar durante la epoca
            parar = True

            # Uso para el calculo del error cuadratico medio en cada epoca
            epoca += 1
            error_cuad_med = 0

            # Saca los pesos de la conexion y los guarda en una lista,
            # uso para comparar si los pesos han sido actualizado
            # La lista seria [[w11, w12, w13], [w21, w22, w23], [w31, w32, w33]]
            ultimo_pesos = []
            for j in range(len(self.perceptron.capas[-1].neuronas)):
                temp = []
                for neurona in self.perceptron.capas[0].neuronas:
                    temp.append(neurona.conexiones[j].peso)
                ultimo_pesos.append(temp)

            # Paso 2, para cada par de entrenamiento, ejecutar paso 3-5
            for m in range(len(y_train)):
                record_x = X_train[m]
                record_y = y_train[m]

                # Paso 3, establecer las activaciones a las neuronas de entrada, a excepcion del bias 
                for i in range(len(self.perceptron.capas[0].neuronas) - 1):
                    self.perceptron.capas[0].neuronas[i].inicializar(record_x[i])
                
                # Paso 4, calcular la respuesta de cada neurona de salida y_in
                self.perceptron.capas[0].disparar()
                self.perceptron.capas[0].propagar()

                # Obtención de las salidas y_in, de la capa de salida
                self.perceptron.capas[-1].disparar()
                
                # Obtencion flag sobre si ha habido error o no en la prediccion para saber realizar el posterior ajuste 
                # de pesos a las conexiones. Ademas obtencion del error cuadratico medio
                error = False   # flag para saber si hay diferencia del valor predecido respecto al real
                for neurona, t in zip(self.perceptron.capas[-1].neuronas, record_y):
                    error_cuad_med += (neurona.valor_salida - t)**2
                    if neurona.valor_salida != t:
                        error = True
                
                # Paso 5, ajustar los pesos en el caso de que alguna clase no coincide
                if error:
                    # Paso 5.a Ajuste de los pesos menos bias
                    for i in range(len(self.perceptron.capas[0].neuronas) - 1):
                        neurona_i = self.perceptron.capas[0].neuronas[i]
                        # Conexiones de la neurona en cuestion
                        for j in range(len(self.perceptron.capas[-1].neuronas)):
                            nuevo_peso = neurona_i.conexiones[j].peso_anterior + self.alpha * record_y[j] * record_x[i]
                            neurona_i.conexiones[j].peso = nuevo_peso
                            neurona_i.conexiones[j].peso_anterior = nuevo_peso

                            # Si hay algun cambio en los pesos respecto anterior, se actualiza el flag
                            if ultimo_pesos[j][i] != nuevo_peso:
                                parar = False

                    # Paso 5.b Ajuste de los pesos en bias
                    bias_i = self.perceptron.capas[0].neuronas[i+1]
                    for j in range(len(self.perceptron.capas[-1].neuronas)):
                        nuevo_peso = bias_i.conexiones[j].peso_anterior + self.alpha * record_y[j]
                        bias_i.conexiones[j].peso = nuevo_peso
                        bias_i.conexiones[j].peso_anterior = nuevo_peso

                        # Si hay algun cambio en los pesos respecto anterior, se actualiza el flag
                        if ultimo_pesos[j][i] != nuevo_peso:
                            parar = False
            
            # El error cuadratico medio se calcula haciendo la media del total de iteraciones sobre registro totales, y valores esperados dentro de cada registro
            error_cuad_med = error_cuad_med/(len(record_y)*len(y_train))
            
            # Impresion por pantalla de epoca completada y error cuadratico medio
            print(f"Epoca: {epoca}, MSE: {error_cuad_med}")
            
            # Paso 6, si peso_actualizado = False, se termina el entrenamiento, sino vuelve al bucle while
    
    # Funcion para la prediccion de la red del perceptron
    def predecir(self, X_test, f_out):
        text = ""
        for i in range(len(self.perceptron.capas[0].neuronas) - 1):
            text += "X{}\t".format(i+1)
        for j in range(len(self.perceptron.capas[-1].neuronas)):
            text += "Y{}\t".format(j+1)
        text += "\n"
        f_out.write(text)

        # Vacia todas las entradas de la red
        self.perceptron.inicializar()

        # Se ejecuta uno a uno el calcula y prediccion para cada registro de entrada
        for x in X_test:
            # Las neuronas de entrada se inicializan con el valor de entrada a la red, salvo el bias que tiene valor 1 por defecto
            for i in range(len(self.perceptron.capas[0].neuronas) - 1):
                self.perceptron.capas[0].neuronas[i].inicializar(x[i])
            
            # Calcula la respuesta de cada neurona de salida y_in
            self.perceptron.capas[0].disparar()
            self.perceptron.capas[0].propagar()

            # Obtención de las salidas y_in, de la capa de salida
            self.perceptron.capas[-1].disparar()

            # Se recorren las neuronas de salida recolectando el valor que dan
            text = ''
            for neurona in self.perceptron.capas[-1].neuronas:
                for x_i in x:
                    text += "{}\t".format(x_i)
                text += "{:.2f}\t".format(neurona.valor_salida)
            text += '\n'
            f_out.write(text)

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
    parser.add_argument('--f_out',
                        nargs=1,
                        metavar='fichero',
                        help='Nombre del fichero de entrada')

    args = parser.parse_args()

    if not args.f_out:
        f_out = stdout
    else:
        f_out = open(args.f_out[0], 'w')

    if args.modo1:
        X_train, X_test, y_train, y_test = LeerFichero.mode1(args.modo1[0], args.modo1[1])
        perceptron = Perceptron(umbral=0.2, alpha=0.1)
        perceptron.train(X_train, y_train)
        perceptron.predecir(X_test, f_out)

    elif args.modo2:
        X, y = LeerFichero.mode2(args.modo2[0])
        perceptron = Perceptron(umbral=0.2, alpha=1)
        perceptron.train(X, y)
        perceptron.predecir(X, f_out)
    elif args.modo3:
        X_train, X_test, y_train, y_test = LeerFichero.mode3(args.modo3[0], args.modo3[1])
        perceptron = Perceptron(umbral=0.2, alpha=0.1)
        perceptron.train(X_train, y_train)
        perceptron.predecir(X_test, f_out)
    else:
        print("Error en los argumentos, necesita especificar algun modo de operacion.")
        exit(1)
    
    f_out.close() if args.f_out else None
