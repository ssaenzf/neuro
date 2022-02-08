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
        if self.tipo == DIRECTO:
            self.valor_salida = self.valor_entrada
    
    def Propagar(self):
        for conexion in self.conexiones:
            conexion.propagar()
