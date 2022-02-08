class Conexion:
    def __init__(self, peso, neurona):
        self.peso = peso
        self.neurona = neurona
    
    def Liberar(self):
        pass

    def Propagar(self, valor):    
        self.valor = valor
        self.neurona.valor_entrada = self.valor * self.peso
        