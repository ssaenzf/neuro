class Conexion:
    def __init__(self, peso, neurona):
        self.peso = peso
        self.neurona = neurona
        self.valor = 0
        self.peso_anterior = None
    
    def liberar(self):
        pass

    def propagar(self, valor):    
        self.valor = valor
