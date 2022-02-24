class Conexion:
    def __init__(self, peso, neurona):
        self.peso = peso
        self.peso_anterior = peso
        self.neurona = neurona
        self.valor = 0
        
    
    def liberar(self):
        pass

    def propagar(self, valor):    
        self.valor = valor
