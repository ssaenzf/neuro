class Conexion:
    def __init__(self, peso, neurona, name=''):
        self.peso = peso
        self.peso_anterior = peso
        self.neurona = neurona
        self.valor = 0
        self.name = name

    def liberar(self):
        pass

    def propagar(self, valor):    
        self.valor = valor

    def __str__(self):
        return 'Conexion ' + self.name + ": " + str(self.peso)