from gpt3 import Gpt3
import operator
from collections import OrderedDict
import pandas as pd

lista_escalas = ['very easy', 'easy', 'neutral', 'difficult', 'very difficult']
min = [0, 0, 0.25, 0.5, 0.75]
max = [0, 0.25, 0.5, 0.75, 1]

data = pd.DataFrame({
    "escala": lista_escalas,
    "minimo": min,
    "maximo": max
})

lista = [
    {"easy": 0.50, "difficult": 0.45, "neutral": 0.05}
]


def strat_3(respuesta_gpt3, probs):
    valor_GP3 = 0

    return valor_GP3


valor = strat_3("easy", lista)
print(valor)
