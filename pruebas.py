from gpt3 import Gpt3
import operator
from collections import OrderedDict


__escala_limite = {
    "very easy": [0],
    "easy": [0.1, 0.25],
    "neutral": [0.26, 0.50],
    "difficult": [0.51, 0.75],
    "very difficult": [0.76, 1]
}

__lista_escalas = ["very easy", "easy", "neutral", "difficult", "very difficult"]


def __asig_valor(valor):
    escala = ""

    if valor == 0:
        escala = "very easy"
    elif 0 < valor <= 0.25:
        escala = "easy"
    elif 0.25 < valor <= 0.50:
        escala = "neutral"
    elif 0.50 < valor <= 0.75:
        escala = "difficult"
    elif 0.75 < valor <= 1:
        escala = "very difficult"

    return escala

def __asig_valor2(valor):
    escala = ""

    for key, item in __escala_limite.items():
        if len(item) == 1:
            escala = key
        elif item[0] <= valor <= item[1]:
            escala = key

    return escala

valor = 0.76
print(__asig_valor(valor))
print(__asig_valor2(valor))

