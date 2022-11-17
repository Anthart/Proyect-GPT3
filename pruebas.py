from gpt3 import Gpt3
import operator
from collections import OrderedDict
import pandas as pd

__rango_escala = {
    'very easy': (0, 0),
    'easy': (0.1, 0.25),
    'neutral': (0.26, 0.50),
    'difficult': (0.51, 0.75),
    'very difficult': (0.76, 1)
}

print(list(__rango_escala.keys()))
