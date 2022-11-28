from gpt3 import Gpt3
import operator
from collections import OrderedDict
import pandas as pd
from functools import reduce

dicc_puntos = {"very easy": 0, "easy": 0.25, "neutral": 0.5, "difficult": 0.75, "very difficult": 1}
list_keys = list(dicc_puntos.keys())

print(list(dicc_puntos.items()))


