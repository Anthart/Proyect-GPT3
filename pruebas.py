from gpt3 import Gpt3
import operator
from collections import OrderedDict
import pandas as pd
from functools import reduce


probs = [{
        'easy': 0.9427575484820901,
        'very': 0.024555334023266522,
        'Easy': 0.0011065233935420998,
        'very easy': 0.0006448358179143308
    }]

print(Gpt3.strat_3("difficult", probs))





