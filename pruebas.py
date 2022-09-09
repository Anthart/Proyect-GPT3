from gpt3 import Gpt3
import operator
from collections import OrderedDict


def ordenar_probs(dicc: dict[str, float]):
    tuples_sort = sorted(dicc.items(), key=lambda item: item[1], reverse=True)
    return {k: v for k, v in tuples_sort}


dic = {"queso": 19, "leche": 5, "papa": 60}
dic = ordenar_probs(dic)
print(dic)
