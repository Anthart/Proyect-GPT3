import numpy as np


def ordenar_probs(dicc: dict[str, float]):
    tuples_sort = sorted(dicc.items(), key=lambda item: item[1], reverse=True)
    return {k: v for k, v in tuples_sort}


def pre_data_prob(dicc: dict[str, float]) -> dict:
    new_dicc = {}
    r = 0
    for d in dicc:
        no_space = d.replace(" ", "")
        lista = list(dicc.keys())
        lista.remove(d)
        if no_space in lista:
            r = np.exp(dicc[no_space])
        val = np.exp(dicc[d])
        new_dicc[no_space] = val + r
        new_dicc = ordenar_probs(new_dicc)
    return new_dicc


def logprobs_to_percent(prob: list[dict[str, float]]):
    new_prob = []
    for item in prob:
        pre_data_prob(item)
        new_prob.append(pre_data_prob(item))
    return new_prob


def logprobs_display(logprobs: list[dict[str, float]]) -> list:
    #probs = logprobs_to_percent(logprobs)
    lista = ["", "", "", "", ""]
    size = len(logprobs)
    count = 5
    for i in range(size):
        items = list(logprobs[i].items())
        text = "" if i == 0 else ","
        for j in range(count):
            if j <= len(items) - 1:
                label = items[j][0]
                weight = round(items[j][1] * 100, 2)
                lista[j] = lista[j] + text + str(label) + ":" + str(weight) + "%"
            else:
                lista[j] = lista[j] + text + "None"
    return lista
