import numpy as np


def ordenar_probs(dicc):
    tuples_sort = sorted(dicc.items(), key=lambda item: item[1], reverse=True)
    return {k: v for k, v in tuples_sort}


def pre_data_prob(dicc) -> dict:
    new_dicc = {}
    r = 0
    ignore = []
    for d in dicc:
        if d not in ignore:
            no_space = d.replace(" ", "")
            lista = list(dicc.keys())
            lista.remove(d)
            if no_space in lista:
                r = np.exp(dicc[no_space])
                ignore.append(no_space)
            val = np.exp(dicc[d])
            new_dicc[no_space] = val + r
            new_dicc = ordenar_probs(new_dicc)
            r = 0
    return new_dicc


def logprobs_to_percent(prob):
    new_prob = []
    for item in prob:
        pre_data_prob(item)
        new_prob.append(pre_data_prob(item))
    return new_prob


def logprob_to_prob(logprob):
    return np.exp(logprob)


def prob_for_label(label, logprobs):
    """
    Returns the predicted probability for the given label as
    a number between 0.0 and 1.0.
    """
    # Initialize probability for this label to zero.
    prob = 0.0
    # Look at the first entry in logprobs. This represents the
    # probabilities for the very next token.
    next_logprobs = logprobs[0]
    for s, logprob in next_logprobs.items():
        # We want labels to be considered case-insensitive. In
        # other words:
        #
        #     prob_for_label("vegetable") =
        #         prob("vegetable") + prob("Vegetable")
        #
        s = s.lower()
        if label.lower() == s:
            # If the prediction matches one of the labels, add
            # the probability to the total probability for that
            # label.
            prob += logprob
        elif label.lower().startswith(s):
            # If the prediction is a prefix of one of the labels, we
            # need to recur. Multiply the probability of the prefix
            # by the probability of the remaining part of the label.
            # In other words:
            #
            #     prob_for_label("vegetable") =
            #         prob("vege") * prob("table")
            #
            rest_of_label = label[len(s):]
            remaining_logprobs = logprobs[1:]
            prob += logprob * prob_for_label(
                rest_of_label,
                remaining_logprobs,
            )
    return prob


def parche_diff(probs):
    pre_new_probs = []
    for index, val in enumerate(probs):
        key = list(val.keys())
        if key[0] == "diff":
            pre_new_probs = probs[index:]
    new_probs = [{"difficult": prob_for_label("difficult", pre_new_probs), **pre_new_probs[0]}]
    return new_probs


def parche_very(probs, respuesta_gpt3):
    pre_new_probs = []
    for index, val in enumerate(probs):
        key = list(val.keys())
        if key[0] == "very":
            pre_new_probs = probs[index:]
    print(pre_new_probs)
    new_probs = [{respuesta_gpt3: prob_for_label(respuesta_gpt3, pre_new_probs), **pre_new_probs[0]}]
    return new_probs


def logprobs_display(logprobs):
    # probs = logprobs_to_percent(logprobs)
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
