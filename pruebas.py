import os.path
import operator
from prompt_file import guardar_prompt
from prompt_file import cargar_lista_prompt
from tokens import calcular_total_pagar
import pandas as pd
import openai
from func import evaluar
from func import logprobs_to_percent
from func import logprobs_display
from func import prob_for_label
from func import pre_data_prob

label = "very easy"
respuesta_gpt3, prob = evaluar("hello, how are you?")
# pro_m = prob_for_label("you", respuesta_gpt3[0].logprobs.top_logprobs)

# for item in prob:
#     for dic in item:
#         item[dic] = round(logprob_to_prob(item[dic]) * 100, 2)

print(logprobs_display(prob[0:2]))

# dic = {
#     "\n": -0.0017886201,
#     "\n\n": -7.7011003,
#     " ": -8.931325,
#     "  ": -9.086402,
#     "                ": -8.658414
# }
# dic = pre_data_prob(dic)
# dic_sort = sorted(dic.items(), key=operator.itemgetter(1), reverse=True)
# dic_aux = {}
# for item in dic_sort:
#     dic_aux[item[0]] = item[1]
# print(dic_aux)
