import os.path

from prompt_file import guardar_prompt
from prompt_file import cargar_lista_prompt
from tokens import calcular_total_pagar
import pandas as pd
import openai
from func import evaluar
from func import logprob_to_prob
from func import prob_for_label
from func import pre_data_prob
label = "very easy"
respuesta_gpt3, prob = evaluar("hello, how are")
# pro_m = prob_for_label("you", respuesta_gpt3[0].logprobs.top_logprobs)

# for item in prob:
#     for dic in item:
#         item[dic] = round(logprob_to_prob(item[dic]) * 100, 2)


new_prob = []
for i in range(2):
    item = prob[i]
    pre_data_prob(item)
    new_prob.append(pre_data_prob(item))
print(new_prob)

# prueba = {"nu": 3}
# if "nu" in prueba.keys():
#     pri = "correcto"
# else:
#     pri = "No existe"
# print(pri)
#
# print(respuesta_gpt3[0].logprobs.top_logprobs[0])
# print(respuesta_gpt3[0].text + " " + str(pro_m))
#
# print(prob[0])
# print(prob[1:])
# print(respuesta_gpt3)
# rest_label = ""
# if label.lower().startswith("very"):
#     rest_label = label[len("very"):]
#     rest_label = rest_label.replace(" ", "")
# if rest_label.lower().startswith("easy"):
#     print("very" + rest_label)
# print(rest_label.replace(" ",""))

