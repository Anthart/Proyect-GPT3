import os.path

from prompt_file import guardar_prompt
from prompt_file import cargar_lista_prompt
from tokens import calcular_total_pagar
import pandas as pd
import openai
from func import evaluar
from func import prob_for_label
from func import quitar_espacios

respuesta_gpt3 = evaluar("hello, how are")
pro_m = prob_for_label("you", respuesta_gpt3[0].logprobs.top_logprobs)

print(respuesta_gpt3[0].logprobs.top_logprobs[0])
print(respuesta_gpt3[0].text + " " + str(pro_m))