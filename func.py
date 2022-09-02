import os
import openai
import sys
import json
import pickle
import time
from datetime import datetime
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
from transformers import GPT2Tokenizer

lista_escalas = ['very easy', 'easy', 'neutral', 'difficult', 'very difficult']
plantilla = "{:^5} {:^20} {:^20} {:^20} {:^20} {:^20} {:^20} {:^20}"


def imprimir_fila(indice, dframe, respuesta_gpt3, rango, complejidad_gpt3,
                  complejidad, complejidad_escala, comparacion):
    token = dframe["token"][indice]

    print(plantilla.format(indice, token, respuesta_gpt3, rango,
                           complejidad_gpt3, complejidad, complejidad_escala,
                           comparacion))


def asig_valor(valor):
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


def asig_medio(valor_escala):
    valor_medio = 0

    if valor_escala == lista_escalas[0]:
        valor_medio = 0
    elif valor_escala == lista_escalas[1]:
        valor_medio = (0.01 + 0.25) / 2
    elif valor_escala == lista_escalas[2]:
        valor_medio = (0.26 + 0.50) / 2
    elif valor_escala == lista_escalas[3]:
        valor_medio = (0.51 + 0.75) / 2
    elif valor_escala == lista_escalas[4]:
        valor_medio = (0.76 + 1) / 2

    return valor_medio


def asig_rango(escala):
    rango = ""

    if escala == "very easy":
        rango = "0"
    if escala == "easy":
        rango = "0.01 - 0.25"
    if escala == "neutral":
        rango = "0.26 - 0.50"
    if escala == "difficult":
        rango = "0.51 - 0.75"
    if escala == "very difficult":
        rango = "0.76 - 1"

    return rango


def promedio_valor_escala(name_file):
    diccionario = {}

    try:
        tf = open("promedio.json", "r")
        diccionario = json.load(tf)
        no_file = False
    except FileNotFoundError:
        no_file = True

    if no_file:
        dframe = pd.read_excel(name_file)
        for valor in lista_escalas:
            aux = dframe.loc[dframe["escala"] == valor]
            calculo = aux["complexity"].mean()
            diccionario[valor] = calculo
        tf = open("promedio.json", "w")
        json.dump(diccionario, tf)
        tf.close()

    return diccionario


def filtro(respuesta_gpt3):
    resultado = ""

    for valor_escala in lista_escalas:
        n_palabras = len(valor_escala.split())
        if n_palabras == 2 and respuesta_gpt3.count(valor_escala) >= 1:
            return valor_escala
        elif respuesta_gpt3.count(valor_escala) >= 1:
            resultado = valor_escala

    return resultado


def load_data():
    with open("temp/datos_temp.pkl", "rb") as tf:
        dicc = pickle.load(tf)

    minimo = dicc["minimo"]
    maximo = dicc["maximo"]
    nombre_archivo = dicc["archivo"]

    df = pd.read_csv(f"temp/{nombre_archivo}")

    return df, minimo, maximo


def temporal_storage(minimo, maximo, data):
    try:
        os.mkdir('temp')
    except OSError:
        print("Correpto !!\tCarpeta temp exite")

    now = datetime.now()
    nombre_archivo = f"{now.year}-{now.month}-{now.day}-{now.hour}-{now.second}.csv"
    data.to_csv(f"temp/{nombre_archivo}")

    dicc = {"minimo": minimo, "maximo": maximo, "archivo": nombre_archivo}

    with open("temp/datos_temp.pkl", "wb") as tf:
        pickle.dump(dicc, tf)


def guardar_metricas(metricas):
    file = 'resultados_metricas.xlsx'
    folder = 'resultados'
    path = f'{folder}/{file}'

    if not os.path.isdir(folder):
        os.mkdir(folder)

    if os.path.isfile(path):
        pandas_metrics = pd.read_excel(path, index_col=[0])
        pandas_metrics = pd.concat([pandas_metrics, metricas], ignore_index=True)
    else:
        pandas_metrics = metricas

    pandas_metrics.to_excel(path)


def evaluar(orden):
    openai.api_key = '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'
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=orden,
        temperature=0,
        max_tokens=5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=10,
        stop=["\n"]
    )
    respuesta = response.choices[0].text
    prob_tokens = response.choices[0].logprobs.top_logprobs
    return respuesta, prob_tokens


def quitar_espacios(dicc: dict[str, float]) -> dict:
    new_dicc = {}
    for d in dicc:
        new_dicc[d.replace(" ", "")] = dicc[d]
    return new_dicc


def logprob_to_prob(logprob: float) -> float:
    return np.exp(logprob)


def prob_for_label(label: str, logprobs: list[dict[str, float]]) -> float:
    prob = 0.0
    next_logprobs = quitar_espacios(logprobs[0])
    for s, logprob in next_logprobs.items():
        s = s.lower()
        if label.lower() == s:
            prob += logprob_to_prob(logprob)
        elif label.lower().startswith(s):
            rest_of_label = label[len(s):]
            remaining_logprobs = logprobs[1:]
            prob += logprob * prob_for_label(
                rest_of_label,
                remaining_logprobs,
            )
    return prob


def palabras_complejas(dframe, orden, dic_escalas, version=False, save_result=False, load=None):
    if load is None:
        resultado = dframe
        resultado["Respuesta GPT3"] = None
        resultado["Rango GPT3"] = None
        resultado["Complejidad GPT3"] = 0.0
        resultado["comparacion"] = None
    elif load is not None:
        resultado = load
        frame = [resultado, dframe.loc[:]]
        resultado = pd.concat(frame)
        resultado = resultado.loc[:, ~resultado.columns.str.contains("Unnamed")]
    else:
        sys.exit("Error de ingreso de parametros")

    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    last = time.time()
    # tokens_prompt = 0
    peticiones = 0

    print(plantilla.format("N", "Token", "Respuesta GPT3", "Rango GPT3", "Complejidad GPT3",
                           "Complejidad compLex", "Rango compLex", "Comparacion") + "\n")

    for indice in dframe.index:
        temp = orden
        temp = temp.replace("@recurso", "\"" + dframe["source"][indice] + "\"")
        temp = temp.replace("@oracion", "\"" + dframe["sentence"][indice] + "\"")
        temp = temp.replace("@aEvaluar", "\"" + dframe["token"][indice] + "\"")

        try:
            respuesta_gpt3, prob_tokens = evaluar(temp)
        except openai.error.RateLimitError as limit_rate_error:
            if indice - 1 != -1:
                temporal_storage(indice, dframe.tail(1).index[0], resultado.loc[0:indice - 1])
            sys.exit(str(limit_rate_error))
        except openai.error.OpenAIError as error_openai:
            if indice - 1 != -1:
                temporal_storage(indice, dframe.tail(1).index[0], resultado.loc[0:indice - 1])
            sys.exit(str(error_openai))

        try:
            respuesta_gpt3 = filtro(respuesta_gpt3)
            prob = prob_for_label(respuesta_gpt3, prob_tokens)
            if respuesta_gpt3 == "":
                raise KeyError
        except KeyError:
            temporal_storage(indice, dframe.tail(1).index[0], resultado.loc[0:indice - 1])
            sys.exit("No se encontro el resultado esperado"
                     " por GPT3")

        # tokens_prompt += len(tokenizer(temp)['input_ids'])
        peticiones += 1

        rango = asig_rango(respuesta_gpt3)
        complejidad_gpt3 = round(dic_escalas[respuesta_gpt3], 15)
        complejidad = dframe["complexity"][indice]
        escala_complex = dframe["escala"][indice]

        resultado.at[indice, "Respuesta GPT3"] = respuesta_gpt3
        resultado.at[indice, "Rango GPT3"] = rango
        resultado.at[indice, "Complejidad GPT3"] = complejidad_gpt3

        if respuesta_gpt3 == escala_complex:
            comparacion = "Si"
        else:
            comparacion = "No"

        resultado.at[indice, "comparacion"] = comparacion

        imprimir_fila(indice, dframe, respuesta_gpt3, rango, complejidad_gpt3,
                      complejidad, escala_complex, comparacion)

        # ****************************** Control de Peticiones ***********************************
        actual = time.time() - last

        if actual >= 60:
            tokens_prompt = 0
            peticiones = 0
            last = time.time()

        # if tokens_prompt >= 150000 or peticiones >= 55:
        if peticiones >= 55:
            seconds_to_wait = 60 - actual
            # tokens_prompt = 0
            peticiones = 0
            time.sleep(seconds_to_wait)
            last = time.time()
        # ************************* Cierre de control de Peticiones ******************************

    true = resultado.loc[:, "complexity"]
    predicted = resultado.loc[:, "Complejidad GPT3"]

    mae = round(mean_absolute_error(true, predicted), 4)
    mse = round(mean_squared_error(true, predicted), 4)
    rmse = round(mean_squared_error(true, predicted, squared=False), 4)
    r2 = round(r2_score(true, predicted), 4)
    pearson = round(true.corr(predicted, method='pearson'), 4)
    spearman = round(true.corr(predicted, method='spearman'), 4)

    print("MAE: " + str(mae))
    print("MSE: " + str(mse))
    print("RMSE: " + str(rmse))
    print("R2: " + str(r2))
    print("Pearson: " + str(pearson))
    print("Spearman: " + str(spearman))
    print("\n")

    resultado = resultado[["id", "sentence", "token", "Respuesta GPT3", "Rango GPT3", "Complejidad GPT3",
                           "complexity", "escala", "comparacion"]]

    resultado["MAE"] = mae
    resultado["MSE"] = mse
    resultado["RMSE"] = rmse
    resultado["R2"] = r2
    resultado["Pearson"] = pearson
    resultado["Sperman"] = spearman

    if version:
        resultado_metricas = {"Version": [version], "MAE": [mae], "MSE": [mse],
                              "RMSE": [rmse], "R2": [r2],
                              "Pearson": [pearson], "Spearman": [spearman]}
        resultado_metricas = pd.DataFrame(resultado_metricas)
        guardar_metricas(resultado_metricas)

    if save_result:
        if version:
            resultado.to_excel(f'resultados/resultado_{version}.xlsx')
        else:
            resultado.to_excel('resultados/resultado_version 12.xlsx')
